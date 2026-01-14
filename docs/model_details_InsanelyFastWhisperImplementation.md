# Model Details: InsanelyFastWhisperImplementation

This document details the model loading, download behavior, and execution flow for the `InsanelyFastWhisperImplementation` in the mac-whisper-speedtest project.

---

## File Reference Legend

| Tag | Meaning |
|-----|---------|
| **[PROJECT]** | Files in this repository (`src/mac_whisper_speedtest/`) |
| **[LIBRARY]** | Files in `.venv/lib/python3.12/site-packages/` (installed packages) |

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| What library performs transcription? | HuggingFace `transformers` pipeline | [PROJECT] `insanely.py:81` imports `pipeline` |
| What model repo is used for "small"? | `openai/whisper-small` | [PROJECT] `insanely.py:150` model mapping |
| What model repo is used for "large"? | `openai/whisper-large-v3-turbo` | [PROJECT] `insanely.py:152` model mapping |
| Where are models cached? | `~/.cache/huggingface/hub/models--openai--whisper-{model}/` | Empirical test observation |
| What handles the download? | `huggingface_hub` library (via transformers) | [LIBRARY] `transformers/pipelines/base.py` |
| Does it use GPU acceleration? | Yes, MPS on Apple Silicon | [PROJECT] `insanely.py:21` device detection |
| Does 4-bit quantization work? | **No** - `bitsandbytes` not installed | Empirical test warning |

---

## Benchmark Execution Flow

### Command: Small Model
```bash
.venv/bin/python3 test_benchmark2.py small 1 InsanelyFastWhisperImplementation
```

### Command: Large Model
```bash
.venv/bin/python3 test_benchmark2.py large 1 InsanelyFastWhisperImplementation
```

### Execution Steps

1. **Entry Point** - `test_benchmark2.py:88-94`
   - Parses command line: `model="small"`, `runs=1`, `implementations="InsanelyFastWhisperImplementation"`
   - Loads audio from `tests/jfk.wav`

2. **Benchmark Setup** - `benchmark.py:110-134`
   - Creates `InsanelyFastWhisperImplementation()` instance
   - Calls `implementation.load_model("small")` (NOT timed)

3. **Implementation Init** - `insanely.py:18-28`
   - Sets `device_id = "mps"` on macOS
   - Calculates `batch_size` based on available RAM (via `psutil`)
   - Sets `compute_type = "float16"`, `quantization = "4bit"`

4. **Model Loading** - `insanely.py:72-142`
   - Maps `"small"` → `"openai/whisper-small"` via `_map_model_name()`
   - Maps `"large"` → `"openai/whisper-large-v3-turbo"`
   - Imports `torch` and `transformers.pipelines.pipeline`
   - Attempts to configure 4-bit quantization (fails if `bitsandbytes` missing)
   - Creates pipeline: `pipeline("automatic-speech-recognition", model="openai/whisper-small", ...)`

5. **Model Download** (if not cached) - `[LIBRARY] transformers/pipelines/__init__.py`
   - Pipeline internally calls `AutoModel.from_pretrained()`
   - Uses `huggingface_hub` to download model files
   - Downloads to `~/.cache/huggingface/hub/models--openai--whisper-small/`

6. **Transcription** - `insanely.py:158-225`
   - Writes audio to temp WAV file (`tempfile.NamedTemporaryFile`)
   - Calls `self._model(temp_file.name, chunk_length_s=20, batch_size=N, ...)`
   - Returns `TranscriptionResult(text, segments, language)`

7. **Timing & Results** - `benchmark.py:143-171`
   - Measures transcription time (excludes model loading)
   - Averages across runs
   - Collects params via `implementation.get_params()`

---

## Summary Table

| Aspect | Small Model | Large Model |
|--------|-------------|-------------|
| **Requested Model** | `small` | `large` |
| **Mapped Model Name** | `openai/whisper-small` | `openai/whisper-large-v3-turbo` |
| **HuggingFace Repo URL** | https://huggingface.co/openai/whisper-small | https://huggingface.co/openai/whisper-large-v3-turbo |
| **Cache Location** | `~/.cache/huggingface/hub/models--openai--whisper-small/` | `~/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/` |
| **Total Size on Disk** | 926 MB | 1.5 GB |
| **Main Weights File** | 922 MB | 1.5 GB |
| **First Download Time** | ~166 seconds | ~363 seconds |
| **Cached Load Time** | ~2 seconds | ~1 second |
| **Transcription Time** | ~0.96 seconds | ~2.43 seconds |

---

## Model Mapping Reference

From `insanely.py:144-156`:

```python
model_map = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3-turbo",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}
# Fallback: f"openai/whisper-{model_name}"
```

**Notable**: Unlike some other implementations, `"large"` maps to the **turbo** variant (`large-v3-turbo`) by default, which is smaller (~800M params) than full `large-v3` (~1.5B params).

---

## Notes

### Apple Silicon Optimizations
This implementation includes several Apple Silicon-specific optimizations:

1. **SDPA Attention** (`insanely.py:106-108`): Uses Scaled Dot Product Attention instead of Flash Attention for better MPS compatibility
2. **Dynamic Batch Sizing** (`insanely.py:30-70`): Adjusts batch size based on available memory
3. **Reduced Chunk Length** (`insanely.py:180`): Uses 20s chunks on MPS vs 30s on other devices
4. **KV Cache & Low CPU Memory** (`insanely.py:119-120`): Enabled for unified memory architecture

### 4-bit Quantization (Non-Functional)
The code attempts to enable 4-bit quantization via `BitsAndBytesConfig`:
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

However, **this fails silently** because:
1. `bitsandbytes` package is not installed in the project
2. `bitsandbytes` has limited macOS/MPS support
3. The warning is logged but `quantization=4bit` still appears in `get_params()` output

### File-Based Processing
Unlike some pure-Python implementations that pass numpy arrays directly, this implementation:
1. Writes audio to a temp WAV file
2. Passes the file path to the transformers pipeline
3. The pipeline handles file reading internally

---

## Key Source Files

| File | Purpose |
|------|---------|
| `[PROJECT] src/mac_whisper_speedtest/implementations/insanely.py` | Main implementation |
| `[PROJECT] test_benchmark2.py` | Non-interactive benchmark runner |
| `[PROJECT] src/mac_whisper_speedtest/benchmark.py` | Benchmark orchestration |
| `[LIBRARY] transformers/pipelines/__init__.py` | Pipeline factory |
| `[LIBRARY] transformers/pipelines/automatic_speech_recognition.py` | ASR pipeline |

---

## Empirical Test Results

### Test Date: 2026-01-12

### Small Model Tests

#### Fresh Download Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 InsanelyFastWhisperImplementation
```

**Terminal Output:**
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - InsanelyFastWhisperImplementation

Starting benchmark with model 'small' (1 run(s))...
2026-01-12 02:11:21 [info     ] Benchmarking InsanelyFastWhisperImplementation with model small
2026-01-12 02:11:21 [info     ] Apple Silicon detected: 6.7GB available, using batch_size=16
2026-01-12 02:11:21 [info     ] Loading model for InsanelyFastWhisperImplementation
2026-01-12 02:11:28 [info     ] Loading Insanely Fast Whisper model openai/whisper-small
2026-01-12 02:11:28 [info     ] Using SDPA attention implementation (optimized for Apple Silicon MPS)
2026-01-12 02:11:28 [warning  ] Failed to configure 4-bit quantization: No package metadata was found for bitsandbytes
2026-01-12 02:14:14 [info     ] Run 1/1 for InsanelyFastWhisperImplementation
2026-01-12 02:14:14 [info     ] Transcribing with Insanely Fast Whisper using model openai/whisper-small
2026-01-12 02:14:14 [info     ] Using automatic language detection
2026-01-12 02:14:15 [info     ] Run 1 completed in 1.2111 seconds
2026-01-12 02:14:15 [info     ] Transcription:  And so my fellow Americans, ask not what your cou...
2026-01-12 02:14:15 [info     ] Average time for InsanelyFastWhisperImplementation: 1.2111 seconds

=== Benchmark Summary for 'small' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
insanely-fast-whisper  1.2111          model=whisper-small, device_id=mps, batch_size=16, compute_type=float16, quantization=4bit
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."
```

**Cache Verification:**
```bash
$ ls -la ~/.cache/huggingface/hub/ | grep whisper
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 02:14 models--openai--whisper-small

$ du -sh ~/.cache/huggingface/hub/models--openai--whisper-small/
926M    /Users/rymalia/.cache/huggingface/hub/models--openai--whisper-small/

$ ls -lah ~/.cache/huggingface/hub/models--openai--whisper-small/blobs/
total 1897240
-rw-r--r--@  1 rymalia  staff   922M Jan 12 02:14 1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b
-rw-r--r--@  1 rymalia  staff   2.4M Jan 12 02:14 1e95340ff836fad1b5932e800fb7b8c5e6d78a74
-rw-r--r--@  1 rymalia  staff   482K Jan 12 02:14 6038932a2a1f09a66991b1c2adae0d14066fa29e
... (additional config/tokenizer files)
```

**Observations:**
- Download time: ~166 seconds (02:11:28 → 02:14:14)
- Model load + download: 922 MB main weights file
- Transcription: 1.2111 seconds
- 4-bit quantization failed (expected - bitsandbytes not installed)

#### Cached Run Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 InsanelyFastWhisperImplementation
```

**Terminal Output:**
```
2026-01-12 02:20:59 [info     ] Apple Silicon detected: 8.2GB available, using batch_size=10
2026-01-12 02:21:01 [info     ] Loading Insanely Fast Whisper model openai/whisper-small
2026-01-12 02:21:01 [warning  ] Failed to configure 4-bit quantization: No package metadata was found for bitsandbytes
2026-01-12 02:21:03 [info     ] Run 1/1 for InsanelyFastWhisperImplementation
2026-01-12 02:21:04 [info     ] Run 1 completed in 0.9583 seconds

insanely-fast-whisper  0.9583          model=whisper-small, device_id=mps, batch_size=10, ...
```

**Observations:**
- Model load from cache: ~2 seconds (vs 166s fresh download)
- Transcription: 0.9583 seconds
- Batch size changed to 10 due to different available memory (8.2GB vs 6.7GB)

---

### Large Model Tests

#### Fresh Download Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 InsanelyFastWhisperImplementation
```

**Terminal Output:**
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - InsanelyFastWhisperImplementation

Starting benchmark with model 'large' (1 run(s))...
2026-01-12 02:14:40 [info     ] Benchmarking InsanelyFastWhisperImplementation with model large
2026-01-12 02:14:40 [info     ] Apple Silicon detected: 7.3GB available, using batch_size=16
2026-01-12 02:14:40 [info     ] Loading model for InsanelyFastWhisperImplementation
2026-01-12 02:14:42 [info     ] Loading Insanely Fast Whisper model openai/whisper-large-v3-turbo
2026-01-12 02:14:42 [info     ] Using SDPA attention implementation (optimized for Apple Silicon MPS)
2026-01-12 02:14:42 [warning  ] Failed to configure 4-bit quantization: No package metadata was found for bitsandbytes
2026-01-12 02:20:45 [info     ] Run 1/1 for InsanelyFastWhisperImplementation
2026-01-12 02:20:45 [info     ] Transcribing with Insanely Fast Whisper using model openai/whisper-large-v3-turbo
2026-01-12 02:20:45 [info     ] Using automatic language detection
2026-01-12 02:20:47 [info     ] Run 1 completed in 2.5331 seconds
2026-01-12 02:20:47 [info     ] Transcription:  And so, my fellow Americans, ask not what your co...
2026-01-12 02:20:47 [info     ] Average time for InsanelyFastWhisperImplementation: 2.5331 seconds

=== Benchmark Summary for 'large' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
insanely-fast-whisper  2.5331          model=whisper-large-v3-turbo, device_id=mps, batch_size=16, compute_type=float16, quantization=4bit
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your..."
```

**Cache Verification:**
```bash
$ ls -la ~/.cache/huggingface/hub/ | grep whisper
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 02:20 models--openai--whisper-large-v3-turbo
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 02:14 models--openai--whisper-small

$ du -sh ~/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/
1.5G    /Users/rymalia/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/

$ ls -lah ~/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/blobs/
total 3168880
-rw-r--r--@  1 rymalia  staff   1.5G Jan 12 02:20 542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1
-rw-r--r--@  1 rymalia  staff   2.6M Jan 12 02:20 17456db595adc78a973f97d69d8cb50bc87c0b1c
... (additional config/tokenizer files)
```

**Observations:**
- Download time: ~363 seconds (02:14:42 → 02:20:45) - about 6 minutes
- Model weights: 1.5 GB
- Transcription: 2.5331 seconds
- **No timeout issues** - download completed within Python's default timeouts

#### Cached Run Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 InsanelyFastWhisperImplementation
```

**Terminal Output:**
```
2026-01-12 02:21:11 [info     ] Apple Silicon detected: 7.7GB available, using batch_size=16
2026-01-12 02:21:13 [info     ] Loading Insanely Fast Whisper model openai/whisper-large-v3-turbo
2026-01-12 02:21:13 [warning  ] Failed to configure 4-bit quantization: No package metadata was found for bitsandbytes
2026-01-12 02:21:14 [info     ] Run 1/1 for InsanelyFastWhisperImplementation
2026-01-12 02:21:16 [info     ] Run 1 completed in 2.4306 seconds

insanely-fast-whisper  2.4306          model=whisper-large-v3-turbo, device_id=mps, batch_size=16, ...
```

**Observations:**
- Model load from cache: ~1 second (vs 363s fresh download)
- Transcription: 2.4306 seconds (consistent with fresh run)

---

## Known Issues / Conflicts Discovered

### 1. Misleading 4-bit Quantization Parameter

**Problem**: The benchmark output shows `quantization=4bit` but quantization is NOT active.

**Evidence**:
```
[warning  ] Failed to configure 4-bit quantization: No package metadata was found for bitsandbytes
...
insanely-fast-whisper  1.2111  model=whisper-small, ..., quantization=4bit
```

**Impact**: Users may believe they're getting quantized inference when they're not.

### 2. Batch Size Logic Inconsistency

**Problem**: The batch size selection for Apple Silicon has non-intuitive thresholds:

```python
# insanely.py:43-52
if available_memory_gb >= 32:
    batch_size = 16  # High-end
elif available_memory_gb >= 16:
    batch_size = 12  # Mid-range
elif available_memory_gb >= 8:
    batch_size = 10  # Base
elif available_memory_gb >= 4:
    batch_size = 16  # ← Larger than 8-16GB range?
else:
    batch_size = 12  # ← Larger than 8-16GB range?
```

**Observation**: With 6.7GB available, got batch_size=16. With 8.2GB, got batch_size=10.

### 3. Deprecation Warning

**Warning**:
```
FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
```

**Impact**: Minor - will need update when transformers removes deprecated API.

### 4. Forced Decoder IDs Conflict Warning

**Warning**:
```
You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50359]] which creates a conflict.
```

**Impact**: The library resolves this automatically, but the warning is confusing.

---

## Recommended Improvements

### Improvement 1: Fix Misleading Quantization Parameter

**Problem**: `get_params()` returns `quantization=4bit` even when quantization is not active.

**Impact**: Misleading benchmark output; users think they're getting quantized inference.

**Location**: `insanely.py:227-235`

**Recommended Fix** (~10 lines):

```python
def __init__(self):
    # ... existing code ...
    self.quantization = "4bit"  # Desired
    self._quantization_active = False  # Track actual state

def load_model(self, model_name: str) -> None:
    # ... in the quantization config section ...
    if self.quantization == "4bit" and BitsAndBytesConfig is not None:
        try:
            # ... existing config code ...
            self._quantization_active = True
            self.log.info("Using 4-bit quantization with BitsAndBytesConfig")
        except Exception as e:
            self._quantization_active = False
            self.log.warning(f"Failed to configure 4-bit quantization: {e}")

def get_params(self) -> Dict[str, Any]:
    return {
        "model": self.model_name,
        "device_id": self.device_id,
        "batch_size": self.batch_size,
        "compute_type": self.compute_type,
        "quantization": self.quantization if self._quantization_active else "none",
    }
```

**Effort**: ~10 lines
**Priority**: P2 (Medium) - Misleading output

---

### Improvement 2: Fix Batch Size Logic

**Problem**: Batch size thresholds are non-monotonic and confusing.

**Impact**: Suboptimal performance, confusing behavior (more memory can mean smaller batch size).

**Location**: `insanely.py:43-52`

**Recommended Fix** (~15 lines):

```python
def _get_optimal_batch_size(self) -> int:
    try:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if platform.system() == "Darwin" and self.device_id == "mps":
            # Apple Silicon: monotonically increasing batch sizes
            if available_memory_gb >= 32:
                batch_size = 24
            elif available_memory_gb >= 16:
                batch_size = 16
            elif available_memory_gb >= 8:
                batch_size = 12
            elif available_memory_gb >= 4:
                batch_size = 8
            else:
                batch_size = 4
            self.log.info(f"Apple Silicon: {available_memory_gb:.1f}GB available, batch_size={batch_size}")
        else:
            # Non-Apple (unchanged)
            ...
```

**Effort**: ~15 lines
**Priority**: P3 (Low) - Works but confusing

---

### Improvement 3: Add Download Progress Indication

**Problem**: No feedback during 6+ minute model downloads.

**Impact**: Users don't know if download is progressing or stuck.

**Location**: `insanely.py:136-142`

**Recommended Fix** (~5 lines):

```python
def load_model(self, model_name: str) -> None:
    # ... existing code ...
    self.log.info(f"Loading Insanely Fast Whisper model {self.model_name}")
    self.log.info("Note: First run will download model files (may take several minutes)")

    # Pipeline creation triggers download
    self._model = pipeline(...)
    self.log.info("Model loaded successfully")
```

**Effort**: ~5 lines
**Priority**: P3 (Low) - Improves UX but transformers shows its own progress

---

### Improvement 4: Suppress or Fix Deprecation Warnings

**Problem**: Noisy warnings in output.

**Impact**: Cluttered benchmark output.

**Location**: External to implementation (transformers library)

**Recommended Fix** (~5 lines in transcribe method):

```python
import warnings

async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    # ... existing code ...
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*input name.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
        result = self._model(temp_file.name, ...)
```

**Effort**: ~5 lines
**Priority**: P3 (Low) - Cosmetic

---

## Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| P2 | Fix misleading quantization parameter | ~10 lines | Users get accurate params | Not started |
| P3 | Fix batch size logic inconsistency | ~15 lines | More intuitive behavior | Not started |
| P3 | Add download progress indication | ~5 lines | Better UX on first run | Not started |
| P3 | Suppress deprecation warnings | ~5 lines | Cleaner output | Not started |

---

## Implementation Order Recommendation

### Phase 1: Quick Wins (P2)
- [ ] Fix misleading quantization parameter in `get_params()`

### Phase 2: Polish (P3)
- [ ] Fix batch size logic to be monotonically increasing
- [ ] Add informational log message about potential download time
- [ ] Suppress noisy deprecation warnings

---

## Comparison with Other Implementations

| Aspect | InsanelyFastWhisper | MLX-Whisper | WhisperKit |
|--------|---------------------|-------------|------------|
| **Cache Location** | HF cache (shared) | HF cache (shared) | Custom (~Library) |
| **Download Library** | huggingface_hub | huggingface_hub | Swift URLSession |
| **GPU Backend** | MPS (PyTorch) | MLX | CoreML |
| **First-Run Time (large)** | ~6 min | ~3 min | ~27 min |
| **Cached Load Time** | ~1-2 sec | ~2 sec | ~7 sec |
| **Timeout Issues** | None observed | None | Critical (P0) |

**Key Advantage**: Uses standard HuggingFace cache, making models shareable across projects.
