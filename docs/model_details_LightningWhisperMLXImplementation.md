# Model Details: LightningWhisperMLXImplementation

This document traces the complete execution flow for `LightningWhisperMLXImplementation`, documenting model download behavior, caching, the ModelHolder pattern, MLX optimizations, and critical issues discovered through empirical testing.

---

## File Reference Legend

Throughout this document, files are categorized as:
- **[PROJECT]** - Files in this repository that you can modify
- **[LIBRARY]** - Installed package files in `.venv/lib/python3.12/site-packages/` (modify at your own risk)

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| Does it re-download on every call? | **NO** - uses `hf_hub_download()` with local caching | lightning.py:87-88, empirical verification |
| Where are models downloaded? | `./mlx_models/{model_name}/` (relative to CWD) | lightning.py:85, empirical: small=459MB, large-v3=2.9GB |
| Does it check HF Hub cache first? | **NO** - downloads directly to local folder, ignores `~/.cache/huggingface/hub/` | lightning.py:85, empirical verification |
| Is download folder same as runtime? | **YES** - `./mlx_models/{model_name}/` for both | lightning.py:85,91 |
| Are files converted after download? | **NO** - `weights.npz` and `config.json` used as-is | load_models.py:22-29 |
| Partial download: resume or restart? | **RESUME** - hf_xet tracks chunks, continues on restart | Empirical: killed download, restarted, completed |
| Complete files: skip download? | **YES** - hf_hub_download checks metadata | Empirical: second run loaded in ~1 second |
| Completeness check method? | **SHA256 checksum** in `.incomplete` filename | Empirical: `4MXqmwCazYcr...05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d.incomplete` |
| Uses HF Hub API? | **YES** - `hf_hub_download()` with `hf_xet` Rust extension | lightning.py:2,87-88 |
| Timeout sufficient for first-run? | **POTENTIALLY NO** - hf_xet hung during testing (possibly related to outdated packages) | Empirical: 30+ min hang, required kill and restart |
| Uses fallback chains? | **NO** - single model lookup, fails if not found | lightning.py:59-60 |
| Uses `~/Library/Application Support/`? | **NO** - only uses `./mlx_models/` | Code analysis |
| CPU or GPU based? | **GPU** - MLX uses Apple Silicon GPU via Metal | whisper.py, MLX framework design |
| Apple Silicon optimizations? | **YES** - MLX is specifically designed for Apple Silicon | Unified memory, Metal acceleration |
| SDPA relevance? | **NO** - SDPA is PyTorch; MLX has its own attention | whisper.py:73-88 manual attention implementation |
| Quantization support? | **YES** - 4-bit and 8-bit available, but project uses `None` | lightning.py:4-52, project lightning.py:23 |
| Distil model support? | **YES** - 4 variants, all map to same repo | lightning.py:15-16,28-29,41-42,49-50 |
| What is ModelHolder? | **Singleton pattern** - caches loaded model in memory | transcribe.py:51-60 |

---

## Benchmark Execution Flow

### Command: Small Model
```bash
.venv/bin/python3 test_benchmark2.py small 1 LightningWhisperMLXImplementation
```

### Command: Large Model
```bash
.venv/bin/python3 test_benchmark2.py large 1 LightningWhisperMLXImplementation
```

### Step-by-Step Execution

1. **Entry Point** — **[PROJECT]** `test_benchmark2.py:88-94`
   - Parses CLI args: `model="small"` or `model="large"`, `runs=1`, `implementations="LightningWhisperMLXImplementation"`
   - Calls `asyncio.run(main(model, runs, implementations))`

2. **Audio Loading** — **[PROJECT]** `test_benchmark2.py:36-55`
   - Loads `tests/jfk.wav` via soundfile
   - Converts to mono if stereo, resamples to 16kHz
   - Audio is float32 normalized to [-1, 1]

3. **Implementation Filtering** — **[PROJECT]** `test_benchmark2.py:58-74`
   - Filters to only `LightningWhisperMLXImplementation`

4. **Benchmark Runner** — **[PROJECT]** `benchmark.py:110-190`
   - Creates instance: `implementation = LightningWhisperMLXImplementation()`

5. **Implementation `__init__`** — **[PROJECT]** `lightning.py:15-28`
   - Sets `batch_size=12`, `quant=None`, `language=None`
   - Validates macOS platform (MLX requires Apple Silicon)

6. **Model Loading** — **[PROJECT]** `benchmark.py:134`
   ```python
   implementation.load_model(config.model_name)  # "small" or "large"
   ```

7. **`load_model()` Method** — **[PROJECT]** `lightning.py:46-78`
   - Imports `LightningWhisperMLX` from `lightning_whisper_mlx`
   - Calls `_map_model_name()`:
     - `"small"` → `"small"` (unchanged)
     - `"large"` → `"large-v3"` (upgraded to latest)
   - Creates: `LightningWhisperMLX(model=mapped_name, batch_size=12, quant=None)`

8. **Library Initialization** — **[LIBRARY]** `lightning_whisper_mlx/lightning.py:54-88`
   - Validates model name exists in `models` dict
   - Looks up HuggingFace repo: `models[name]["base"]`
   - Sets `local_dir = f"./mlx_models/{name}"`
   - **Downloads via `hf_hub_download()`:**
     ```python
     hf_hub_download(repo_id=repo_id, filename="weights.npz", local_dir=local_dir)
     hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=local_dir)
     ```

9. **Transcription** — **[PROJECT]** `lightning.py:84-154`
   - Writes audio to temp WAV file at 16kHz
   - Calls `self.whisper_model.transcribe(audio_path=temp_path, language=None)`
   - Cleans up temp file

10. **Library Transcription** — **[LIBRARY]** `lightning.py:90-91`
    ```python
    result = transcribe_audio(audio_path, path_or_hf_repo=f'./mlx_models/{name}', language=language, batch_size=12)
    ```

11. **Model Loading for Inference** — **[LIBRARY]** `transcribe.py:148-149`
    - Uses `ModelHolder` singleton for in-memory caching
    - Calls `load_model("./mlx_models/{name}", dtype=mx.float16)`

12. **`load_model()` Function** — **[LIBRARY]** `load_models.py:14-39`
    - Checks if path exists locally
    - Loads `config.json` and `weights.npz`
    - Creates `whisper.Whisper` model with MLX
    - Applies quantization if specified in config

---

## Summary Table

| Attribute | Small Model | Large Model |
|-----------|-------------|-------------|
| **Requested Model** | `small` | `large` |
| **Mapped Model Name** | `small` | `large-v3` |
| **HuggingFace Repo ID** | `mlx-community/whisper-small-mlx` | `mlx-community/whisper-large-v3-mlx` |
| **Files Downloaded** | `weights.npz`, `config.json` | `weights.npz`, `config.json` |
| **Download Size** | 459 MB | 2.9 GB |
| **Local Model Directory** | `./mlx_models/small/` | `./mlx_models/large-v3/` |
| **First Run Time** | ~3 min (download) + 1.79s | ~10 min (download+resume) + 9.38s |
| **Cached Run Time** | ~2s load + 1.54s | ~1s load + 5.35s |

---

## Model Mapping Reference

### Project-Level Mapping — **[PROJECT]** `lightning.py:30-44`

| Input | Output | Notes |
|-------|--------|-------|
| `"large"` | `"large-v3"` | Upgrades to latest version |
| *(all others)* | *(unchanged)* | Passed through as-is |

### Library-Level Mapping — **[LIBRARY]** `lightning_whisper_mlx/lightning.py:4-52`

| Model Name | Base (No Quant) | 4-bit | 8-bit |
|------------|-----------------|-------|-------|
| `tiny` | `mlx-community/whisper-tiny` | `mlx-community/whisper-tiny-mlx-4bit` | `mlx-community/whisper-tiny-mlx-8bit` |
| `small` | `mlx-community/whisper-small-mlx` | `mlx-community/whisper-small-mlx-4bit` | `mlx-community/whisper-small-mlx-8bit` |
| `base` | `mlx-community/whisper-base-mlx` | `mlx-community/whisper-base-mlx-4bit` | `mlx-community/whisper-base-mlx-8bit` |
| `medium` | `mlx-community/whisper-medium-mlx` | `mlx-community/whisper-medium-mlx-4bit` | `mlx-community/whisper-medium-mlx-8bit` |
| `large` | `mlx-community/whisper-large-mlx` | `mlx-community/whisper-large-mlx-4bit` | `mlx-community/whisper-large-mlx-8bit` |
| `large-v2` | `mlx-community/whisper-large-v2-mlx` | `mlx-community/whisper-large-v2-mlx-4bit` | `mlx-community/whisper-large-v2-mlx-8bit` |
| `large-v3` | `mlx-community/whisper-large-v3-mlx` | `mlx-community/whisper-large-v3-mlx-4bit` | `mlx-community/whisper-large-v3-mlx-8bit` |
| `distil-small.en` | `mustafaaljadery/distil-whisper-mlx` | — | — |
| `distil-medium.en` | `mustafaaljadery/distil-whisper-mlx` | — | — |
| `distil-large-v2` | `mustafaaljadery/distil-whisper-mlx` | — | — |
| `distil-large-v3` | `mustafaaljadery/distil-whisper-mlx` | — | — |

### Distil Models Note

All four distil variants point to the **same HuggingFace repo**: `mustafaaljadery/distil-whisper-mlx`. The library differentiates them by filename within that repo, not by separate repos.

---

## Notes

### 1. What is MLX?

[MLX](https://github.com/ml-explore/mlx) is Apple's open-source array framework designed specifically for Apple Silicon:

- **Unified Memory**: CPU and GPU share the same memory - no copying data back and forth
- **Metal GPU Acceleration**: Uses Apple's Metal API for GPU compute
- **Zero-Copy Operations**: Arrays stay in unified memory, accessed by any device without movement
- **Optimized for Apple Silicon**: Leverages Neural Engine on M4/M5 chips

**Why it matters**: MLX is significantly faster than PyTorch on Apple Silicon because it avoids the memory copying overhead that PyTorch incurs when moving data between CPU and GPU.

### 2. SDPA (Scaled Dot-Product Attention) Relevance

**NOT relevant to this implementation.** SDPA is a PyTorch optimization. Lightning-whisper-mlx uses MLX with its own native attention implementation in `whisper.py:73-88`:

```python
def qkv_attention(self, q, k, v, mask=None):
    # Native MLX attention - not PyTorch SDPA
    qk = q @ k
    w = mx.softmax(qk, axis=-1).astype(q.dtype)
    out = (w @ v).transpose(0, 2, 1, 3)
```

### 3. ModelHolder: Singleton Pattern for Model Caching

**[LIBRARY]** `transcribe.py:51-60`

```python
class ModelHolder:
    model = None       # Shared across ALL uses
    model_path = None  # Shared across ALL uses

    @classmethod
    def get_model(cls, model_path: str, dtype: mx.Dtype):
        if cls.model is None or model_path != cls.model_path:
            cls.model = load_model(model_path, dtype=dtype)
            cls.model_path = model_path
        return cls.model
```

**Plain English**: A "model locker" that caches the loaded model in memory. First transcription loads from disk (~2-5 seconds), subsequent transcriptions reuse the cached model (instant).

**When it reloads**: If you switch model sizes (e.g., small → large), it detects the path change and reloads.

### 4. Data Type (dtype) Selection

**[LIBRARY]** `transcribe.py:148`

```python
dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
```

- `float16` (default): Half-precision, faster on Apple Silicon, uses less memory
- `float32`: Full precision, more accurate but slower

The default `float16` is optimal for Apple Silicon - the Neural Engine and GPU are optimized for half-precision operations.

### 5. Quantization Support

The library supports 4-bit and 8-bit quantization, which reduces model size and memory usage at the cost of some accuracy. Our project uses `quant=None` (full precision) for maximum accuracy.

**Available quantized models** (example for small):
- Full precision: `mlx-community/whisper-small-mlx` (459MB)
- 4-bit: `mlx-community/whisper-small-mlx-4bit` (~115MB, 75% smaller)
- 8-bit: `mlx-community/whisper-small-mlx-8bit` (~230MB, 50% smaller)

### 6. Working Directory Dependency

The local model path (`./mlx_models/`) is **relative to the current working directory**. Running benchmarks from different directories creates separate model caches:

```bash
cd /path/to/project && python test_benchmark2.py ...  # → ./mlx_models/
cd /tmp && python /path/to/project/test_benchmark2.py ...  # → /tmp/mlx_models/ (DIFFERENT!)
```

### 7. No Fallback Chains

Unlike some implementations (e.g., faster-whisper), lightning-whisper-mlx does **NOT** use fallback chains. If the requested model isn't in the `models` dict, it raises `ValueError("Please select a valid model")`.

### 8. Cross-Implementation Cache Inefficiency

Different MLX implementations use different cache locations:
- `lightning-whisper-mlx` → `./mlx_models/{name}/`
- `mlx-whisper` → `~/.cache/huggingface/hub/models--{repo-name}/`

Running both implementations downloads the same model twice to different locations.

---

## Key Source Files

### Project Files (You Can Modify)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `test_benchmark2.py` | Entry point, CLI argument parsing | 88-94 |
| `benchmark.py` | Benchmark orchestration, timing | 134 (load_model) |
| `lightning.py` | Implementation wrapper | 30-44 (mapping), 46-78 (load_model) |

### Library Files (Installed Package)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `lightning_whisper_mlx/lightning.py` | LightningWhisperMLX class, model dict | 4-52 (models), 54-91 |
| `lightning_whisper_mlx/load_models.py` | Model loading with MLX | 14-39 |
| `lightning_whisper_mlx/transcribe.py` | Transcription logic, ModelHolder | 51-60 (ModelHolder), 148 (dtype) |
| `lightning_whisper_mlx/whisper.py` | Whisper model architecture | 73-88 (attention) |

---

## Empirical Test Results

**Test Date**: January 12, 2026
**Test Environment**: macOS, Apple Silicon M4, huggingface-hub 0.30.2, hf-xet 1.0.3

### Small Model Tests

#### Test 1: Fresh Download

**Initial state**: Empty `./mlx_models/` folder

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py small 1 LightningWhisperMLXImplementation
```

**Terminal output**:
```
2026-01-12 20:38:25 [info] Loading LightningWhisperMLX model small
2026-01-12 20:41:16 [info] LightningWhisperMLX model small loaded successfully
2026-01-12 20:41:18 [info] Run 1 completed in 1.7938 seconds
```

**Download time**: ~2 minutes 51 seconds (459MB at ~2.7 MB/s)

**Files created**:
```
$ ls -la ./mlx_models/small/
-rw-r--r--@ 1 rymalia  staff        266 Jan 12 20:41 config.json
-rw-r--r--@ 1 rymalia  staff  481307592 Jan 12 20:41 weights.npz
```

**Total size**: 459 MB

#### Test 2: Cached Run

**Command run** (same as above):
```bash
.venv/bin/python3 test_benchmark2.py small 1 LightningWhisperMLXImplementation
```

**Terminal output**:
```
2026-01-12 20:45:58 [info] Loading LightningWhisperMLX model small
2026-01-12 20:46:00 [info] LightningWhisperMLX model small loaded successfully
2026-01-12 20:46:01 [info] Run 1 completed in 1.5438 seconds
```

**Model load time**: ~2 seconds (no download)
**Transcription time**: 1.54 seconds

**Conclusion**: ✅ No re-download on cached run.

### Large Model Tests

#### Test 1: Initial Download (with hf_xet hang)

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py large 1 LightningWhisperMLXImplementation
```

**Observations**:
- Download started at 20:48:09
- File reached full size (3.08GB) within ~8 minutes
- Process hung for 30+ additional minutes with 17 TCP connections stuck
- `lsof` showed process waiting in `_pthread_cond_wait`
- Stack trace revealed hang in `hf_xet::download_files` Rust extension

**Checksum during hang** (file size correct but content incomplete):
```
Expected: 05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d
Actual:   759ee71faac92a1ed397e0f72e2db4575c06a534ade8033b26df794238993031
```

**Action taken**: Killed process with SIGKILL

#### Test 2: Resume After Kill

**Command run** (restart):
```bash
.venv/bin/python3 test_benchmark2.py large 1 LightningWhisperMLXImplementation
```

**Key observation**: hf_xet **RESUMED** the download, filling in missing chunks:

```
Checksum at 21:27:32: 7a4f17221018763e4d60fc6e566d3ad2bfa992395c32ce6fcef95f00176b17d9
Checksum at 21:29:01: bb8db501e825fd4f0cab2436cb4700bcc871501f615f09c5793b72f7b3cf3cb6
Checksum at 21:31:28: bc3af8effd200e1f3a0d21f336db83f58223725c33575dceb27b3f4b338e73e5
...
Final checksum: 05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d ✓
```

**Terminal output**:
```
2026-01-12 21:26:56 [info] Loading LightningWhisperMLX model large
2026-01-12 21:37:21 [info] LightningWhisperMLX model large loaded successfully
2026-01-12 21:37:30 [info] Run 1 completed in 9.3783 seconds
```

**Resume + complete time**: ~10.5 minutes
**Transcription time**: 9.38 seconds

#### Test 3: Cached Large Model

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py large 1 LightningWhisperMLXImplementation
```

**Terminal output**:
```
2026-01-12 21:42:39 [info] Loading LightningWhisperMLX model large
2026-01-12 21:42:40 [info] LightningWhisperMLX model large loaded successfully
2026-01-12 21:42:46 [info] Run 1 completed in 5.3503 seconds
```

**Model load time**: ~1 second
**Transcription time**: 5.35 seconds

**Final files**:
```
$ ls -la ./mlx_models/large-v3/
-rw-r--r--@ 1 rymalia  staff         269 Jan 12 21:37 config.json
-rw-r--r--@ 1 rymalia  staff  3083520416 Jan 12 21:37 weights.npz

$ shasum -a 256 ./mlx_models/large-v3/weights.npz | cut -d' ' -f1
05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d ✓
```

---

## Known Issues / Bugs Discovered

### P0 (Critical) - hf_xet Hang During Large Downloads

**Problem**: The `hf_xet` Rust extension hung indefinitely during large file downloads in our testing. This may be related to the outdated package versions (see below), but causation was not confirmed.

**Observed behavior**:
- File reaches full size (3.08GB) but `.incomplete` suffix not removed
- 17 TCP connections stuck in ESTABLISHED state
- Process blocks in `_pthread_cond_wait` for 30+ minutes
- Checksum shows file content is incomplete despite correct size

**Outdated packages detected** (may or may not be related to hang):
| Package | Installed | Latest |
|---------|-----------|--------|
| `huggingface-hub` | **0.30.2** | **1.3.1** |
| `hf-xet` | **1.0.3** | **1.2.0** |

Note: `hf-xet` integration started at huggingface-hub 0.32.0, but this environment has 0.30.2.

**Workaround**: Kill the process and restart - hf_xet will resume and complete the download.

**Recommended Fix**: Upgrade packages:
```bash
uv pip install --upgrade huggingface-hub hf-xet
```

**Alternative**: Disable hf_xet entirely:
```bash
HF_HUB_DISABLE_XET_FETCH=1 python test_benchmark2.py large 1 LightningWhisperMLXImplementation
```

---

## Recommended Improvements

### 1. Upgrade HuggingFace Packages

**Problem**: Outdated packages may contribute to download hangs (observed but not confirmed)
**Impact**: Large model downloads may hang indefinitely
**Location**: `pyproject.toml` dependencies
**Recommended Fix**:
```toml
dependencies = [
    "huggingface-hub>=1.0.0",  # Was unconstrained
]
```
**Effort**: 1 line
**Priority**: P0

### 2. Add Download Progress Feedback

**Problem**: No feedback during potentially long downloads
**Impact**: Users may think process is frozen
**Location**: **[PROJECT]** `lightning.py:69-74`
**Recommended Fix**: Log before download starts:
```python
self.log.info(f"Downloading model {mapped_model_name} (this may take several minutes for large models)")
```
**Effort**: 1 line
**Priority**: P2

### 3. Consider Quantized Models Option

**Problem**: Full precision models are larger and slower
**Impact**: Users with limited memory can't run large models efficiently
**Location**: **[PROJECT]** `lightning.py:22-23`
**Recommended Fix**: Add CLI option for quantization level
**Effort**: ~20 lines
**Priority**: P3

---

## Priority Summary

| Priority | Issue | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| P0 | hf_xet hang during large downloads (possibly outdated packages) | upgrade | May block large model downloads | Not Fixed |
| P2 | No download progress feedback | 1 line | Poor UX | Not Fixed |
| P3 | No quantization option | ~20 lines | Memory efficiency | Not Fixed |

---

## Troubleshooting Techniques

### Checksum Monitoring for Download Progress

When a download appears stuck, compute the SHA256 checksum repeatedly to determine if data is still being written:

```bash
# Run multiple times, seconds apart
shasum -a 256 ./mlx_models/large-v3/.cache/huggingface/download/*.incomplete | cut -d' ' -f1
```

- **Checksum changing**: Chunks are still being written (download progressing)
- **Checksum static**: Download truly stuck

### Identify Stuck Download Processes

```bash
# Find the process
ps aux | grep test_benchmark2

# Check open files and connections
lsof -p <PID> | grep -E "TCP|incomplete"

# Sample the process to see what it's waiting on
sample <PID> 5
```

### Resume vs Restart Detection

The `hf_xet` library pre-allocates files to full size, so file size alone doesn't indicate completion. Check for:
- `.incomplete` suffix on the file
- Checksum match with expected (embedded in filename)
- Lock file presence (`.lock`)

---

## Implementation Order Recommendation

**Phase 1: Critical Fixes**
- [ ] Upgrade `huggingface-hub` to >=1.0.0
- [ ] Upgrade `hf-xet` to latest
- [ ] Add warning log for first-run large model downloads

**Phase 2: UX Improvements**
- [ ] Add download progress indication
- [ ] Document expected download times in README

**Phase 3: Future Enhancements**
- [ ] Add `--quant` CLI option for quantized models
- [ ] Consider using `~/.cache/huggingface/hub/` for cross-project model sharing
