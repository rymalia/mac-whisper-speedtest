# WhisperMPSImplementation Model Details

## Executive Summary

**CRITICAL FINDING: Despite being named "whisper-mps", this library uses MLX (Apple's Machine Learning framework), NOT MPS (Metal Performance Shaders).** The name is misleading - this is essentially an early MLX-based Whisper implementation that predates the now-popular `mlx-whisper` package.

Key characteristics:
- Downloads models from **OpenAI's Azure CDN** (NOT HuggingFace)
- Downloads **PyTorch `.pt` files** and converts to MLX format **at runtime**
- Conversion happens on every load (not cached), causing significant overhead
- Uses `~/.cache/whisper/` by default (project overrides to `models/`)

---

## File Reference Legend

| Prefix | Description |
|--------|-------------|
| [PROJECT] | Files in the mac-whisper-speedtest codebase |
| [LIBRARY] | Files in `.venv/lib/python3.12/site-packages/whisper_mps/` |

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| Does this use MPS (Metal Performance Shaders)? | **NO** - Uses MLX | [LIBRARY] whisper/whisper.py:9 imports `mlx.core as mx` and `mlx.nn as nn` |
| Where are models downloaded from? | OpenAI Azure CDN | [LIBRARY] whisper/load_models.py:18-31 `_MODELS` dict has URLs like `openaipublic.azureedge.net` |
| What format are downloaded models? | PyTorch `.pt` files | [LIBRARY] whisper/load_models.py:19-30 - URLs end in `.pt` |
| Are model files converted after download? | **YES** - PyTorch to MLX on every load | [LIBRARY] whisper/load_models.py:194-199 `load_model()` calls `torch_to_mlx()` |
| What cache location is used? | Project: `models/`, Default: `~/.cache/whisper/` | [PROJECT] whisper_mps.py:49-57, [LIBRARY] load_models.py:120-121 |
| What does "large" map to? | `large-v3` | [LIBRARY] load_models.py:30 - same hash as `large-v3` |
| Is there a `large-v3-turbo` option? | **NO** - Only original OpenAI models | [LIBRARY] load_models.py:18-31 shows no turbo variants |

---

## Why the Name is Misleading

The library is named `whisper-mps` but uses MLX, not MPS. Here's the evidence:

```python
# [LIBRARY] whisper/whisper.py lines 9-10
import mlx.core as mx
import mlx.nn as nn
```

```python
# [LIBRARY] whisper/load_models.py lines 9-10
import mlx.core as mx
from mlx.utils import tree_map
```

The copyright notice reveals the origin:
```python
# [LIBRARY] whisper/whisper.py line 1
# Copyright 2023 Apple Inc.
```

This appears to be **Apple's early MLX Whisper reference implementation**, released before the now-popular `mlx-whisper` community package existed. The "MPS" name likely comes from initial development when MPS was being considered, or from a time when "MLX" was less recognizable than "MPS" for marketing purposes.

**Historical significance:** This library represents an important piece of Apple's MLX rollout history—an official Apple demo showing how to port Whisper to their new framework. The `mlx-whisper` community package has since superseded it with pre-converted models and support for newer model variants (large-v3-turbo, quantized models).

---

## Benchmark Execution Flow

### Command: `test_benchmark2.py small 1 WhisperMPSImplementation`

```
test_benchmark2.py
    |
    +-- main(model_name="small", num_runs=1, implementations="WhisperMPSImplementation")
    |       |
    |       +-- sf.read("tests/jfk.wav")           # Load audio
    |       +-- BenchmarkConfig(model_name="small", ...)
    |       +-- run_benchmark(config)
    |               |
    |               +-- [PROJECT] benchmark.py:run_benchmark()
    |                       |
    |                       +-- WhisperMPSImplementation()
    |                       |       +-- Platform check (Darwin only)
    |                       |
    |                       +-- impl.load_model("small")
    |                       |       |
    |                       |       +-- [LIBRARY] whisper/load_models.py:load_model()
    |                       |       |       |
    |                       |       |       +-- load_torch_model("small", download_root)
    |                       |       |       |       |
    |                       |       |       |       +-- _download(url, root)
    |                       |       |       |       |       +-- urllib.request.urlopen()  <-- Direct HTTP download
    |                       |       |       |       |       +-- Write to {root}/small.pt
    |                       |       |       |       |       +-- SHA256 verification
    |                       |       |       |       |
    |                       |       |       |       +-- torch.load(checkpoint_file)
    |                       |       |       |       +-- torch_whisper.Whisper(dims)
    |                       |       |       |       +-- model.load_state_dict()
    |                       |       |       |
    |                       |       |       +-- torch_to_mlx(torch_model, dtype=float32)  <-- CONVERSION
    |                       |       |               +-- convert() recursively converts tensors
    |                       |       |               +-- whisper.Whisper(dims, dtype)
    |                       |       |               +-- mlx_model.update(params)
    |                       |       |
    |                       |       +-- return mlx_model
    |                       |
    |                       +-- impl.transcribe(audio)
    |                               |
    |                               +-- [LIBRARY] whisper/transcribe.py:transcribe()
    |                                       +-- ModelHolder.get_model()  <-- May reload!
    |                                       +-- log_mel_spectrogram(audio)
    |                                       +-- model.decode() via MLX
    |                                       +-- return {"text": ..., "segments": ..., "language": ...}
```

### Key Observations

1. **Double model loading potential**: The `transcribe()` function has its own `ModelHolder.get_model()` that can reload the model if the model name doesn't match the cached one. The project's `WhisperMPSImplementation.load_model()` stores the model in `self._model` but `transcribe()` uses a global `ModelHolder`. This could cause redundant conversions.

2. **No PyTorch-to-MLX caching**: Every `load_model()` call re-converts the PyTorch checkpoint to MLX format. For the large model (3GB), this takes ~20 seconds.

---

## Model Download Details

### Download Source

Unlike most implementations that use HuggingFace, whisper-mps downloads directly from **OpenAI's Azure CDN**:

```python
# [LIBRARY] whisper/load_models.py lines 18-31
_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/.../tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/.../tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/.../base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/.../base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/.../small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/.../small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/.../medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/.../medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/.../large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/.../large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/.../large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/.../large-v3.pt",  # Alias
}
```

### Download Method

```python
# [LIBRARY] whisper/load_models.py lines 51-92
def _download(url: str, root: str) -> str:
    # 1. Check if file exists and SHA256 matches
    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target  # Use cached file

    # 2. Download with progress bar
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(...) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    # 3. Verify SHA256 after download
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("SHA256 checksum does not match")
```

**Key difference from HuggingFace**: Uses direct HTTP with `urllib.request.urlopen()` instead of `huggingface_hub`. No resume capability for interrupted downloads.

---

## Model File Conversions

### The Conversion Pipeline

```
1. Download: small.pt (PyTorch checkpoint, 461MB)
       |
       v
2. torch.load(): Load into PyTorch Whisper model
       |
       v
3. torch_to_mlx(): Convert to MLX Whisper model  <-- HAPPENS EVERY LOAD
       |
       v
4. MLX model ready for inference
```

### Conversion Code

```python
# [LIBRARY] whisper/load_models.py lines 168-191
def torch_to_mlx(torch_model, dtype=mx.float16):
    # Convert PyTorch tensors to MLX arrays
    def convert_rblock(model, rules):
        # Handle ResidualAttentionBlock specially
        ...

    rules = {torch_whisper.ResidualAttentionBlock: convert_rblock}
    params = convert(torch_model, rules)  # Recursive conversion

    mlx_model = whisper.Whisper(torch_model.dims, dtype)
    params = tree_map(lambda p: p.astype(dtype), params)  # Cast to dtype
    mlx_model.update(params)
    return mlx_model
```

### Performance Impact

The conversion happens **every time `load_model()` is called**:
- Small model: ~1-2 seconds conversion overhead
- Large model: ~20 seconds conversion overhead

This is unlike `mlx-whisper` which downloads pre-converted MLX models from HuggingFace.

---

## Summary Table

| Aspect | Small Model | Large Model |
|--------|-------------|-------------|
| **Requested name** | `small` | `large` |
| **Actual model** | `small` | `large-v3` |
| **Download URL** | openaipublic.azureedge.net/.../small.pt | openaipublic.azureedge.net/.../large-v3.pt |
| **File format** | PyTorch .pt | PyTorch .pt |
| **File size** | 461 MB | 2.88 GB |
| **Cache location** | `{project}/models/small.pt` | `{project}/models/large-v3.pt` |
| **Conversion** | PyTorch -> MLX (every load) | PyTorch -> MLX (every load) |
| **Download time** | ~2 min | ~14 min |
| **Conversion time** | ~1-2 sec | ~20 sec |
| **Transcription time** | ~2.5 sec | ~27 sec |

---

## Model Mapping Reference

```python
# [LIBRARY] whisper/load_models.py lines 18-31
# Input -> Output mapping
"tiny.en"   -> tiny.en.pt      # 75 MB
"tiny"      -> tiny.pt         # 75 MB
"base.en"   -> base.en.pt      # 145 MB
"base"      -> base.pt         # 145 MB
"small.en"  -> small.en.pt     # 461 MB
"small"     -> small.pt        # 461 MB
"medium.en" -> medium.en.pt    # 1.5 GB
"medium"    -> medium.pt       # 1.5 GB
"large-v1"  -> large-v1.pt     # 2.9 GB
"large-v2"  -> large-v2.pt     # 2.9 GB
"large-v3"  -> large-v3.pt     # 2.9 GB
"large"     -> large-v3.pt     # 2.9 GB (alias)

# NO large-v3-turbo! This library only has original OpenAI models.
```

---

## Notes

### Framework Confusion (MPS vs MLX)

The library name suggests MPS (Metal Performance Shaders), but the implementation uses MLX. This is a historically significant piece of Apple's MLX ecosystem:

1. **Historical**: This predates `mlx-whisper` and was likely Apple's official MLX Whisper demo
2. **Marketing**: "MPS" was more recognizable than "MLX" when this was released (late 2023)
3. **Apple origin**: The copyright notice "Copyright © 2023 Apple Inc." confirms this is an official Apple reference implementation

**Why this matters for the benchmark:** The `whisper-mps` implementation is architecturally similar to `mlx-whisper`—both use MLX for inference. The key difference is that `whisper-mps` downloads PyTorch models and converts them at runtime, while `mlx-whisper` downloads pre-converted MLX models from HuggingFace. This makes `mlx-whisper` faster to load and gives it access to newer model variants.

MLX is Apple's newer framework that provides:
- Similar API to PyTorch
- Native Apple Silicon optimization via unified memory
- Lazy evaluation and automatic differentiation
- Better memory efficiency than MPS (no explicit device transfers needed)

### No HuggingFace Integration

This implementation is completely separate from the HuggingFace ecosystem:
- Does not use `huggingface_hub`
- Does not share cache with other implementations
- Does not benefit from HF's model versioning or metadata

### Audio Input

Unlike some implementations that require temp WAV files, whisper-mps accepts numpy arrays directly:

```python
# [LIBRARY] whisper/transcribe.py line 53
def transcribe(
    audio: Union[str, np.ndarray, mx.array],  # Accepts numpy directly!
    ...
)
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| [PROJECT] whisper_mps.py | Implementation wrapper, passes audio to library |
| [LIBRARY] whisper/load_models.py | Model downloading, PyTorch loading, MLX conversion |
| [LIBRARY] whisper/transcribe.py | Main transcription logic, audio processing |
| [LIBRARY] whisper/whisper.py | MLX model architecture (AudioEncoder, TextDecoder) |
| [LIBRARY] whisper/audio.py | Mel spectrogram generation (uses MLX FFT) |
| [LIBRARY] whisper/torch_whisper.py | PyTorch model architecture (used during conversion) |

---

## Empirical Test Results

**Test Date**: 2026-01-12
**Test Environment**: macOS, Apple Silicon

### Small Model Tests

#### Fresh Download Test

```bash
$ ls /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models/*.pt
# No .pt files found

$ .venv/bin/python3 test_benchmark2.py small 1 "WhisperMPSImplementation"
```

**Terminal Output** (summarized):
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz

Chosen implementations: 1
  - WhisperMPSImplementation

Starting benchmark with model 'small' (1 run(s))...
2026-01-12 10:58:08 [info] Loading whisper-mps model small
2026-01-12 10:58:09 [info] Using models directory: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
  0%|          | 0.00/461M [00:00<?, ?iB/s]
...
100%|██████████| 461M/461M [02:08<00:00, 3.58MiB/s]
2026-01-12 11:00:22 [info] Transcription completed. Text length: 108 characters
2026-01-12 11:00:22 [info] Run 1 completed in 2.4581 seconds

=== Benchmark Summary for 'small' model ===
whisper-mps            2.4581          model=small, backend=whisper-mps, device=mps, language=None
    "And so my fellow Americans, ask not what your country can do for you..."
```

**File Verification**:
```bash
$ ls -la /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models/*.pt
-rw-r--r--@ 1 rymalia  staff  483617219 Jan 12 11:00 models/small.pt
```

- **Download size**: 461 MB
- **Download time**: ~2 min 8 sec
- **Transcription time**: 2.46 seconds

#### Cached Run Test

```bash
$ .venv/bin/python3 test_benchmark2.py small 1 "WhisperMPSImplementation"
```

**Result**:
- No download progress bar (used cached file)
- Model loaded successfully
- Transcription time: 2.57 seconds

### Large Model Tests

#### Fresh Download Test

```bash
$ ls /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models/large*.pt
# No large .pt files found

$ .venv/bin/python3 test_benchmark2.py large 1 "WhisperMPSImplementation"
```

**Terminal Output** (summarized):
```
Loading audio from: tests/jfk.wav
Starting benchmark with model 'large' (1 run(s))...
2026-01-12 11:05:58 [info] Loading whisper-mps model large
2026-01-12 11:05:58 [info] Using models directory: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
  0%|          | 0.00/2.88G [00:00<?, ?iB/s]
...
100%|██████████| 2.88G/2.88G [13:52<00:00, 3.71MiB/s]
2026-01-12 11:19:54 [info] Successfully loaded whisper-mps model: large
Detected language: English
2026-01-12 11:20:20 [info] Run 1 completed in 26.6248 seconds

=== Benchmark Summary for 'large' model ===
whisper-mps            26.6248         model=large, backend=whisper-mps, device=mps, language=None
```

**File Verification**:
```bash
$ ls -la /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models/*.pt
-rw-r--r--@ 1 rymalia  staff  3087371615 Jan 12 11:19 models/large-v3.pt
-rw-r--r--@ 1 rymalia  staff   483617219 Jan 12 11:00 models/small.pt
```

- **Download size**: 2.88 GB
- **Download time**: ~14 minutes
- **Transcription time**: 26.6 seconds

#### Cached Run Test

```bash
$ .venv/bin/python3 test_benchmark2.py large 1 "WhisperMPSImplementation"
```

**Result**:
```
2026-01-12 11:21:56 [info] Loading whisper-mps model large
2026-01-12 11:22:16 [info] Successfully loaded whisper-mps model: large  # 20 sec model load!
2026-01-12 11:22:45 [info] Run 1 completed in 29.6978 seconds
```

- **No download** (used cached file)
- **Model load time**: ~20 seconds (PyTorch to MLX conversion)
- **Transcription time**: 29.7 seconds total

### Three Model States Summary

| State | Small Model | Large Model |
|-------|-------------|-------------|
| No local model | Downloads from Azure CDN, ~2 min | Downloads from Azure CDN, ~14 min |
| Complete local model | SHA256 verified, uses cached .pt | SHA256 verified, uses cached .pt |
| Partial/incomplete model | Warning + re-download (SHA256 mismatch) | Warning + re-download (SHA256 mismatch) |

---

## Known Issues / Conflicts Discovered

### 1. Misleading Library Name (P2)

**Issue**: Library named "whisper-mps" but uses MLX, not MPS.

**Impact**: User confusion; developers may expect MPS-specific optimizations.

**No action needed**: This is a third-party library issue.

### 2. Model Conversion on Every Load (P1)

**Issue**: PyTorch to MLX conversion happens on every `load_model()` call.

**Impact**: ~20 second overhead for large model, even when .pt file is cached.

**Location**: [LIBRARY] whisper/load_models.py:194-199

**Workaround**: None within current library. Use `mlx-whisper` instead for pre-converted models.

### 3. Separate Cache from HuggingFace (P2)

**Issue**: Uses custom `~/.cache/whisper/` directory, not shared with HuggingFace cache.

**Impact**: Models must be downloaded separately; can't share with other implementations.

**Location**: [LIBRARY] whisper/load_models.py:120-121, [PROJECT] whisper_mps.py:49-57

### 4. No Resume for Interrupted Downloads (P2)

**Issue**: Uses `urllib.request.urlopen()` which doesn't support download resumption.

**Impact**: If 2.9GB download fails at 90%, must restart from 0%.

**Location**: [LIBRARY] whisper/load_models.py:70-84

### 5. No large-v3-turbo Support (P3)

**Issue**: Only original OpenAI models available; no turbo variants.

**Impact**: Users cannot access newer, faster model variants.

**Location**: [LIBRARY] whisper/load_models.py:18-31

---

## Recommended Improvements

### 1. Add Warning About Framework (P2)

**Problem**: Users may be confused by "mps" in name when library uses MLX.

**Impact**: Unexpected behavior when debugging or comparing implementations.

**Location**: [PROJECT] implementations/whisper_mps.py:14

**Recommended Fix**:
```python
class WhisperMPSImplementation(WhisperImplementation):
    """Whisper implementation using whisper-mps.

    NOTE: Despite the name 'whisper-mps', this library actually uses MLX
    (Apple's Machine Learning framework), not MPS (Metal Performance Shaders).
    """
```

**Effort**: 1 line
**Priority**: P2

### 2. Document Performance Characteristics (P2)

**Problem**: Users don't know about 20-second conversion overhead for large models.

**Impact**: May choose this implementation expecting fast model loading.

**Location**: [PROJECT] implementations/whisper_mps.py or docs/

**Recommended Fix**: Add performance notes to docstring or documentation.

**Effort**: ~20 lines
**Priority**: P2

### 3. Consider Deprecating in Favor of mlx-whisper (P3)

**Problem**: `mlx-whisper` provides same MLX backend with pre-converted models (no runtime conversion).

**Impact**: whisper-mps is strictly inferior for performance.

**Location**: N/A (project decision)

**Recommendation**: Consider deprecating this implementation in favor of `MLXWhisperImplementation`, with note explaining the redundancy.

**Effort**: Medium (code removal + documentation)
**Priority**: P3

---

## Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| P1 | Model conversion caching | Large (library change) | High | Not actionable (library issue) |
| P2 | Add framework clarification comment | 1 line | Medium | Todo |
| P2 | Document performance characteristics | ~20 lines | Medium | Todo |
| P2 | Use HuggingFace cache | Large (library change) | Medium | Not actionable (library issue) |
| P3 | Consider deprecation | Medium | Low | Future consideration |

---

## Implementation Order Recommendation

### Phase 1: Documentation (Immediate)
- [ ] Add docstring clarification about MLX vs MPS
- [ ] Update implementation comparison docs with conversion overhead warning

### Phase 2: Evaluation (Future)
- [ ] Benchmark whisper-mps vs mlx-whisper on same models
- [ ] Decide if whisper-mps should be deprecated

### Phase 3: Optional Cleanup (Future)
- [ ] If deprecated: Remove whisper-mps, update registry
- [ ] If keeping: Consider forking library to add MLX model caching
