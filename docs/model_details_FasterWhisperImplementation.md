# Model Details: FasterWhisperImplementation

This document provides comprehensive analysis of the FasterWhisperImplementation, tracing execution flow, model handling, and caching behavior.

## File Reference Legend

| Tag | Meaning |
|-----|---------|
| [PROJECT] | Files in this project's source code |
| [LIBRARY] | Files in `.venv/lib/python3.12/site-packages/faster_whisper/` |

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| **Does it download to HF HUB cache (`~/.cache/huggingface/hub/`)?** | **NO** | Uses project's `models/` directory via `download_root` parameter ([PROJECT] `faster.py:166`) |
| **Does it check HF HUB cache for existing models?** | **NO** | The `cache_dir` parameter overrides default behavior; library downloads directly to specified directory |
| **Does it download to project's `models/` folder?** | **YES** | Confirmed via empirical test - creates `models/models--Systran--faster-whisper-small/` |
| **Are model files converted/renamed after download?** | **NO** | Files are used as-is in CTranslate2 format (already pre-converted by Systran/mobiuslabs) |
| **Is it CPU or GPU based?** | **CPU only on Apple Silicon** | CTranslate2 only supports `cpu` device on macOS ([PROJECT] `faster.py:32`) |
| **Are there Apple Silicon optimizations?** | **YES** | P-core/E-core aware thread optimization ([PROJECT] `faster.py:58-121`) |
| **Does it use quantization?** | **YES** | Uses `int8` compute type for 2-4x speedup ([PROJECT] `faster.py:33`) |
| **What is CTranslate2?** | C++ inference engine for Transformers | Optimized runtime that faster-whisper uses instead of PyTorch |

---

## Benchmark Execution Flow

### Command Traced
```bash
.venv/bin/python3 test_benchmark2.py small 1 FasterWhisperImplementation
.venv/bin/python3 test_benchmark2.py large 1 FasterWhisperImplementation
```

### Execution Steps

1. **Entry Point** - `test_benchmark2.py:88-94`
   - Parses CLI args: `model="small"` or `"large"`, `num_runs=1`
   - Calls `main(model, runs, implementations)`

2. **Audio Loading** - `test_benchmark2.py:36-55`
   - Loads `tests/jfk.wav` (176,000 samples at 16kHz)
   - Ensures mono, float32 format

3. **Implementation Discovery** - `test_benchmark2.py:58-74`
   - Gets `FasterWhisperImplementation` from registry

4. **Benchmark Config** - `test_benchmark2.py:77-82`
   - Creates `BenchmarkConfig(model_name="small", ...)`

5. **Run Benchmark** - `benchmark.py:110-190`
   - Creates instance: `FasterWhisperImplementation()`
   - Calls `implementation.load_model("small")` or `load_model("large")`
   - Model loading is NOT timed
   - Runs transcription N times, measures average time

6. **Implementation Init** - [PROJECT] `faster.py:25-38`
   ```python
   self.device = "cpu"           # Only option on Apple Silicon
   self.compute_type = "int8"    # 8-bit quantization for speed
   self.beam_size = 1            # Greedy decoding for speed
   self.cpu_threads = self._get_optimal_cpu_threads()  # P-core aware
   ```

7. **Load Model** - [PROJECT] `faster.py:123-191`
   - For `"large"`: Applies fallback chain `["large-v3-turbo", "large-v3", "large"]`
   - For `"small"`: No fallback, uses directly
   - Gets `models_dir` from `utils.get_models_dir()` → `<project>/models/`
   - Calls `WhisperModel(model_size_or_path, download_root=models_dir, ...)`

8. **Library Model Loading** - [LIBRARY] `transcribe.py:588-680`
   ```python
   # WhisperModel.__init__
   model_path = download_model(
       model_size_or_path,
       local_files_only=local_files_only,
       cache_dir=download_root,  # <- Project's models/ folder
   )
   self.model = ctranslate2.models.Whisper(model_path, ...)
   ```

9. **Model Download** - [LIBRARY] `utils.py:49-123`
   - Maps model name to HuggingFace repo ID via `_MODELS` dict
   - Downloads using `huggingface_hub.snapshot_download(repo_id, cache_dir=...)`
   - Returns path to downloaded model

10. **Transcription** - [PROJECT] `faster.py:193-224`
    - Calls `self._model.transcribe(audio, beam_size=1, vad_filter=True, ...)`
    - Returns `TranscriptionResult` with text, segments, language

---

## Summary Table

| Requested Model | Actual Model Used | HuggingFace Repo ID | Cache Location | Files Downloaded |
|-----------------|-------------------|---------------------|----------------|------------------|
| `small` | `small` | `Systran/faster-whisper-small` | `models/models--Systran--faster-whisper-small/` | `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.txt` |
| `large` | `large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/` | Same file types |

---

## Model Mapping Reference

### Two-Stage Model Name Resolution

FasterWhisperImplementation has a **unique two-stage model name resolution** that no other implementation in this project has:

#### Stage 1: Project Fallback Chain ([PROJECT] `faster.py:40-56`)

```python
def _get_model_fallback_chain(self, model_name: str) -> List[str]:
    if model_name == "large":
        return ["large-v3-turbo", "large-v3", "large"]
    return [model_name]
```

This is a **unique feature** among all implementations:
- When user requests `"large"`, it first tries `"large-v3-turbo"` (the fastest large model)
- If that fails, it tries `"large-v3"`, then `"large"`
- Other model sizes pass through unchanged

**Why this matters:** The `large-v3-turbo` model is significantly faster than `large-v3` while maintaining quality. This fallback ensures users get the best available model.

#### Stage 2: Library Model Mapping ([LIBRARY] `utils.py:12-31`)

```python
_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
```

### Complete Resolution Examples

| User Request | Stage 1 (Fallback Chain) | Stage 2 (Library Mapping) | Final Repo ID |
|--------------|--------------------------|---------------------------|---------------|
| `small` | `["small"]` | `small` → | `Systran/faster-whisper-small` |
| `large` | `["large-v3-turbo", "large-v3", "large"]` | `large-v3-turbo` → | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| `medium` | `["medium"]` | `medium` → | `Systran/faster-whisper-medium` |
| `turbo` | `["turbo"]` | `turbo` → | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |

### Potential Redundancy/Confusion

There is a minor redundancy in the mapping:
- Project's fallback chain maps `"large"` → tries `"large-v3-turbo"` first
- Library's `_MODELS` maps `"large"` → `"Systran/faster-whisper-large-v3"`

If the fallback chain didn't exist, requesting `"large"` would get `large-v3` (from Systran). With the fallback chain, it gets `large-v3-turbo` (from mobiuslabsgmbh). The fallback chain overrides the library's default behavior for `"large"`.

---

## CTranslate2 Deep Dive

### What is CTranslate2?

CTranslate2 is a **C++ inference engine for Transformer models** that provides:
- **Optimized inference** - 2-4x faster than PyTorch for CPU inference
- **Quantization support** - int8, int16, float16, float32 compute types
- **Model conversion** - Converts PyTorch models to an optimized format
- **Multi-threading** - Efficient CPU parallelization

### Why faster-whisper Uses CTranslate2

OpenAI's original Whisper uses PyTorch, which is slow for inference. faster-whisper:
1. Uses pre-converted CTranslate2 models from HuggingFace (Systran, mobiuslabs)
2. Runs inference through CTranslate2's optimized C++ runtime
3. Achieves 2-4x speedup over original Whisper

### Compute Types on Apple Silicon

```python
>>> import ctranslate2
>>> ctranslate2.get_supported_compute_types('cpu')
{'int8', 'int8_float32', 'float32'}
```

| Compute Type | Description | Speed | Memory |
|--------------|-------------|-------|--------|
| `float32` | Full precision | Baseline | Highest |
| `int8_float32` | Weights int8, compute float32 | ~2x faster | ~4x smaller |
| `int8` | Full int8 quantization | ~2-4x faster | ~4x smaller |

This implementation uses `int8` for maximum speed.

### GPU Support Limitations

**CTranslate2 does NOT support MPS (Metal Performance Shaders) on Apple Silicon.**

- Only `cpu` and `cuda` devices are supported
- `cuda` requires NVIDIA GPUs
- Apple Silicon users are limited to CPU inference

This is why the implementation explicitly sets `self.device = "cpu"` and includes the docstring warning about GPU limitations.

---

## Apple Silicon Optimizations

### P-Core/E-Core Aware Thread Optimization ([PROJECT] `faster.py:58-121`)

FasterWhisperImplementation has sophisticated Apple Silicon CPU detection:

```python
def _get_optimal_cpu_threads(self) -> int:
    # Parse system_profiler for core info
    result = subprocess.run(["system_profiler", "SPHardwareDataType"], ...)

    # Extract: "Total Number of Cores: 14 (10 performance and 4 efficiency)"
    # Use: perf_cores + 2 efficiency cores
    optimal_threads = min(perf_cores + 2, total_cores)
```

**Strategy:**
- Uses all P-cores (performance cores) for maximum throughput
- Uses 2 E-cores (efficiency cores) for additional parallelism without thermal throttling
- Leaves remaining E-cores for system responsiveness

**Example for M3 Pro (8 total cores: 4P + 4E):**
- Detected: 8 total cores, 4 performance
- Optimal threads: min(4 + 2, 8) = 6 threads

### Other Optimizations

| Optimization | Setting | Rationale |
|--------------|---------|-----------|
| `beam_size=1` | Greedy decoding | Fastest inference, minimal quality loss for short audio |
| `compute_type=int8` | 8-bit quantization | 2-4x speedup with minimal quality loss |
| `vad_filter=True` | Voice activity detection | Skips silence, reduces compute |

---

## Notes

### Cache Behavior

1. **Project-local cache**: Models download to `<project>/models/` not the standard HF cache
2. **HuggingFace cache structure**: Uses standard `models--org--name/` directory layout
3. **Symlinks**: Files in `snapshots/` are symlinks to `blobs/` for deduplication
4. **No cross-project sharing**: Models are not shared with other projects using faster-whisper

### Model Format

The downloaded models are **already converted to CTranslate2 format**:
- `model.bin` - CTranslate2 binary weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer (HuggingFace format)
- `vocabulary.txt` - Vocabulary list

No conversion happens at runtime - models are ready to use immediately.

### Transcription Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `beam_size` | 1 | Greedy decoding (fast) |
| `language` | None | Auto-detect |
| `vad_filter` | True | Skip silence |
| `min_silence_duration_ms` | 500 | VAD sensitivity |

---

## Key Source Files

| File | Purpose |
|------|---------|
| [PROJECT] `faster.py` | Implementation with Apple Silicon optimizations |
| [PROJECT] `utils.py` | `get_models_dir()` returns project's `models/` path |
| [LIBRARY] `transcribe.py` | `WhisperModel` class, main transcription logic |
| [LIBRARY] `utils.py` | `_MODELS` mapping dict, `download_model()` function |
| [LIBRARY] `audio.py` | Audio decoding utilities |
| [LIBRARY] `vad.py` | Voice activity detection using Silero VAD |

---

## Empirical Test Results

### Test Date: 2026-01-12

### Small Model Tests

#### Fresh Download Test
```bash
$ .venv/bin/python3 test_benchmark2.py small 1 FasterWhisperImplementation
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - FasterWhisperImplementation

Starting benchmark with model 'small' (1 run(s))...
2026-01-12 12:53:49 [info     ] Benchmarking FasterWhisperImplementation with model small
2026-01-12 12:53:50 [info     ] Apple Silicon detected: 8 total cores (4 performance), using 6 threads
2026-01-12 12:53:50 [info     ] Loading model for FasterWhisperImplementation
2026-01-12 12:53:51 [info     ] Using models directory: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
2026-01-12 12:53:51 [info     ] Loading Faster Whisper model 'small'
2026-01-12 12:54:25 [info     ] Successfully loaded model 'small'
2026-01-12 12:54:25 [info     ] Run 1/1 for FasterWhisperImplementation
2026-01-12 12:54:25 [info     ] Transcribing with Faster Whisper using model small
2026-01-12 12:54:28 [info     ] Run 1 completed in 2.2376 seconds
2026-01-12 12:54:28 [info     ] Average time for FasterWhisperImplementation: 2.2376 seconds

=== Benchmark Summary for 'small' model ===
Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
faster-whisper         2.2376          model=small, device=cpu, compute_type=int8, beam_size=1, cpu_threads=6
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."
```

**Download time:** ~34 seconds
**Transcription time:** 2.24 seconds

#### Files Created
```bash
$ ls -la models/
total 16
drwxr-xr-x   5 rymalia  staff   160 Jan 12 12:53 .
drwxr-xr-x@  3 rymalia  staff    96 Jan 12 12:53 .locks
drwxr-xr-x@  5 rymalia  staff   160 Jan 12 12:54 models--Systran--faster-whisper-small

$ du -sh models/models--Systran--faster-whisper-small/
464M    models/models--Systran--faster-whisper-small/

$ ls -la models/models--Systran--faster-whisper-small/snapshots/*/
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 12 12:53 config.json -> ../../blobs/...
lrwxr-xr-x@ 1 rymalia  staff   76 Jan 12 12:54 model.bin -> ../../blobs/...
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 12 12:53 tokenizer.json -> ../../blobs/...
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 12 12:53 vocabulary.txt -> ../../blobs/...
```

#### Cached Run Test
```bash
$ .venv/bin/python3 test_benchmark2.py small 1 FasterWhisperImplementation
...
2026-01-12 12:54:53 [info     ] Loading Faster Whisper model 'small'
2026-01-12 12:54:54 [info     ] Successfully loaded model 'small'
...
faster-whisper         2.3246          model=small, device=cpu, compute_type=int8, beam_size=1, cpu_threads=6
```

**Model loading time (cached):** ~1 second
**Transcription time:** 2.32 seconds

### Large Model Tests

#### Fresh Download Test
```bash
$ .venv/bin/python3 test_benchmark2.py large 1 FasterWhisperImplementation
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - FasterWhisperImplementation

Starting benchmark with model 'large' (1 run(s))...
2026-01-12 12:55:13 [info     ] Benchmarking FasterWhisperImplementation with model large
2026-01-12 12:55:13 [info     ] Apple Silicon detected: 8 total cores (4 performance), using 6 threads
2026-01-12 12:55:13 [info     ] Loading model for FasterWhisperImplementation
2026-01-12 12:55:14 [info     ] Model fallback chain for 'large': large-v3-turbo → large-v3 → large
2026-01-12 12:55:14 [info     ] Using models directory: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
2026-01-12 12:55:14 [info     ] Attempting to load preferred model 'large-v3-turbo' (requested: 'large')
2026-01-12 13:00:30 [info     ] Successfully loaded model 'large-v3-turbo' (substituted from 'large')
2026-01-12 13:00:30 [info     ] Run 1/1 for FasterWhisperImplementation
2026-01-12 13:00:30 [info     ] Transcribing with Faster Whisper using model large-v3-turbo
2026-01-12 13:00:39 [info     ] Run 1 completed in 9.2701 seconds
2026-01-12 13:00:39 [info     ] Average time for FasterWhisperImplementation: 9.2701 seconds

=== Benchmark Summary for 'large' model ===
Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
faster-whisper         9.2701          model=large-v3-turbo, device=cpu, compute_type=int8, beam_size=1, cpu_threads=6, original_model_requested=large
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your..."
```

**Key observations:**
- **Fallback chain active:** `large-v3-turbo → large-v3 → large`
- **First fallback succeeded:** `large-v3-turbo` was available and downloaded
- **Model substitution tracked:** `original_model_requested=large` in params
- **Download time:** ~5 minutes (for 1.5GB model)
- **Transcription time:** 9.27 seconds

#### Files Created
```bash
$ ls -la models/
drwxr-xr-x@  5 rymalia  staff   160 Jan 12 13:00 models--mobiuslabsgmbh--faster-whisper-large-v3-turbo
drwxr-xr-x@  5 rymalia  staff   160 Jan 12 12:54 models--Systran--faster-whisper-small

$ du -sh models/*/
1.5G    models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/
464M    models/models--Systran--faster-whisper-small/
```

#### Cached Run Test
```bash
$ .venv/bin/python3 test_benchmark2.py large 1 FasterWhisperImplementation
...
2026-01-12 15:19:20 [info     ] Attempting to load preferred model 'large-v3-turbo' (requested: 'large')
2026-01-12 15:19:22 [info     ] Successfully loaded model 'large-v3-turbo' (substituted from 'large')
...
faster-whisper         9.7057          model=large-v3-turbo, device=cpu, compute_type=int8, beam_size=1, cpu_threads=6, original_model_requested=large
```

**Model loading time (cached):** ~2 seconds
**Transcription time:** 9.71 seconds

---

## Known Issues / Conflicts Discovered

### 1. Models Not Shared with Standard HuggingFace Cache

**Issue:** The implementation downloads models to `<project>/models/` instead of `~/.cache/huggingface/hub/`, preventing model sharing across projects.

**Impact:** Each project using faster-whisper downloads its own copy of large models (464MB-1.5GB each).

**Location:** [PROJECT] `faster.py:166` - passes `download_root=str(models_dir)`

### 2. No MPS/GPU Support on Apple Silicon

**Issue:** CTranslate2 only supports CPU on macOS, limiting performance compared to GPU-accelerated alternatives.

**Impact:** Slower than MLX-based or CoreML implementations for large models.

**Location:** [PROJECT] `faster.py:29-32` - hardcoded `device="cpu"`

---

## Recommended Improvements

### Improvement 1: Add Option to Use Standard HuggingFace Cache

**Problem:** Models are downloaded to project-local `models/` folder instead of standard HF cache, preventing cross-project model sharing.

**Impact:** Users download duplicate copies of 464MB-1.5GB models for each project.

**Location:** [PROJECT] `faster.py:166`

**Current Code:**
```python
self._model = self._faster_whisper(
    model_size_or_path=model_to_try,
    device=self.device,
    compute_type=self.compute_type,
    download_root=str(models_dir),  # Forces project-local cache
    cpu_threads=self.cpu_threads,
)
```

**Recommended Fix - Quick (1 line):**
```python
# Remove download_root to use default HF cache
self._model = self._faster_whisper(
    model_size_or_path=model_to_try,
    device=self.device,
    compute_type=self.compute_type,
    # download_root=str(models_dir),  # Removed - use standard HF cache
    cpu_threads=self.cpu_threads,
)
```

**Recommended Fix - Better (~20 lines):**
Add an environment variable or constructor parameter to control cache location:
```python
def __init__(self, use_hf_cache: bool = True):
    self.use_hf_cache = use_hf_cache
    # ...

def load_model(self, model_name: str) -> None:
    # ...
    kwargs = {
        "model_size_or_path": model_to_try,
        "device": self.device,
        "compute_type": self.compute_type,
        "cpu_threads": self.cpu_threads,
    }
    if not self.use_hf_cache:
        kwargs["download_root"] = str(get_models_dir())

    self._model = self._faster_whisper(**kwargs)
```

**Effort:** 1 line (quick) / ~20 lines (better)
**Priority:** P2 (Medium) - Saves disk space and download time for multi-project users

### Improvement 2: Document Fallback Chain Behavior

**Problem:** The unique fallback chain feature is not documented in docstrings or README.

**Impact:** Users may not understand why `large` gives them `large-v3-turbo` or how the fallback works.

**Location:** [PROJECT] `faster.py:40-56`

**Recommended Fix (~10 lines):**
Add docstring explaining the fallback:
```python
def _get_model_fallback_chain(self, model_name: str) -> List[str]:
    """Get the fallback chain for a given model name.

    UNIQUE FEATURE: FasterWhisperImplementation is the only implementation
    that provides automatic model fallback. When requesting 'large', it will:
    1. First try 'large-v3-turbo' (fastest large model, from mobiuslabs)
    2. Fall back to 'large-v3' (from Systran)
    3. Fall back to 'large' (maps to large-v3 in library)

    This ensures users always get the best available model variant.
    """
```

**Effort:** ~10 lines
**Priority:** P3 (Low) - Documentation improvement

### Improvement 3: Add Progress Indicator for Large Model Downloads

**Problem:** Large model downloads (1.5GB+) take 5+ minutes with no visible progress.

**Impact:** Users may think the process is hung during first-time downloads.

**Location:** [LIBRARY] `utils.py:96` - uses `disabled_tqdm` class

**Recommended Fix (~5 lines in project):**
The library disables tqdm by default. We could add logging to show download is in progress:
```python
self.log.info(f"Downloading model '{model_to_try}' - this may take several minutes for large models...")
```

**Effort:** 1 line (basic) / ~20 lines (with progress callback if library supports it)
**Priority:** P2 (Medium) - Improves UX for first-time users

---

## Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| P2 | Use standard HF cache option | 1-20 lines | Saves disk/bandwidth | Not started |
| P2 | Add download progress indicator | 1-20 lines | Better UX | Not started |
| P3 | Document fallback chain | ~10 lines | Better docs | Not started |

---

## Implementation Order Recommendation

### Phase 1: Quick Wins
- [ ] Add logging for download progress (1 line)
- [ ] Document fallback chain in docstring (10 lines)

### Phase 2: Cache Improvements
- [ ] Add option to use standard HF cache (20 lines)
- [ ] Consider environment variable for cache location

### Phase 3: Future Considerations
- [ ] Monitor CTranslate2 for MPS support (external dependency)
- [ ] Consider distil-whisper models for faster inference
