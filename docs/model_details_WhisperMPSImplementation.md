# WhisperMPSImplementation - Model Details and Execution Flow

This document traces the execution flow for `WhisperMPSImplementation`, documenting how models are loaded, cached, and verified.

## Overview

| Aspect | Value |
|--------|-------|
| **Implementation File** | `src/mac_whisper_speedtest/implementations/whisper_mps.py` |
| **Backend Library** | `whisper-mps` (Apple MLX acceleration) |
| **Model Source** | OpenAI Azure CDN (`openaipublic.azureedge.net`) |
| **Download Method** | Direct HTTP download (NOT HuggingFace) |
| **Default Cache** | `~/.cache/whisper/` (library default) |
| **Override Cache** | `{project_root}/models/` (implementation overrides this) |

---

## Benchmark Execution Flow

**Command:**
```bash
.venv/bin/python3 test_benchmark.py medium 1 WhisperMPSImplementation
```

### Step-by-Step Execution

1. **Entry Point** - `test_benchmark.py:76-82`
   - Parses CLI arguments: `model="medium"`, `runs=1`, `implementations="WhisperMPSImplementation"`
   - Calls `asyncio.run(main(model, runs, implementations))`

2. **Implementation Discovery** - `test_benchmark.py:46-57`
   - `get_all_implementations()` returns list of available implementation classes
   - Filters to include only `WhisperMPSImplementation`

3. **Benchmark Runner** - `benchmark.py:135-146`
   - Creates implementation instance: `implementation = WhisperMPSImplementation()`
   - Calls `implementation.load_model("medium")`

4. **WhisperMPSImplementation.load_model()** - `whisper_mps.py:31-67`
   ```python
   from whisper_mps.whisper.load_models import load_model, available_models

   # Get project models directory
   models_dir = get_models_dir()  # Returns {project_root}/models/

   # Load model with custom download root
   self._model = load_model(
       name=model_name,           # "medium"
       download_root=str(models_dir)  # "{project}/models/"
   )
   ```

5. **Library load_model()** - `whisper_mps/whisper/load_models.py:194-199`
   ```python
   def load_model(name: str, download_root: str = None, dtype=mx.float32):
       return torch_to_mlx(load_torch_model(name, download_root), dtype)
   ```

6. **Library load_torch_model()** - `whisper_mps/whisper/load_models.py:100-141`
   - Looks up URL in `_MODELS` dict:
     ```python
     _MODELS = {
         "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
         ...
     }
     ```
   - Calls `_download(_MODELS[name], download_root)` to download if not cached
   - Loads PyTorch checkpoint from `.pt` file

7. **Library _download()** - `whisper_mps/whisper/load_models.py:51-92`
   - Download target: `os.path.join(root, os.path.basename(url))`
   - For "medium": `{download_root}/medium.pt`
   - Verifies SHA256 checksum (extracted from URL path)
   - Downloads with progress bar if not cached

8. **Model Conversion** - `whisper_mps/whisper/load_models.py:168-191`
   - `torch_to_mlx()` converts PyTorch model to MLX format
   - Applies `mx.float16` or `mx.float32` dtype

9. **Transcription** - `whisper_mps.py:69-125`
   ```python
   from whisper_mps.whisper.transcribe import transcribe

   result = transcribe(
       audio=audio,
       model=self.model_name,  # "medium" - NOTE: passes model NAME, not loaded model!
       temperature=0.0,
       language=self.language,
       ...
   )
   ```

10. **Library transcribe()** - `whisper_mps/whisper/transcribe.py:53-118`
    - Uses `ModelHolder.get_model(model, dtype)` singleton pattern
    - **ISSUE**: Will re-download model to default cache `~/.cache/whisper/` if not found there (empirically confirmed)

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Model URL** | `https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt` |
| **Download Location** | `{project_root}/models/medium.pt` |
| **Secondary Location** | `~/.cache/whisper/medium.pt` (see architectural issue below) |
| **File Size** | ~1.5 GB |
| **Checksum Validation** | SHA256 (embedded in URL) |

---

## Model Mapping Reference

The library's `_MODELS` dictionary (in `whisper_mps/whisper/load_models.py`):

| Model Name | Download URL | Saved Filename |
|------------|--------------|----------------|
| `tiny.en` | `https://openaipublic.azureedge.net/.../tiny.en.pt` | `tiny.en.pt` |
| `tiny` | `https://openaipublic.azureedge.net/.../tiny.pt` | `tiny.pt` |
| `base.en` | `https://openaipublic.azureedge.net/.../base.en.pt` | `base.en.pt` |
| `base` | `https://openaipublic.azureedge.net/.../base.pt` | `base.pt` |
| `small.en` | `https://openaipublic.azureedge.net/.../small.en.pt` | `small.en.pt` |
| `small` | `https://openaipublic.azureedge.net/.../small.pt` | `small.pt` |
| `medium.en` | `https://openaipublic.azureedge.net/.../medium.en.pt` | `medium.en.pt` |
| `medium` | `https://openaipublic.azureedge.net/.../medium.pt` | `medium.pt` |
| `large-v1` | `https://openaipublic.azureedge.net/.../large-v1.pt` | `large-v1.pt` |
| `large-v2` | `https://openaipublic.azureedge.net/.../large-v2.pt` | `large-v2.pt` |
| `large-v3` | `https://openaipublic.azureedge.net/.../large-v3.pt` | `large-v3.pt` |
| `large` | `https://openaipublic.azureedge.net/.../large-v3.pt` | **`large-v3.pt`** |

**Note**: The `"large"` key maps to the `large-v3.pt` URL, so the downloaded file is named `large-v3.pt`, NOT `large.pt`.

---

## Notes

### 1. Non-HuggingFace Download
Unlike most other implementations, `whisper-mps` downloads directly from OpenAI's Azure CDN (`openaipublic.azureedge.net`), not from HuggingFace Hub. This means:
- No HuggingFace cache integration
- Uses standard `urllib` for downloads
- SHA256 checksum validation (hash embedded in URL path)

### 2. MLX Backend (NOT MPS)
**IMPORTANT**: Despite the "whisper-mps" library name, the library does NOT use Metal Performance Shaders (MPS). It uses Apple's MLX framework exclusively:
```python
import mlx.core as mx
from mlx.utils import tree_map
import mlx.nn as nn
```
The model is converted from PyTorch format to MLX format during loading via `torch_to_mlx()`. A grep for "mps", "MPS", or "Metal" in the library source returns zero matches. The library name is misleading.

### 3. Model Holder Singleton
The library uses a `ModelHolder` class to cache loaded models:
```python
class ModelHolder:
    model = None
    model_name = None

    @classmethod
    def get_model(cls, model: str, dtype):
        if cls.model is None or model != cls.model_name:
            cls.model = load_model(model, dtype=dtype)  # No download_root!
            cls.model_name = model
        return cls.model
```
This will cause the model to be downloaded to the default location (`~/.cache/whisper/`) even if the implementation already loaded it from a custom location (empirically confirmed - models exist in both locations).

---

## check-models Command Flow

**Command:**
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations WhisperMPSImplementation
```

### Step-by-Step Execution

1. **CLI Entry** - `cli.py:101-161`
   - Parses options: `model="medium"`, `implementations="WhisperMPSImplementation"`
   - Creates `ModelChecker(verify_method=None, verbose=False)`

2. **Check All Models** - `check_models.py:171-216`
   ```python
   impl = WhisperMPSImplementation()
   model_info = impl.get_model_info("medium")
   ```

3. **get_model_info()** - `whisper_mps.py:136-168`
   ```python
   def get_model_info(self, model_name: str) -> ModelInfo:
       models_dir = get_models_dir()  # {project_root}/models/

       # Uses model name directly - whisper-mps handles versioning internally
       model_file = models_dir / f"{model_name}.pt"  # e.g., "medium.pt"

       return ModelInfo(
           model_name=model_name,      # "medium"
           repo_id=None,               # NOT HuggingFace
           cache_paths=[model_file],   # [{project}/models/medium.pt]
           expected_size_mb=1500,      # For size verification
           verification_method="size", # Local file verification
           download_trigger="native",  # Use load_model() to download
           timeout_seconds=15
       )
   ```

4. **Check HF Cache** - `check_models.py:218-238`
   - Returns `"n/a"` because `repo_id=None`

5. **Check Local Cache** - `check_models.py:337-384`
   - Checks if `cache_paths` exist: `{project}/models/medium.pt`
   - If exists, calculates size and verifies against `expected_size_mb`
   - Uses `_verify_by_loading()` with timeout protection

6. **Verify by Loading** - `check_models.py:299-335`
   - Calls `impl.load_model("medium")` with timeout
   - If load completes within timeout → "complete"
   - If timeout or FileNotFoundError → "incomplete"/"missing"

---

## check-models Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **repo_id** | `None` (not HuggingFace) |
| **verification_method** | `"size"` |
| **Expected cache_paths** | `[{project_root}/models/medium.pt]` |
| **expected_size_mb** | `1500` |
| **download_trigger** | `"native"` (calls load_model) |
| **HF Cache Column** | `n/a` |
| **Local Cache Column** | `complete` / `missing` / `incomplete` |

---

## Known Issue 1: Large Model Variant Mismatch

### Problem

When user requests model `"large"`:

| Component | What it uses | Result |
|-----------|--------------|--------|
| **load_model("large")** | `_MODELS["large"]` → URL `.../large-v3.pt` | Downloads `large-v3.pt` |
| **get_model_info("large")** | `models_dir / f"{model_name}.pt"` | Looks for `large.pt` |

### Impact

- `check-models` reports "missing" for `large` model
- Benchmark runs successfully (downloads `large-v3.pt`)
- Confusing status: model works but shows as missing

### Evidence

```bash
$ ls models/*.pt | grep large
models/large-v3.pt   # File exists as large-v3.pt

$ .venv/bin/mac-whisper-speedtest check-models --model large --implementations WhisperMPSImplementation
# Reports "missing" because it looks for models/large.pt
```

### Root Cause

The library's download function uses `os.path.basename(url)` to determine the filename:
```python
# whisper_mps/whisper/load_models.py:55
download_target = os.path.join(root, os.path.basename(url))
# For "large" → URL ends with "large-v3.pt" → saves as "large-v3.pt"
```

But `get_model_info()` naively constructs the path as `{model_name}.pt`:
```python
# whisper_mps.py:149
model_file = models_dir / f"{model_name}.pt"  # "large.pt" - WRONG!
```

### Potential Fix

Update `get_model_info()` to match the library's filename logic:

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    from whisper_mps.whisper.load_models import _MODELS
    import os

    models_dir = get_models_dir()

    # Match the library's filename logic exactly
    if model_name in _MODELS:
        url = _MODELS[model_name]
        filename = os.path.basename(url)  # "large-v3.pt" for "large"
    else:
        filename = f"{model_name}.pt"

    model_file = models_dir / filename
    # ... rest of ModelInfo
```

---

## Known Issue 2: Dual Download Location (Architectural)

### Problem

The model may be downloaded twice to different locations:

| Step | Download Location | Triggered By |
|------|-------------------|--------------|
| `load_model()` | `{project}/models/` | Implementation's explicit `download_root` |
| `transcribe()` | `~/.cache/whisper/` | Library's `ModelHolder.get_model()` with no `download_root` |

### Why This Happens

1. **Implementation's load_model()** passes custom `download_root`:
   ```python
   self._model = load_model(name=model_name, download_root=str(models_dir))
   ```

2. **Implementation's transcribe()** calls library's transcribe function:
   ```python
   result = transcribe(audio=audio, model=self.model_name, ...)
   ```

3. **Library's transcribe()** ignores the already-loaded model and uses `ModelHolder`:
   ```python
   # whisper_mps/whisper/transcribe.py:118
   model = ModelHolder.get_model(model, dtype)  # Calls load_model() with NO download_root
   ```

4. **ModelHolder.get_model()** calls `load_model()` without `download_root`:
   ```python
   cls.model = load_model(model, dtype=dtype)  # Defaults to ~/.cache/whisper/
   ```

### Evidence

Both locations contain the same models:
```bash
$ ls models/small.pt ~/.cache/whisper/small.pt
/Users/rymalia/projects/mac-whisper-speedtest/models/small.pt
/Users/rymalia/.cache/whisper/small.pt
```

### Impact

- **Wasted disk space**: Same model stored twice (~1.5GB for medium, ~3GB for large)
- **Confusing behavior**: Implementation claims to use project cache, but library may use system cache
- **No functional impact**: Transcription works correctly

### Potential Fix

The implementation stores the loaded model in `self._model`, but never uses it during transcription. A fix would require modifying how `transcribe()` works - either:

1. Pass the loaded model directly to the library's decode functions (requires significant refactoring)
2. Fork/patch the library to accept a pre-loaded model
3. Accept the dual download as a trade-off

---

## Apple Silicon Optimizations

The `whisper-mps` library uses Apple's MLX framework for hardware acceleration:

| Optimization | Value | Purpose |
|--------------|-------|---------|
| **Backend** | MLX | Apple's ML framework optimized for Apple Silicon |
| **Default dtype** | `mx.float16` | Half-precision for faster inference |
| **Conversion** | `torch_to_mlx()` | Converts PyTorch weights to MLX format |
| **Unified Memory** | Automatic | MLX manages GPU/CPU memory sharing |

### Implementation Parameters

```python
def get_params(self) -> Dict[str, Any]:
    return {
        "model": self.model_name,
        "backend": "whisper-mps",
        "device": "mlx",  # Note: despite library name, uses MLX not MPS
        "language": self.language,
    }
```

---

## File Locations Summary

| File Type | Location |
|-----------|----------|
| **Implementation** | `src/mac_whisper_speedtest/implementations/whisper_mps.py` |
| **Library source** | `.venv/lib/python3.12/site-packages/whisper_mps/` |
| **Model loading** | `.venv/.../whisper_mps/whisper/load_models.py` |
| **Transcription** | `.venv/.../whisper_mps/whisper/transcribe.py` |
| **Primary cache** | `{project_root}/models/*.pt` |
| **Secondary cache** | `~/.cache/whisper/*.pt` |
