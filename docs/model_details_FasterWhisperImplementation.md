# FasterWhisperImplementation - Model Download & Cache Details

This document traces the execution flow, download URLs, and cache locations for the `FasterWhisperImplementation`.

## Overview

The `FasterWhisperImplementation` uses the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library, which is a CTranslate2-based reimplementation of OpenAI's Whisper model that runs on CPU. On Apple Silicon, it uses the Accelerate framework but does NOT support GPU acceleration (no MPS/Metal support).

**Key Characteristics:**
- Backend: CTranslate2 (CPU-only on macOS)
- Model Format: CTranslate2-converted models (`.bin` files)
- Download Source: HuggingFace Hub (Systran & mobiuslabsgmbh repos)
- Cache Location: Custom directory (`{project_root}/models/`)

---

## Benchmark Execution Flow

This section traces the execution flow for:
```bash
.venv/bin/python3 test_benchmark.py medium 1 FasterWhisperImplementation
```

### 1. Entry Point: `test_benchmark.py`

```
test_benchmark.py:78-80
↓
model = "medium"
runs = 1
implementations = "FasterWhisperImplementation"
↓
asyncio.run(main(model, runs, implementations))
```

### 2. Benchmark Runner: `benchmark.py:146`

```
benchmark.py:146
implementation.load_model("medium")
```

### 3. FasterWhisper Implementation: `faster.py:148-216`

```python
def load_model(self, model_name: str) -> None:
    from faster_whisper import WhisperModel

    self.original_model_name = model_name  # "medium"

    # Get fallback chain for requested model
    model_chain = self._get_model_fallback_chain(model_name)
    # → for "medium": ["medium"]
    # → for "large": ["large-v3-turbo", "large-v3", "large"]

    # Get models directory
    models_dir = get_models_dir()  # → {project_root}/models/

    # Try each model in fallback chain
    for model_to_try in model_chain:
        self._model = WhisperModel(
            model_size_or_path=model_to_try,  # "medium"
            device="cpu",
            compute_type="int8",
            download_root=str(models_dir),  # Custom cache directory
            cpu_threads=self.cpu_threads,
        )
        self.model_name = model_to_try
        break  # Success
```

### 4. Library's WhisperModel.__init__(): `faster_whisper/transcribe.py:588-681`

```python
class WhisperModel:
    def __init__(self, model_size_or_path, ..., download_root=None, ...):
        # If not a local path, download from HuggingFace
        if not os.path.isdir(model_size_or_path):
            model_path = download_model(
                model_size_or_path,  # "medium"
                local_files_only=local_files_only,
                cache_dir=download_root,  # {project_root}/models/
            )

        # Load with CTranslate2
        self.model = ctranslate2.models.Whisper(model_path, ...)
```

### 5. Library's download_model(): `faster_whisper/utils.py:49-124`

```python
_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",  # ← SELECTED for "medium"
    "large": "Systran/faster-whisper-large-v3",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    # ... more variants
}

def download_model(size_or_id, ..., cache_dir=None):
    # Look up model name in _MODELS dict
    repo_id = _MODELS.get(size_or_id)  # "Systran/faster-whisper-medium"

    # Define files to download
    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    # Download using HuggingFace Hub
    return huggingface_hub.snapshot_download(
        repo_id,  # "Systran/faster-whisper-medium"
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,  # {project_root}/models/
    )
```

### 6. HuggingFace Hub Download

The `huggingface_hub.snapshot_download()` function downloads from:

```
https://huggingface.co/Systran/faster-whisper-medium/resolve/main/config.json
https://huggingface.co/Systran/faster-whisper-medium/resolve/main/preprocessor_config.json
https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin
https://huggingface.co/Systran/faster-whisper-medium/resolve/main/tokenizer.json
https://huggingface.co/Systran/faster-whisper-medium/resolve/main/vocabulary.json
```

### 7. Cache Behavior (Custom Directory)

| Aspect | Location |
|--------|----------|
| **Custom HF Cache** | `{project_root}/models/models--Systran--faster-whisper-medium/` |
| **Snapshot Directory** | `{project_root}/models/models--Systran--faster-whisper-medium/snapshots/{revision}/` |

**Flow:**
```
HuggingFace CDN (https://huggingface.co/Systran/faster-whisper-medium/...)
      ↓ snapshot_download with cache_dir
{project_root}/models/models--Systran--faster-whisper-medium/blobs/
      ↓ symlinked to
{project_root}/models/models--Systran--faster-whisper-medium/snapshots/{revision}/model.bin
{project_root}/models/models--Systran--faster-whisper-medium/snapshots/{revision}/config.json
{project_root}/models/models--Systran--faster-whisper-medium/snapshots/{revision}/tokenizer.json
```

**Note:** This implementation does NOT use the default HuggingFace cache (`~/.cache/huggingface/hub/`). All models are stored in the project's `models/` directory via the `download_root` parameter.

### 8. Fallback Chain Mechanism: `faster.py:63-81`

The implementation has a fallback chain for the "large" model to handle model availability:

```python
def _get_model_fallback_chain(self, model_name: str) -> List[str]:
    if model_name == "large":
        return ["large-v3-turbo", "large-v3", "large"]
    # No fallback chain for other models
    return [model_name]
```

For "large" requests, it tries:
1. `large-v3-turbo` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo`
2. `large-v3` → `Systran/faster-whisper-large-v3` (fallback)
3. `large` → `Systran/faster-whisper-large-v3` (fallback)

### 9. Transcription-Time Model Usage: `faster.py:218-249`

When `transcribe()` is called:

```python
async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    # Model already loaded in load_model()
    segments, info = self._model.transcribe(
        audio,
        beam_size=self.beam_size,  # 1 (optimized for speed)
        language=self.language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments_list = list(segments)
    text = " ".join([segment.text for segment in segments_list])

    return TranscriptionResult(text=text, segments=segments_list, language=info.language)
```

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Actual Repo Used** | `Systran/faster-whisper-medium` |
| **Download URL Pattern** | `https://huggingface.co/Systran/faster-whisper-medium/resolve/main/{file}` |
| **Custom Cache Location** | `{project_root}/models/models--Systran--faster-whisper-medium/` |
| **Default HF Cache** | NOT USED |
| **Files Downloaded** | `config.json`, `preprocessor_config.json`, `model.bin`, `tokenizer.json`, `vocabulary.json` |
| **Backend** | CTranslate2 (CPU-only on macOS) |
| **Compute Type** | `int8` (quantized) |
| **Beam Size** | `1` (speed optimized) |

---

## Model Mapping Reference

### Implementation's `_get_model_map()`: `faster.py:44-61`

```python
def _get_model_map(self) -> Dict[str, str]:
    return {
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "large-v3": "Systran/faster-whisper-large-v3",
        "large": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # Primary for "large"
        "medium": "Systran/faster-whisper-medium",
        "small": "Systran/faster-whisper-small",
        "base": "Systran/faster-whisper-base",
        "tiny": "Systran/faster-whisper-tiny",
    }
```

### Library's `_MODELS` Dict: `faster_whisper/utils.py:12-31`

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

### Comparison Table

| Model Name | Implementation `_get_model_map()` | Library `_MODELS` | Match? |
|------------|-----------------------------------|-------------------|--------|
| `tiny` | `Systran/faster-whisper-tiny` | `Systran/faster-whisper-tiny` | YES |
| `base` | `Systran/faster-whisper-base` | `Systran/faster-whisper-base` | YES |
| `small` | `Systran/faster-whisper-small` | `Systran/faster-whisper-small` | YES |
| `medium` | `Systran/faster-whisper-medium` | `Systran/faster-whisper-medium` | YES |
| `large` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `Systran/faster-whisper-large-v3` | NO* |
| `large-v3` | `Systran/faster-whisper-large-v3` | `Systran/faster-whisper-large-v3` | YES |
| `large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | YES |

*Note: The `large` mapping differs, but the fallback chain ensures correct behavior (see below).

---

## Notes

### Apple Silicon Optimization

FasterWhisper is CPU-only on macOS but includes Apple Silicon optimizations:

```python
def _get_optimal_cpu_threads(self) -> int:
    # Detects Apple Silicon chip architecture via system_profiler
    # Uses performance cores + 2 efficiency cores for optimal balance
    result = subprocess.run(["system_profiler", "SPHardwareDataType"], ...)

    # Example: M4 Pro with 14 cores (10 performance + 4 efficiency)
    # → Uses 12 threads (10 perf + 2 eff)
```

### Compute Type

The implementation uses `int8` quantization for speed on CPU:

```python
self.compute_type = "int8"  # Faster on Apple Silicon CPU
self.beam_size = 1          # Speed over quality
```

### Fallback Chain for "large"

Unlike other implementations, FasterWhisper's `load_model()` doesn't directly map model names. Instead:

1. The implementation's `_get_model_fallback_chain("large")` returns `["large-v3-turbo", "large-v3", "large"]`
2. It passes each name directly to the library's `WhisperModel()` constructor
3. The library looks up the name in its own `_MODELS` dictionary

This means for "large":
- `load_model("large")` tries "large-v3-turbo" first → library uses `mobiuslabsgmbh/faster-whisper-large-v3-turbo`
- `get_model_info("large")` also uses "large-v3-turbo" as primary → matches

---

## check-models Command Flow

This section traces the execution flow for:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations FasterWhisperImplementation
```

### 1. CLI Entry Point: `cli.py:101-161`

```python
@app.command(name="check-models")
def check_models(model="medium", implementations="FasterWhisperImplementation", ...):
    # Filter to just FasterWhisperImplementation
    impls_to_check = [FasterWhisperImplementation]

    # Create checker and check all models
    checker = ModelChecker(verify_method=verify_method, verbose=verbose)
    statuses = checker.check_all_models("medium", impls_to_check)
```

### 2. ModelChecker.check_all_models(): `check_models.py:171-216`

```python
def check_all_models(self, model_size="medium", implementations):
    for impl_class in implementations:
        impl = FasterWhisperImplementation()

        # Get model info from implementation
        model_info = impl.get_model_info("medium")

        # Check HuggingFace cache (default location)
        hf_status, hf_size = self._check_hf_cache(model_info, impl, "medium")

        # Check local cache (custom location)
        local_status, local_size = self._check_local_cache(model_info, impl, "medium")
```

### 3. FasterWhisperImplementation.get_model_info(): `faster.py:267-291`

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    from mac_whisper_speedtest.utils import get_models_dir

    # Get primary model from fallback chain
    fallback_chain = self._get_model_fallback_chain(model_name)
    primary_model = fallback_chain[0]  # "medium" for medium, "large-v3-turbo" for large

    # Use base class helper to map to HuggingFace repo ID
    repo_id = self._map_model_name(primary_model)
    # → for "medium": "Systran/faster-whisper-medium"
    # → for "large": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"

    return ModelInfo(
        model_name=primary_model,
        repo_id=repo_id,
        cache_paths=[],  # HuggingFace manages cache automatically
        expected_size_mb=None,
        verification_method="huggingface",
        download_trigger="auto",
        hf_cache_dir=str(get_models_dir()),  # Custom cache: {project_root}/models/
        timeout_seconds=30 if "large" in model_name else 15
    )
```

### 4. HuggingFace Cache Check: `check_models.py:218-238`

```python
def _check_hf_cache(self, model_info, impl_instance, model_name):
    if model_info.verification_method == "huggingface":
        # Checks DEFAULT HF cache (~/.cache/huggingface/hub/)
        # Since FasterWhisper uses custom cache, this will likely show "missing"
        return self._verify_hf_model(
            repo_id="Systran/faster-whisper-medium",
            cache_dir=None,  # Default cache
            impl_instance=impl_instance,
            model_name="medium"
        )
```

### 5. Local Cache Check: `check_models.py:337-384`

```python
def _check_local_cache(self, model_info, impl_instance, model_name):
    # For HF-based implementations with custom cache directory
    if model_info.hf_cache_dir and model_info.repo_id:
        return self._verify_hf_model(
            repo_id="Systran/faster-whisper-medium",
            cache_dir=model_info.hf_cache_dir,  # {project_root}/models/
            impl_instance=impl_instance,
            model_name="medium"
        )
```

### 6. HF Model Verification: `check_models.py:240-297`

```python
def _verify_hf_model(self, repo_id, cache_dir, impl_instance, model_name):
    from huggingface_hub import scan_cache_dir

    # Scan the specified cache directory
    if cache_dir:
        cache_info = scan_cache_dir(cache_dir=cache_dir)  # Custom
    else:
        cache_info = scan_cache_dir()  # Default ~/.cache/huggingface/hub/

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:  # Looking for "Systran/faster-whisper-medium"
            # Check for .incomplete markers
            # If found and no issues, verify by loading
            if impl_instance and model_name:
                is_complete = self._verify_by_loading(impl_instance, model_name)
                return is_complete, size_mb

    return "missing", None
```

### 7. Verification by Loading: `check_models.py:299-335`

```python
def _verify_by_loading(self, impl_instance, model_name):
    model_info = impl_instance.get_model_info(model_name)
    timeout = self._calculate_timeout(model_name, model_info)  # 15 seconds for medium

    # Uses timeout-protected loading
    status, _ = self._verify_with_timeout(impl_instance, "medium", timeout)
    return status  # "complete", "incomplete", or "error"
```

---

## check-models Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **get_model_info() returns** | `Systran/faster-whisper-medium` |
| **load_model() uses** | `Systran/faster-whisper-medium` (SAME - no mismatch!) |
| **Default HF Cache Check** | `~/.cache/huggingface/hub/` → likely "missing" |
| **Custom Cache Check** | `{project_root}/models/` → shows actual status |
| **Verification Method** | `huggingface` (with load-based verification) |
| **Download Trigger** | `auto` (uses snapshot_download) |

---

## Consistency Analysis

### How `load_model()` Maps Model Names

1. `load_model("medium")` calls `_get_model_fallback_chain("medium")` → `["medium"]`
2. Passes "medium" directly to library's `WhisperModel(model_size_or_path="medium", ...)`
3. Library's `download_model()` looks up "medium" in its `_MODELS` dict → `Systran/faster-whisper-medium`

### How `get_model_info()` Maps Model Names

1. `get_model_info("medium")` calls `_get_model_fallback_chain("medium")` → `["medium"]`
2. `primary_model = "medium"`
3. Calls `self._map_model_name("medium")` → looks up in `_get_model_map()` → `Systran/faster-whisper-medium`

### Verification: Model Mapping Consistency

| Model | `load_model()` Path | `get_model_info()` Path | Match? |
|-------|---------------------|-------------------------|--------|
| `medium` | Library `_MODELS["medium"]` → `Systran/faster-whisper-medium` | `_get_model_map()["medium"]` → `Systran/faster-whisper-medium` | YES |
| `small` | Library `_MODELS["small"]` → `Systran/faster-whisper-small` | `_get_model_map()["small"]` → `Systran/faster-whisper-small` | YES |
| `large` | Fallback chain tries "large-v3-turbo" → Library `_MODELS["large-v3-turbo"]` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | Fallback chain uses "large-v3-turbo" → `_get_model_map()["large-v3-turbo"]` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | YES |

**Result:** Both methods produce the same repo ID for all model sizes. The implementation correctly mirrors the library's `_MODELS` dictionary in its `_get_model_map()` and uses the same fallback chain in both methods.

---

## No Known Issues

Unlike some other implementations, `FasterWhisperImplementation` correctly implements the standardized pattern:

1. **Single source of truth**: `_get_model_map()` mirrors the library's `_MODELS` dictionary
2. **Consistent fallback chain**: `_get_model_fallback_chain()` is used in both `load_model()` and `get_model_info()`
3. **Consistent cache directory**: Both use `get_models_dir()` as the custom HF cache location

**Result:**
- `check-models` correctly identifies cached models
- No false "missing" reports for downloaded models
- Consistent verification with actual benchmark behavior

This implementation serves as a good example of how to properly integrate with a library that has its own model mapping while maintaining consistency for the `check-models` command.
