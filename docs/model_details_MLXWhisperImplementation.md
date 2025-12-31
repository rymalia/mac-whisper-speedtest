# MLXWhisperImplementation - Model Download & Cache Details

This document traces the execution flow, download URLs, and cache locations for the `MLXWhisperImplementation`.

## Benchmark Execution Flow

This section traces the execution flow for:
```bash
.venv/bin/python3 test_benchmark.py medium 1 MLXWhisperImplementation
```

### 1. Entry Point: `test_benchmark.py`

```
test_benchmark.py:78-80
↓
model = "medium"
runs = 1
implementations = "MLXWhisperImplementation"
↓
asyncio.run(main(model, runs, implementations))
```

### 2. Benchmark Runner: `benchmark.py:146`

```
benchmark.py:146
implementation.load_model("medium")
```

### 3. MLX Implementation: `mlx.py:61-125`

```python
def load_model(self, model_name: str) -> None:
    from huggingface_hub import snapshot_download
    from mlx_whisper.load_models import load_model

    self.model_name = model_name  # "medium"

    # Use base class helper to get model repo ID (quantized versions preferred)
    self.hf_repo = self._map_model_name(self.model_name)
    # → calls _get_model_map() → returns "mlx-community/whisper-medium-mlx-8bit"

    # Get the models directory from the utility function
    models_dir = str(get_models_dir())  # → {project_root}/models/

    # Download the model to custom cache
    model_path = snapshot_download(
        repo_id=self.hf_repo,  # "mlx-community/whisper-medium-mlx-8bit"
        cache_dir=models_dir,  # {project_root}/models/
    )

    # Load the model using mlx_whisper library
    self._model = load_model(model_path)
    self._model_path = model_path
```

### 4. Model Mapping: `mlx.py:29-43`

```python
def _get_model_map(self) -> Dict[str, str]:
    return {
        "tiny": "mlx-community/whisper-tiny-mlx-q4",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx-4bit",
        "medium": "mlx-community/whisper-medium-mlx-8bit",  # ← SELECTED
        "large": "mlx-community/whisper-large-v3-turbo",
        "large-v2": "mlx-community/whisper-large-v2-mlx-4bit",
        "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
    }
```

### 5. HuggingFace Hub Download

The `huggingface_hub.snapshot_download()` function downloads the entire repo:

```
https://huggingface.co/mlx-community/whisper-medium-mlx-8bit/resolve/main/config.json
https://huggingface.co/mlx-community/whisper-medium-mlx-8bit/resolve/main/weights.safetensors
```

### 6. Cache Behavior (Custom Directory)

| Aspect | Location |
|--------|----------|
| **Custom HF Cache** | `{project_root}/models/models--mlx-community--whisper-medium-mlx-8bit/` |
| **Snapshot Directory** | `{project_root}/models/models--mlx-community--whisper-medium-mlx-8bit/snapshots/{revision}/` |

**Flow:**
```
HuggingFace CDN (https://huggingface.co/mlx-community/whisper-medium-mlx-8bit/...)
      ↓ snapshot_download
{project_root}/models/models--mlx-community--whisper-medium-mlx-8bit/blobs/
      ↓ symlinked to
{project_root}/models/models--mlx-community--whisper-medium-mlx-8bit/snapshots/{revision}/config.json
{project_root}/models/models--mlx-community--whisper-medium-mlx-8bit/snapshots/{revision}/weights.safetensors
```

**Note:** This implementation does NOT use the default HuggingFace cache (`~/.cache/huggingface/hub/`). All models are stored in the project's `models/` directory.

### 7. Fallback Mechanism: `mlx.py:105-125`

If the quantized model fails to load, the implementation falls back to non-quantized:

```python
def _get_fallback_model(self, model_name: str) -> str:
    fallback_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",  # ← fallback for medium
        "large": "mlx-community/whisper-large-v3-turbo",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    }
    return fallback_map.get(model_name, model_name)
```

### 8. Model Loading: `mlx_whisper/load_models.py:14-46`

When `load_model(model_path)` is called from mlx_whisper library:

```python
def load_model(path_or_hf_repo: str, dtype: mx.Dtype = mx.float32) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        # Would call snapshot_download - but we already downloaded
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    # Load config and check for quantization
    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        quantization = config.pop("quantization", None)

    # Load weights (prefers safetensors)
    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    # Create model and apply quantization if specified in config
    model = whisper.Whisper(model_args, dtype)
    if quantization is not None:
        nn.quantize(model, **quantization, ...)

    return model
```

### 9. Transcription-Time Model Usage: `mlx.py:127-183`

When `transcribe()` is called:

```python
async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    from mlx_whisper import transcribe

    # Uses the already-downloaded model path
    result = transcribe(
        audio=audio,
        path_or_hf_repo=self._model_path,  # Already downloaded
        temperature=0.0,
        language=self.language,
        task="transcribe"
    )

    return TranscriptionResult(text=result.get("text", ""), ...)
```

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Actual Repo Used** | `mlx-community/whisper-medium-mlx-8bit` (8-bit quantized) |
| **Download URL Pattern** | `https://huggingface.co/mlx-community/whisper-medium-mlx-8bit/resolve/main/{file}` |
| **Custom Cache Location** | `{project_root}/models/models--mlx-community--whisper-medium-mlx-8bit/` |
| **Default HF Cache** | NOT USED |
| **Files Downloaded** | `config.json`, `weights.safetensors` (or `weights.npz`) |
| **Fallback Model** | `mlx-community/whisper-medium-mlx` (non-quantized) |
| **Transcription Uses** | Cached model from `_model_path` |

## Model Mapping Reference

The `MLXWhisperImplementation` uses this model mapping in `_get_model_map()`:
`src/mac_whisper_speedtest/implementations/mlx.py:29-43`

| Model Name | Primary (Quantized) | Fallback (Non-quantized) |
|------------|---------------------|--------------------------|
| `tiny` | `mlx-community/whisper-tiny-mlx-q4` | `mlx-community/whisper-tiny-mlx` |
| `base` | `mlx-community/whisper-base-mlx` | `mlx-community/whisper-base-mlx` |
| `small` | `mlx-community/whisper-small-mlx-4bit` | `mlx-community/whisper-small-mlx` |
| `medium` | `mlx-community/whisper-medium-mlx-8bit` | `mlx-community/whisper-medium-mlx` |
| `large` | `mlx-community/whisper-large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` |
| `large-v2` | `mlx-community/whisper-large-v2-mlx-4bit` | `mlx-community/whisper-large-v2-mlx` |
| `large-v3` | `mlx-community/whisper-large-v3-mlx-8bit` | `mlx-community/whisper-large-v3-mlx` |

## Notes

- The `MLXWhisperImplementation._get_model_map()` is used by BOTH `load_model()` and `get_model_info()`, ensuring consistency.
- The implementation uses a **custom cache directory** (`{project_root}/models/`) via the `cache_dir` parameter in `snapshot_download()`.
- When a quantized model fails to load (e.g., unsupported quantization format), the implementation automatically falls back to the non-quantized version.
- The `get_params()` method reports the actual quantization status after loading.

---

## check-models Command Flow

This section traces the execution flow for:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations MLXWhisperImplementation
```

### 1. CLI Entry Point: `cli.py:101-161`

```python
@app.command(name="check-models")
def check_models(model="medium", implementations="MLXWhisperImplementation", ...):
    # Filter to just MLXWhisperImplementation
    impls_to_check = [MLXWhisperImplementation]

    # Create checker and check all models
    checker = ModelChecker(verify_method=verify_method, verbose=verbose)
    statuses = checker.check_all_models("medium", impls_to_check)
```

### 2. ModelChecker.check_all_models(): `check_models.py:171-216`

```python
def check_all_models(self, model_size="medium", implementations):
    for impl_class in implementations:
        impl = MLXWhisperImplementation()

        # Get model info from implementation
        model_info = impl.get_model_info("medium")

        # Check HuggingFace cache (default location)
        hf_status, hf_size = self._check_hf_cache(model_info, impl, "medium")

        # Check local cache (custom location for MLX)
        local_status, local_size = self._check_local_cache(model_info, impl, "medium")
```

### 3. MLXWhisperImplementation.get_model_info(): `mlx.py:212-232`

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    from mac_whisper_speedtest.utils import get_models_dir

    # Use base class helper (same as load_model) - single source of truth
    repo_id = self._map_model_name(model_name)
    # → "mlx-community/whisper-medium-mlx-8bit"

    return ModelInfo(
        model_name=repo_id,
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
        # Since MLX uses custom cache, this will likely show "missing"
        return self._verify_hf_model(
            repo_id="mlx-community/whisper-medium-mlx-8bit",
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
            repo_id="mlx-community/whisper-medium-mlx-8bit",
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
        if repo.repo_id == repo_id:  # Looking for "mlx-community/whisper-medium-mlx-8bit"
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

## check-models Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **get_model_info() returns** | `mlx-community/whisper-medium-mlx-8bit` |
| **load_model() uses** | `mlx-community/whisper-medium-mlx-8bit` (SAME - no mismatch!) |
| **Default HF Cache Check** | `~/.cache/huggingface/hub/` → likely "missing" |
| **Custom Cache Check** | `{project_root}/models/` → shows actual status |
| **Verification Method** | `huggingface` (with load-based verification) |
| **Download Trigger** | `auto` (uses snapshot_download) |

---

## Consistency Note

Unlike some other implementations, `MLXWhisperImplementation` correctly follows the standardized pattern:

| Method | Repo for "medium" |
|--------|-------------------|
| `get_model_info()` via `_map_model_name()` | `mlx-community/whisper-medium-mlx-8bit` |
| `load_model()` via `_map_model_name()` | `mlx-community/whisper-medium-mlx-8bit` |

**Result:** Both methods use the same `_get_model_map()` dictionary, ensuring:
- `check-models` correctly identifies cached models
- No false "missing" reports for downloaded models
- Consistent verification with actual benchmark behavior

This implementation serves as a good example of the recommended pattern where `_get_model_map()` is the single source of truth for model mappings.
