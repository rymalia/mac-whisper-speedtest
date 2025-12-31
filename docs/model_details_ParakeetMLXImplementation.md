# ParakeetMLXImplementation - Model Download & Cache Details

This document traces the execution flow, download URLs, and cache locations for the `ParakeetMLXImplementation`.

> **Note:** Parakeet is NVIDIA's speech recognition model, NOT a Whisper model. In this project all standard Whisper model sizes (tiny, small, medium, large) map to the same Parakeet model (`parakeet-tdt-0.6b-v2`), which is English-only.

## Benchmark Execution Flow

This section traces the execution flow for:
```bash
.venv/bin/python3 test_benchmark.py medium 1 ParakeetMLXImplementation
```

### 1. Entry Point: `test_benchmark.py`

```
test_benchmark.py:78-80
↓
model = "medium"
runs = 1
implementations = "ParakeetMLXImplementation"
↓
asyncio.run(main(model, runs, implementations))
```

### 2. Benchmark Runner: `benchmark.py:146`

```
benchmark.py:146
implementation.load_model("medium")
```

### 3. Parakeet Implementation: `parakeet_mlx.py:59-102`

```python
def load_model(self, model_name: str) -> None:
    from parakeet_mlx import from_pretrained

    self.model_name = model_name

    # Uses base class helper to get model repo ID
    self._hf_repo = self._map_model_name(self.model_name)
    # → "mlx-community/parakeet-tdt-0.6b-v2"

    # Set HuggingFace cache directory to our models directory
    models_dir = get_models_dir()  # → {project_root}/models/
    original_cache_dir = os.environ.get('HF_HOME')
    os.environ['HF_HOME'] = str(models_dir)

    try:
        # Load the model - THIS triggers the download
        self._model = from_pretrained(self._hf_repo)
    finally:
        # Restore original cache directory
        if original_cache_dir is not None:
            os.environ['HF_HOME'] = original_cache_dir
        else:
            os.environ.pop('HF_HOME', None)
```

### 4. parakeet-mlx Library: `.venv/.../parakeet_mlx/utils.py:59-78`

```python
def from_pretrained(hf_id_or_path: str, *, dtype: mx.Dtype = mx.bfloat16) -> BaseParakeet:
    """Loads model from Hugging Face or local directory"""
    try:
        # Downloads config and weights via huggingface_hub
        config = json.load(open(hf_hub_download(hf_id_or_path, "config.json"), "r"))
        weight = hf_hub_download(hf_id_or_path, "model.safetensors")
    except Exception:
        # Fallback to local directory
        config = json.load(open(Path(hf_id_or_path) / "config.json", "r"))
        weight = str(Path(hf_id_or_path) / "model.safetensors")

    model = from_config(config)
    model.load_weights(weight)

    # Cast to bfloat16
    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
```

### 5. Download URLs

The `huggingface_hub.hf_hub_download()` function calls:

```
https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/resolve/main/config.json
https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/resolve/main/model.safetensors
```

### 6. Cache Behavior

| Aspect | Value |
|--------|-------|
| **Custom Cache** | `{project_root}/models/hub/models--mlx-community--parakeet-tdt-0.6b-v2/` |
| **Default HF Cache** | NOT USED (due to `HF_HOME` override) |
| **Set via** | `os.environ['HF_HOME'] = str(models_dir)` before download |

**Flow:**
```
HuggingFace CDN (https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/...)
      ↓ download (with HF_HOME temporarily set)
{project_root}/models/hub/models--mlx-community--parakeet-tdt-0.6b-v2/blobs/
      ↓ symlinks to snapshots
{project_root}/models/hub/models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/{commit_hash}/
    ├── config.json
    └── model.safetensors
```

### 7. Transcription-Time Model Loading

When `transcribe()` is called, the model is already loaded in memory (`self._model`):

```python
async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    if self._model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # parakeet-mlx expects file paths
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio, 16000, format="WAV")

        # Transcribe using the pre-loaded model
        result = self._model.transcribe(temp_file.name)
```

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Actual Repo Used** | `mlx-community/parakeet-tdt-0.6b-v2` |
| **Download URL Pattern** | `https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/resolve/main/{file}` |
| **HF Cache Location** | NOT USED (default `~/.cache/huggingface/hub/` bypassed) |
| **Custom Cache Location** | `{project_root}/models/hub/models--mlx-community--parakeet-tdt-0.6b-v2/` |
| **Files Downloaded** | `config.json`, `model.safetensors` (~1.2GB) |
| **Data Type** | `bfloat16` (converted after loading) |

## Model Mapping Reference

The `ParakeetMLXImplementation._get_model_map()` method maps Whisper model names to Parakeet repos:
`src/mac_whisper_speedtest/implementations/parakeet_mlx.py:33-57`

| Model Name | Repo ID |
|------------|---------|
| `parakeet-tdt-0.6b` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `parakeet-tdt-0.6b-v2` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `parakeet-tdt-1.1b` | `mlx-community/parakeet-tdt-1.1b` |
| `parakeet-ctc-0.6b` | `mlx-community/parakeet-ctc-0.6b` |
| `parakeet-ctc-1.1b` | `mlx-community/parakeet-ctc-1.1b` |
| `tiny` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `small` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `base` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `medium` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `large` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `large-v2` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `large-v3` | `mlx-community/parakeet-tdt-0.6b-v2` |

**Note:** All standard Whisper sizes map to `parakeet-tdt-0.6b-v2` as it's the recommended general-purpose English model.

## Notes

1. **Not a Whisper Model**: Parakeet is NVIDIA's speech recognition model with a different architecture (Conformer-based). It uses Token-and-Duration Transducer (TDT) decoding.

2. **English-Only**: Parakeet TDT models are optimized for English. The implementation includes heuristics to detect non-English input (`_contains_non_english_patterns()`).

3. **Standardized Pattern**: This implementation correctly uses `_map_model_name()` in both `load_model()` and `get_model_info()`, ensuring consistency between benchmark and check-models.

4. **Audio Input**: Requires file paths rather than audio arrays. The implementation saves audio to a temporary WAV file for transcription.

5. **Model Variants**:
   - **TDT (Token-and-Duration Transducer)**: `parakeet-tdt-*` - Uses duration prediction
   - **CTC (Connectionist Temporal Classification)**: `parakeet-ctc-*` - Traditional CTC decoding
   - **Sizes**: 0.6B and 1.1B parameter versions available

---

## check-models Command Flow

This section traces the execution flow for:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations ParakeetMLXImplementation
```

### 1. CLI Entry Point: `cli.py:101-161`

```python
@app.command(name="check-models")
def check_models(model="medium", implementations="ParakeetMLXImplementation", ...):
    # Filter to just ParakeetMLXImplementation
    impls_to_check = [ParakeetMLXImplementation]

    # Create checker and check all models
    checker = ModelChecker()
    statuses = checker.check_all_models("medium", impls_to_check)
```

### 2. ModelChecker.check_all_models(): `check_models.py:171-216`

```python
def check_all_models(self, model_size="medium", implementations):
    for impl_class in implementations:
        impl = ParakeetMLXImplementation()

        # Get model info from implementation
        model_info = impl.get_model_info("medium")

        # Check HuggingFace cache
        hf_status, hf_size = self._check_hf_cache(model_info, impl, "medium")

        # Check local cache
        local_status, local_size = self._check_local_cache(model_info, impl, "medium")
```

### 3. ParakeetMLXImplementation.get_model_info(): `parakeet_mlx.py:231-252`

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    # Uses base class helper (same as load_model) - single source of truth
    repo_id = self._map_model_name(model_name)
    # → "mlx-community/parakeet-tdt-0.6b-v2"

    return ModelInfo(
        model_name=repo_id,
        repo_id=repo_id,
        cache_paths=[],
        expected_size_mb=None,
        verification_method="huggingface",
        download_trigger="auto",
        timeout_seconds=30 if "large" in model_name else 15
    )
```

### 4. Model Mapping: `parakeet_mlx.py:33-57`

```python
def _get_model_map(self) -> Dict[str, str]:
    return {
        "parakeet-tdt-0.6b": "mlx-community/parakeet-tdt-0.6b-v2",
        "parakeet-tdt-0.6b-v2": "mlx-community/parakeet-tdt-0.6b-v2",
        "parakeet-tdt-1.1b": "mlx-community/parakeet-tdt-1.1b",
        "parakeet-ctc-0.6b": "mlx-community/parakeet-ctc-0.6b",
        "parakeet-ctc-1.1b": "mlx-community/parakeet-ctc-1.1b",
        # Size-based mappings (all map to best general-purpose model)
        "tiny": "mlx-community/parakeet-tdt-0.6b-v2",
        "small": "mlx-community/parakeet-tdt-0.6b-v2",
        "base": "mlx-community/parakeet-tdt-0.6b-v2",
        "medium": "mlx-community/parakeet-tdt-0.6b-v2",  # ← SELECTED
        "large": "mlx-community/parakeet-tdt-0.6b-v2",
        ...
    }
```

**Result:** `get_model_info("medium")` returns `repo_id = "mlx-community/parakeet-tdt-0.6b-v2"`

### 5. HuggingFace Cache Check: `check_models.py:218-238`

```python
def _check_hf_cache(self, model_info, impl_instance, model_name):
    if model_info.verification_method == "huggingface":
        # Checks DEFAULT HF cache (not custom location!)
        return self._verify_hf_model(
            repo_id="mlx-community/parakeet-tdt-0.6b-v2",
            cache_dir=None,  # Uses ~/.cache/huggingface/hub/
            impl_instance=impl_instance,
            model_name="medium"
        )
```

### 6. Local Cache Check: `check_models.py:337-384`

```python
def _check_local_cache(self, model_info, impl_instance, model_name):
    # For HF-based implementations with custom cache directory
    if model_info.hf_cache_dir and model_info.repo_id:
        return self._verify_hf_model(repo_id, cache_dir=model_info.hf_cache_dir, ...)

    # hf_cache_dir is None, cache_paths is empty
    if not model_info.cache_paths:
        return "n/a", None  # ← Returns this!
```

Since `get_model_info()` doesn't specify `hf_cache_dir`, the local cache check returns "n/a".

### 7. Verification by Loading: `check_models.py:299-335`

If the repo is found in the default HF cache, verification proceeds by loading:

```python
def _verify_by_loading(self, impl_instance, model_name):
    model_info = impl_instance.get_model_info(model_name)
    timeout = self._calculate_timeout(model_name, model_info)  # 15 seconds

    # Calls load_model() which sets HF_HOME to custom location
    status, _ = self._verify_with_timeout(impl_instance, "medium", timeout)
```

The `load_model()` call sets `HF_HOME` temporarily, so if the model was previously downloaded to the custom cache, it will load successfully.

## check-models Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **check-models looks for** | `mlx-community/parakeet-tdt-0.6b-v2` |
| **load_model() actually uses** | `mlx-community/parakeet-tdt-0.6b-v2` |
| **HF Cache Verified** | `~/.cache/huggingface/hub/` (default) |
| **Local Cache Status** | "n/a" (no custom cache paths defined in ModelInfo) |
| **Actual Cache Location** | `{project_root}/models/hub/` (set via HF_HOME in load_model) |

---

## Variant Mismatch: None

**Good news:** `ParakeetMLXImplementation` correctly follows the standardized pattern.

| Method | Repo for "medium" |
|--------|-------------------|
| `get_model_info()` via `_map_model_name()` | `mlx-community/parakeet-tdt-0.6b-v2` |
| `load_model()` via `_map_model_name()` | `mlx-community/parakeet-tdt-0.6b-v2` |

Both methods use the same `_map_model_name()` helper, ensuring consistency.

---

## Potential Improvement: Cache Location Mismatch

While there's no variant mismatch, there is a **cache location verification issue**:

| Operation | Cache Location |
|-----------|----------------|
| `load_model()` downloads to | `{project_root}/models/hub/` (via HF_HOME override) |
| `check-models` verifies in | `~/.cache/huggingface/hub/` (default) |
| `get_model_info().hf_cache_dir` | `None` (not specified) |

**Impact:**
- If the model was downloaded via `load_model()`, it exists in `{project_root}/models/`
- `check-models` looks in default HF cache → may report "missing"
- However, verification by loading will succeed (since it uses the same HF_HOME override)

**Potential Fix:**
Update `get_model_info()` to specify `hf_cache_dir`:

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    repo_id = self._map_model_name(model_name)
    models_dir = get_models_dir()

    return ModelInfo(
        model_name=repo_id,
        repo_id=repo_id,
        cache_paths=[],
        expected_size_mb=None,
        verification_method="huggingface",
        download_trigger="auto",
        hf_cache_dir=str(models_dir),  # ← ADD THIS
        timeout_seconds=30 if "large" in model_name else 15
    )
```
