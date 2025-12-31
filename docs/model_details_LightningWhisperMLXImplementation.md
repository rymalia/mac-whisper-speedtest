# LightningWhisperMLXImplementation - Model Download & Cache Details

This document traces the execution flow, download URLs, and cache locations for the `LightningWhisperMLXImplementation`.

## Benchmark Execution Flow

This section traces the execution flow for:
```bash
.venv/bin/python3 test_benchmark.py medium 1 LightningWhisperMLXImplementation
```

### 1. Entry Point: `test_benchmark.py`

```
test_benchmark.py:78-80
↓
model = "medium"
runs = 1
implementations = "LightningWhisperMLXImplementation"
↓
asyncio.run(main(model, runs, implementations))
```

### 2. Benchmark Runner: `benchmark.py:146`

```
benchmark.py:146
implementation.load_model("medium")
```

### 3. Lightning Implementation: `lightning.py:53-88`

```python
def load_model(self, model_name: str):
    from lightning_whisper_mlx import LightningWhisperMLX

    # Model stays "medium" (no remapping)
    model_for_loading = "medium"

    # Create instance - THIS triggers the download
    self.whisper_model = LightningWhisperMLX(
        model="medium",
        batch_size=12,
        quant=None  # Uses "base" variant
    )
```

### 4. LightningWhisperMLX Library: `.venv/.../lightning_whisper_mlx/lightning.py:54-88`

```python
class LightningWhisperMLX():
    def __init__(self, model="medium", batch_size=12, quant=None):
        # Model mapping (line 4-52):
        models = {
            "medium": {
                "base": "mlx-community/whisper-medium-mlx",     # ← SELECTED (quant=None)
                "4bit": "mlx-community/whisper-medium-mlx-4bit",
                "8bit": "mlx-community/whisper-medium-mlx-8bit"
            },
            ...
        }

        # Since quant=None, uses "base" variant
        repo_id = models["medium"]["base"]  # → "mlx-community/whisper-medium-mlx"

        # Downloads to LOCAL project directory (lines 82-88):
        filename1 = "weights.npz"
        filename2 = "config.json"
        local_dir = "./mlx_models/medium"

        hf_hub_download(repo_id="mlx-community/whisper-medium-mlx",
                        filename="weights.npz",
                        local_dir="./mlx_models/medium")
        hf_hub_download(repo_id="mlx-community/whisper-medium-mlx",
                        filename="config.json",
                        local_dir="./mlx_models/medium")
```

### 5. Download URLs

The `huggingface_hub.hf_hub_download()` function calls:

```
https://huggingface.co/mlx-community/whisper-medium-mlx/resolve/main/weights.npz
https://huggingface.co/mlx-community/whisper-medium-mlx/resolve/main/config.json
```

### 6. Cache Behavior (Two-Stage)

| Stage | Location | Purpose |
|-------|----------|---------|
| **Initial HF Cache** | `~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx/` | Standard HF cache (blob storage with symlinks) |
| **Local Copy** | `./mlx_models/medium/` | Copied to project's `mlx_models/` directory |

**Flow:**
```
HuggingFace CDN (https://huggingface.co/mlx-community/whisper-medium-mlx/...)
      ↓ download
~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx/blobs/
      ↓ hf_hub_download copies/links to local_dir
./mlx_models/medium/weights.npz
./mlx_models/medium/config.json
```

### 7. Transcription-Time Model Loading: `load_models.py:14-39`

When `transcribe()` is called:

```python
# transcribe.py:149
model = ModelHolder.get_model("./mlx_models/medium", dtype=mx.float16)

# load_models.py:14-39
def load_model(path_or_hf_repo: str, dtype):
    model_path = Path("./mlx_models/medium")

    if model_path.exists():
        # Uses LOCAL copy directly - no more downloads
        config = json.load(model_path / "config.json")
        weights = mx.load(str(model_path / "weights.npz"))

        return whisper.Whisper(model_args, dtype)
```

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Actual Repo Used** | `mlx-community/whisper-medium-mlx` (base, non-quantized) |
| **Download URL Pattern** | `https://huggingface.co/mlx-community/whisper-medium-mlx/resolve/main/{file}` |
| **HF Cache Location** | `~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx/` |
| **Local Model Storage** | `./mlx_models/medium/` (project-relative) |
| **Files Downloaded** | `weights.npz` (~1.5GB), `config.json` |
| **Transcription Uses** | `./mlx_models/medium/` (local copy) |

## Model Mapping Reference

The `lightning_whisper_mlx` library uses this internal model mapping:
`.venv/lib/python3.12/site-packages/lightning_whisper_mlx/lightning.py`

| Model Name | Base (quant=None) | 4-bit | 8-bit |
|------------|-------------------|-------|-------|
| `tiny` | `mlx-community/whisper-tiny` | `mlx-community/whisper-tiny-mlx-4bit` | `mlx-community/whisper-tiny-mlx-8bit` |
| `base` | `mlx-community/whisper-base-mlx` | `mlx-community/whisper-base-mlx-4bit` | `mlx-community/whisper-base-mlx-8bit` |
| `small` | `mlx-community/whisper-small-mlx` | `mlx-community/whisper-small-mlx-4bit` | `mlx-community/whisper-small-mlx-8bit` |
| `medium` | `mlx-community/whisper-medium-mlx` | `mlx-community/whisper-medium-mlx-4bit` | `mlx-community/whisper-medium-mlx-8bit` |
| `large` | `mlx-community/whisper-large-mlx` | `mlx-community/whisper-large-mlx-4bit` | `mlx-community/whisper-large-mlx-8bit` |
| `large-v2` | `mlx-community/whisper-large-v2-mlx` | `mlx-community/whisper-large-v2-mlx-4bit` | `mlx-community/whisper-large-v2-mlx-8bit` |
| `large-v3` | `mlx-community/whisper-large-v3-mlx` | `mlx-community/whisper-large-v3-mlx-4bit` | `mlx-community/whisper-large-v3-mlx-8bit` |

## Notes

- The `LightningWhisperMLXImplementation._get_model_map()` method is only used by `get_model_info()` for cache verification in the `check-models` command.
- The actual download is controlled by the library's internal `models` dict in `lightning_whisper_mlx/lightning.py`.
- When `quant=None` (default), the library uses the "base" (non-quantized) variant.
- The library always downloads to `./mlx_models/{model_name}/` relative to the current working directory.

---

## check-models Command Flow

This section traces the execution flow for:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations LightningWhisperMLXImplementation
```

### 1. CLI Entry Point: `cli.py:101-161`

```python
@app.command(name="check-models")
def check_models(model="medium", implementations="LightningWhisperMLXImplementation", ...):
    # Filter to just LightningWhisperMLXImplementation
    impls_to_check = [LightningWhisperMLXImplementation]

    # Create checker and check all models
    checker = ModelChecker()
    statuses = checker.check_all_models("medium", impls_to_check)
```

### 2. ModelChecker.check_all_models(): `check_models.py:171-216`

```python
def check_all_models(self, model_size="medium", implementations):
    for impl_class in implementations:
        impl = LightningWhisperMLXImplementation()

        # Get model info from implementation
        model_info = impl.get_model_info("medium")

        # Check HuggingFace cache
        hf_status, hf_size = self._check_hf_cache(model_info, impl, "medium")

        # Check local cache
        local_status, local_size = self._check_local_cache(model_info, impl, "medium")
```

### 3. LightningWhisperMLXImplementation.get_model_info(): `lightning.py:181-198`

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    # Uses base class helper
    repo_id = self._map_model_name(model_name)  # → calls _get_model_map()

    return ModelInfo(
        model_name=repo_id,
        repo_id=repo_id,
        cache_paths=[],
        verification_method="huggingface",
        download_trigger="auto",
        timeout_seconds=15  # medium model
    )
```

### 4. Model Mapping: `lightning.py:34-51`

```python
def _get_model_map(self) -> Dict[str, str]:
    return {
        "tiny": "mlx-community/whisper-tiny-mlx-q4",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx-4bit",
        "medium": "mlx-community/whisper-medium-mlx-8bit",  # ← 8-bit variant
        "large": "mlx-community/whisper-large-v3-turbo",
        "large-v2": "mlx-community/whisper-large-v2-mlx-4bit",
        "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
    }
```

**Result:** `get_model_info("medium")` returns `repo_id = "mlx-community/whisper-medium-mlx-8bit"`

### 5. HuggingFace Cache Check: `check_models.py:218-238`

```python
def _check_hf_cache(self, model_info, impl_instance, model_name):
    if model_info.verification_method == "huggingface":
        # Looks for repo_id in HF cache
        return self._verify_hf_model(
            repo_id="mlx-community/whisper-medium-mlx-8bit",
            impl_instance=impl_instance,
            model_name="medium"
        )
```

### 6. HF Model Verification: `check_models.py:240-297`

```python
def _verify_hf_model(self, repo_id, impl_instance, model_name):
    from huggingface_hub import scan_cache_dir
    cache_info = scan_cache_dir()  # Scans ~/.cache/huggingface/hub/

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:  # Looking for "mlx-community/whisper-medium-mlx-8bit"
            # Check for .incomplete markers
            # If found and no issues, proceed to load verification
            ...

    return "missing", None  # If not found
```

### 7. Verification by Loading: `check_models.py:299-335`

If the repo is found in cache, it verifies by actually loading:

```python
def _verify_by_loading(self, impl_instance, model_name):
    model_info = impl_instance.get_model_info(model_name)
    timeout = self._calculate_timeout(model_name, model_info)  # 15 seconds for medium

    # This calls load_model() which uses DIFFERENT mapping!
    status, _ = self._verify_with_timeout(impl_instance, "medium", timeout)
```

### 8. load_model() Uses Library's Internal Mapping

**`load_model()` in `lightning.py:53-88` does NOT use `_get_model_map()`:**

```python
def load_model(self, model_name: str) -> None:
    from lightning_whisper_mlx import LightningWhisperMLX

    # Just passes "medium" directly - library has its own mapping
    model_for_loading = "large-v3" if model_name == "large" else model_name

    self.whisper_model = LightningWhisperMLX(
        model="medium",      # ← Passes simple name to library
        batch_size=12,
        quant=None           # ← None selects "base" (non-quantized) variant
    )
```

**Library's internal mapping (`lightning_whisper_mlx/lightning.py:4-52`):**

```python
models = {
    "medium": {
        "base": "mlx-community/whisper-medium-mlx",      # ← SELECTED (quant=None)
        "4bit": "mlx-community/whisper-medium-mlx-4bit",
        "8bit": "mlx-community/whisper-medium-mlx-8bit"
    },
}

# Selection logic (lines 67-70):
if quant and "distil" not in model:
    repo_id = models[model][quant]   # Would use 4bit or 8bit
else:
    repo_id = models[model]['base']  # ← USES THIS (quant=None)
```

### 9. Local Model Storage

The library downloads to a project-relative directory:

```python
# lightning_whisper_mlx/lightning.py:82-88
filename1 = "weights.npz"
filename2 = "config.json"
local_dir = "./mlx_models/medium"

hf_hub_download(repo_id="mlx-community/whisper-medium-mlx",
                filename="weights.npz",
                local_dir="./mlx_models/medium")
```

**Download URLs (for actual load_model):**
```
https://huggingface.co/mlx-community/whisper-medium-mlx/resolve/main/weights.npz
https://huggingface.co/mlx-community/whisper-medium-mlx/resolve/main/config.json
```

## check-models Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **check-models looks for** | `mlx-community/whisper-medium-mlx-8bit` (via `_get_model_map()`) |
| **load_model() actually uses** | `mlx-community/whisper-medium-mlx` (via library, quant=None) |
| **HF Cache Location** | `~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx/` |
| **Local Model Storage** | `./mlx_models/medium/` (project-relative) |
| **Files Downloaded** | `weights.npz` (~1.5GB), `config.json` |

---

## ⚠️ Known Issue: Variant Mismatch (TODO: Fix Later)

There is a **mismatch** between what `check-models` verifies and what `load_model()` actually downloads:

| Method | Repo for "medium" | Variant |
|--------|-------------------|---------|
| `get_model_info()` via `_get_model_map()` | `mlx-community/whisper-medium-mlx-8bit` | 8-bit quantized |
| `load_model()` via library (quant=None) | `mlx-community/whisper-medium-mlx` | base (non-quantized) |

**Impact:**
- `check-models` may report the model as "missing" even when it's cached (looking for wrong repo)
- Verification by loading will succeed (uses correct repo), but initial cache scan may be misleading

**Root Cause:**
- `_get_model_map()` specifies quantized variants (8-bit for medium)
- `load_model()` passes `quant=None` to the library, which selects the "base" (non-quantized) variant
- The `load_model()` method does not use `_get_model_map()` at all

**Potential Fix:**
Either:
1. Update `_get_model_map()` to match the "base" variants that `load_model()` actually uses
2. Or modify `load_model()` to use `_get_model_map()` and pass the appropriate `quant` parameter to the library
