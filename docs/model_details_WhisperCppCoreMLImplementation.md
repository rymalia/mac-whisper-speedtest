# WhisperCppCoreMLImplementation - Model Download & Cache Details

This document traces the execution flow, download URLs, and cache locations for the `WhisperCppCoreMLImplementation`.

## Overview

The `WhisperCppCoreMLImplementation` uses the [pywhispercpp](https://github.com/absadiki/pywhispercpp) library, which provides Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp). It downloads pre-quantized GGML models and optionally uses CoreML encoders for hardware acceleration on Apple Silicon.

**Key Characteristics:**
- Backend: whisper.cpp (C++ with Python bindings)
- Model Format: GGML quantized (`.bin` files) + optional CoreML encoders (`.mlmodelc` directories)
- Download Source: Direct HTTP from HuggingFace (not HF Hub API)
- Cache Location: Custom directory (`{project_root}/models/`)
- CoreML Support: Optional, requires manually downloaded encoder models

---

## Benchmark Execution Flow

This section traces the execution flow for:
```bash
.venv/bin/python3 test_benchmark.py medium 1 WhisperCppCoreMLImplementation
```

### 1. Entry Point: `test_benchmark.py`

```
test_benchmark.py:78-80
↓
model = "medium"
runs = 1
implementations = "WhisperCppCoreMLImplementation"
↓
asyncio.run(main(model, runs, implementations))
```

### 2. Benchmark Runner: `benchmark.py:146`

```
benchmark.py:146
implementation.load_model("medium")
```

### 3. WhisperCppCoreML Implementation: `coreml.py:63-99`

```python
def load_model(self, model_name: str) -> None:
    import pywhispercpp.model

    # Use base class helper to map to quantized GGML model name
    self.model_name = self._map_model_name(model_name)
    # → for "medium": "medium-q5_0"
    # → for "small": "small"
    # → for "large": "large-v3-turbo-q5_0"

    # Check if CoreML is enabled (via environment variable)
    self.coreml_enabled = os.environ.get("WHISPER_COREML", "0") == "1"

    if self.coreml_enabled:
        # Get CoreML encoder model name using helper method
        coreml_model_map = self._get_coreml_model_map()
        coreml_model_name = coreml_model_map.get(model_name, model_name)
        # → for "medium": "medium"

        # Check if CoreML model files exist
        coreml_model_path = self.models_dir / f"ggml-{coreml_model_name}-encoder.mlmodelc"
        # → models/ggml-medium-encoder.mlmodelc

        if not coreml_model_path.exists():
            self.log.warning("CoreML model not found. Will use CPU fallback.")
            self.coreml_enabled = False

    # Load the model
    self._model = pywhispercpp.model.Model(
        self.model_name,                    # "medium-q5_0"
        models_dir=str(self.models_dir),    # {project_root}/models/
        n_threads=self.n_threads,           # 4
    )
```

### 4. Library's Model.__init__(): `pywhispercpp/model.py:68-96`

```python
class Model:
    def __init__(self, model='tiny', models_dir=None, ...):
        if Path(model).is_file():
            self.model_path = model
        else:
            # Download the model if not already present
            self.model_path = utils.download_model(model, models_dir)
            # → calls download_model("medium-q5_0", "{project_root}/models/")

        # Initialize whisper.cpp context
        self._ctx = pw.whisper_init_from_file(self.model_path)
```

### 5. Library's download_model(): `pywhispercpp/utils.py:29-73`

```python
def download_model(model_name: str, download_dir=None, chunk_size=1024) -> str:
    # Validate model name against AVAILABLE_MODELS list
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"Invalid model name `{model_name}`")
        return

    if download_dir is None:
        download_dir = MODELS_DIR  # ~/Library/Application Support/pywhispercpp/models/

    os.makedirs(download_dir, exist_ok=True)

    # Construct download URL
    url = _get_model_url(model_name=model_name)
    # → https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin

    file_path = Path(download_dir) / os.path.basename(url)
    # → {project_root}/models/ggml-medium-q5_0.bin

    # Check if the file is already there
    if file_path.exists():
        logger.info(f"Model {model_name} already exists in {download_dir}")
    else:
        # Download from HuggingFace (direct HTTP, not HF Hub API)
        resp = requests.get(url, stream=True)
        with open(file_path, 'wb') as file:
            for data in resp.iter_content(chunk_size=chunk_size):
                file.write(data)

    return str(file_path.absolute())
```

### 6. URL Construction: `pywhispercpp/constants.py:15-18`

```python
MODELS_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp"
MODELS_PREFIX_URL = "resolve/main/ggml"

# URL pattern: {MODELS_BASE_URL}/{MODELS_PREFIX_URL}-{model_name}.bin
# Example: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin
```

### 7. Available Models: `pywhispercpp/constants.py:27-61`

```python
AVAILABLE_MODELS = [
    "base", "base-q5_1", "base-q8_0",
    "base.en", "base.en-q5_1", "base.en-q8_0",
    "large-v1", "large-v2", "large-v2-q5_0", "large-v2-q8_0",
    "large-v3", "large-v3-q5_0",
    "large-v3-turbo", "large-v3-turbo-q5_0", "large-v3-turbo-q8_0",
    "medium", "medium-q5_0", "medium-q8_0",
    "medium.en", "medium.en-q5_0", "medium.en-q8_0",
    "small", "small-q5_1", "small-q8_0",
    "small.en", "small.en-q5_1", "small.en-q8_0",
    "tiny", "tiny-q5_1", "tiny-q8_0",
    "tiny.en", "tiny.en-q5_1", "tiny.en-q8_0",
]
```

### 8. Cache Behavior

| Aspect | Location |
|--------|----------|
| **GGML Model** | `{project_root}/models/ggml-medium-q5_0.bin` |
| **CoreML Encoder** | `{project_root}/models/ggml-medium-encoder.mlmodelc/` |
| **Default pywhispercpp Cache** | `~/Library/Application Support/pywhispercpp/models/` (NOT USED - overridden) |

**Flow:**
```
HuggingFace Direct URL (https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin)
      ↓ requests.get() stream download
{project_root}/models/ggml-medium-q5_0.bin
      ↓ whisper_init_from_file()
whisper.cpp context loaded
```

**Note:** This implementation does NOT use the HuggingFace Hub API. It downloads directly via HTTP from the `ggerganov/whisper.cpp` repository's resolved file URLs.

### 9. Transcription-Time Model Usage: `coreml.py:101-128`

When `transcribe()` is called:

```python
async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    # Model already loaded in load_model()
    # Uses CoreML acceleration if encoder is present and WHISPER_COREML=1

    # Transcribe with automatic language detection
    segments = self._model.transcribe(audio, language=None, translate=False)
    text = " ".join([segment.text for segment in segments])

    return TranscriptionResult(
        text=text,
        segments=segments,
        language=None,  # Language detection handled by whisper.cpp
    )
```

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Mapped Model Name** | `medium-q5_0` (via `_get_model_map()`) |
| **Download URL** | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin` |
| **GGML Model Path** | `{project_root}/models/ggml-medium-q5_0.bin` |
| **CoreML Encoder Path** | `{project_root}/models/ggml-medium-encoder.mlmodelc/` (optional) |
| **Default HF Cache** | NOT USED |
| **pywhispercpp Default Cache** | NOT USED (overridden by `models_dir` parameter) |
| **Backend** | whisper.cpp (C++) with Python bindings |
| **Quantization** | q5_0 or q5_1 (5-bit GGML quantization) |
| **Threading** | 4 threads |

---

## Model Mapping Reference

### Implementation's `_get_model_map()`: `coreml.py:35-47`

Maps standard Whisper model names to GGML quantized variants:

```python
def _get_model_map(self) -> Dict[str, str]:
    return {
        "tiny": "tiny-q5_1",          # 5-bit quantized
        "base": "base-q5_1",          # 5-bit quantized
        "small": "small",             # NOT quantized (full precision)
        "medium": "medium-q5_0",      # 5-bit quantized
        "large": "large-v3-turbo-q5_0",  # large-v3-turbo + 5-bit quantized
    }
```

### Implementation's `_get_coreml_model_map()`: `coreml.py:49-61`

Maps standard Whisper model names to CoreML encoder names:

```python
def _get_coreml_model_map(self) -> Dict[str, str]:
    return {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v3-turbo",
    }
```

### Comparison Table

| Model | GGML Name (`_get_model_map()`) | CoreML Encoder Name | GGML File | CoreML File |
|-------|--------------------------------|---------------------|-----------|-------------|
| `tiny` | `tiny-q5_1` | `tiny` | `ggml-tiny-q5_1.bin` | `ggml-tiny-encoder.mlmodelc` |
| `base` | `base-q5_1` | `base` | `ggml-base-q5_1.bin` | `ggml-base-encoder.mlmodelc` |
| `small` | `small` | `small` | `ggml-small.bin` | `ggml-small-encoder.mlmodelc` |
| `medium` | `medium-q5_0` | `medium` | `ggml-medium-q5_0.bin` | `ggml-medium-encoder.mlmodelc` |
| `large` | `large-v3-turbo-q5_0` | `large-v3-turbo` | `ggml-large-v3-turbo-q5_0.bin` | `ggml-large-v3-turbo-encoder.mlmodelc` |

---

## Notes

### CoreML Acceleration

CoreML acceleration is enabled via environment variable:

```python
# Set at module load time (coreml.py:5-6)
os.environ["WHISPER_COREML"] = "1"
```

**CoreML acceleration requires:**
1. Environment variable `WHISPER_COREML=1`
2. CoreML encoder model at `{models_dir}/ggml-{model}-encoder.mlmodelc/`
3. macOS with Apple Silicon

**CoreML encoder files are NOT auto-downloaded.** They must be obtained manually from:
- whisper.cpp releases: https://github.com/ggerganov/whisper.cpp/releases
- Or generated using whisper.cpp's `coreml-encode.py` script

### Quantization Note

The "small" model is NOT quantized in `_get_model_map()`:

```python
"small": "small",  # Full precision, not quantized
```

This is intentional - the small model uses full precision for potentially better quality, while other sizes use q5_0 or q5_1 quantization for reduced file size and faster inference.

### File Sizes (Approximate)

| Model | GGML Size | CoreML Encoder Size | Combined |
|-------|-----------|---------------------|----------|
| `tiny-q5_1` | ~31 MB | ~16 MB | ~47 MB |
| `base-q5_1` | ~142 MB | ~108 MB | ~250 MB |
| `small` | ~466 MB | ~234 MB | ~700 MB |
| `medium-q5_0` | ~1500 MB | ~700 MB | ~2200 MB |
| `large-v3-turbo-q5_0` | ~547 MB | ~1200 MB | ~1747 MB |

### Two Cache Types

Unlike HuggingFace-based implementations, this implementation has two distinct file types:

1. **GGML Model** (auto-downloaded): The main model weights in GGML format
2. **CoreML Encoder** (manual download): Optional Apple Neural Engine acceleration

---

## check-models Command Flow

This section traces the execution flow for:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations WhisperCppCoreMLImplementation
```

### 1. CLI Entry Point: `cli.py:101-161`

```python
@app.command(name="check-models")
def check_models(model="medium", implementations="WhisperCppCoreMLImplementation", ...):
    # Filter to just WhisperCppCoreMLImplementation
    impls_to_check = [WhisperCppCoreMLImplementation]

    # Create checker and check all models
    checker = ModelChecker(verify_method=verify_method, verbose=verbose)
    statuses = checker.check_all_models("medium", impls_to_check)
```

### 2. ModelChecker.check_all_models(): `check_models.py:171-216`

```python
def check_all_models(self, model_size="medium", implementations):
    for impl_class in implementations:
        impl = WhisperCppCoreMLImplementation()

        # Get model info from implementation
        model_info = impl.get_model_info("medium")

        # Check HuggingFace cache (default location)
        hf_status, hf_size = self._check_hf_cache(model_info, impl, "medium")

        # Check local cache (custom location)
        local_status, local_size = self._check_local_cache(model_info, impl, "medium")
```

### 3. WhisperCppCoreMLImplementation.get_model_info(): `coreml.py:138-179`

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    from pathlib import Path
    from mac_whisper_speedtest.utils import get_models_dir

    # Use base class helper for GGML model name (single source of truth)
    ggml_model_name = self._map_model_name(model_name)
    # → for "medium": "medium-q5_0"

    # Use CoreML helper for CoreML encoder name
    coreml_model_map = self._get_coreml_model_map()
    coreml_model_name = coreml_model_map.get(model_name, model_name)
    # → for "medium": "medium"

    models_dir = Path(get_models_dir())

    # Cache paths include GGML model + optional CoreML encoder
    cache_paths = [
        models_dir / f"ggml-{ggml_model_name}.bin",
        # → models/ggml-medium-q5_0.bin
        models_dir / f"ggml-{coreml_model_name}-encoder.mlmodelc",
        # → models/ggml-medium-encoder.mlmodelc
    ]

    # Expected sizes (GGML + CoreML encoder combined)
    size_map = {
        "tiny": 50,
        "base": 250,
        "small": 700,
        "medium": 2200,
        "large": 1800,
    }

    return ModelInfo(
        model_name=f"{ggml_model_name} + CoreML",  # Display name
        repo_id=None,                              # No HF repo (direct HTTP download)
        cache_paths=cache_paths,
        expected_size_mb=size_map.get(model_name, 100),
        verification_method="size",                # Size-based verification
        download_trigger="manual",                 # Manual download required
        timeout_seconds=30 if "large" in model_name else 15
    )
```

### 4. HuggingFace Cache Check: `check_models.py:218-238`

```python
def _check_hf_cache(self, model_info, impl_instance, model_name):
    if not model_info.repo_id:
        # No repo_id means not a HuggingFace model
        return "n/a", None
```

**Result:** Returns "n/a" because `repo_id=None` in ModelInfo.

### 5. Local Cache Check: `check_models.py:337-384`

```python
def _check_local_cache(self, model_info, impl_instance, model_name):
    # No custom HF cache dir (hf_cache_dir is not set)
    # Check cache_paths directly

    if not model_info.cache_paths:
        return "n/a", None

    # Check if all cache paths exist
    all_exist = all(path.exists() for path in model_info.cache_paths)
    # → Checks:
    #   - models/ggml-medium-q5_0.bin
    #   - models/ggml-medium-encoder.mlmodelc

    if not all_exist:
        any_exist = any(path.exists() for path in model_info.cache_paths)
        if any_exist:
            return "incomplete", self._calculate_total_size(model_info.cache_paths)
        return "missing", None

    # All paths exist, calculate size
    total_size_mb = self._calculate_total_size(model_info.cache_paths)

    # For local path-based models, verify by loading
    is_complete = self._verify_by_loading(impl_instance, model_name)

    if is_complete != "complete":
        return is_complete, total_size_mb

    # Verify size if expected size is provided
    if model_info.expected_size_mb and model_info.verification_method == "size":
        expected = model_info.expected_size_mb  # 2200 MB for medium
        if total_size_mb < expected * 0.9:      # Allow 10% variance
            return "incomplete", total_size_mb

    return "complete", total_size_mb
```

### 6. Verification by Loading: `check_models.py:299-335`

```python
def _verify_by_loading(self, impl_instance, model_name):
    model_info = impl_instance.get_model_info(model_name)
    timeout = self._calculate_timeout(model_name, model_info)  # 15 seconds for medium

    # Uses timeout-protected loading
    status, _ = self._verify_with_timeout(impl_instance, "medium", timeout)
    # → Attempts impl_instance.load_model("medium")
    # → If succeeds within timeout: "complete"
    # → If times out (would trigger download): "incomplete"
    return status
```

---

## check-models Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **get_model_info() returns** | `medium-q5_0 + CoreML` (display name) |
| **repo_id** | `None` (not HuggingFace-based) |
| **HF Cache Status** | `n/a` (no repo_id) |
| **Local Cache Paths Checked** | `models/ggml-medium-q5_0.bin`, `models/ggml-medium-encoder.mlmodelc` |
| **Verification Method** | `size` (with load-based verification) |
| **Download Trigger** | `manual` (user must download manually) |
| **Expected Size** | 2200 MB (GGML + CoreML combined) |

---

## Consistency Analysis

### How `load_model()` Maps Model Names

1. `load_model("medium")` calls `self._map_model_name(model_name)`
2. `_map_model_name()` uses `_get_model_map()` → "medium-q5_0"
3. Passes "medium-q5_0" to `pywhispercpp.model.Model(...)`
4. pywhispercpp downloads from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin`

### How `get_model_info()` Maps Model Names

1. `get_model_info("medium")` calls `self._map_model_name(model_name)`
2. `_map_model_name()` uses `_get_model_map()` → "medium-q5_0"
3. Returns cache path: `models/ggml-medium-q5_0.bin`

### Verification: Model Mapping Consistency

| Model | `load_model()` GGML Name | `get_model_info()` GGML Name | Match? |
|-------|--------------------------|------------------------------|--------|
| `tiny` | `tiny-q5_1` | `tiny-q5_1` | YES |
| `base` | `base-q5_1` | `base-q5_1` | YES |
| `small` | `small` | `small` | YES |
| `medium` | `medium-q5_0` | `medium-q5_0` | YES |
| `large` | `large-v3-turbo-q5_0` | `large-v3-turbo-q5_0` | YES |

**Result:** Both methods use the same `_map_model_name()` helper, ensuring consistency. The implementation correctly follows the standardized pattern.

---

## No Known Issues

The `WhisperCppCoreMLImplementation` correctly implements the standardized pattern:

1. **Single source of truth**: Both `load_model()` and `get_model_info()` use `_map_model_name()` which calls `_get_model_map()`
2. **Consistent cache paths**: Both use `get_models_dir()` as the custom cache location
3. **Separate CoreML mapping**: `_get_coreml_model_map()` handles CoreML encoder naming (different from GGML names for large model)

**Result:**
- `check-models` correctly identifies cached models
- Local cache verification works properly
- Size-based verification with load confirmation

### Why `download_trigger="manual"`

Unlike HuggingFace-based implementations that can auto-download, this implementation requires manual download because:

1. **GGML models** are auto-downloaded by pywhispercpp, BUT
2. **CoreML encoders** are NOT available via pywhispercpp and must be obtained separately
3. The combined setup (GGML + CoreML) requires manual intervention

If only GGML is needed (no CoreML acceleration), the implementation will still work - it falls back to CPU-only inference.

---

## Actual Files in models/ Directory

Based on the project's current state:

```
models/
├── ggml-tiny-q5_1.bin                    # GGML model (tiny)
├── ggml-tiny-encoder.mlmodelc/           # CoreML encoder (tiny)
├── ggml-small.bin                        # GGML model (small, NOT quantized)
├── ggml-small-encoder.mlmodelc/          # CoreML encoder (small)
├── ggml-large-v3-turbo-q5_0.bin          # GGML model (large)
├── ggml-large-v3-turbo-encoder.mlmodelc/ # CoreML encoder (large)
└── ... (other implementation models)
```

Note: The `medium` model files (`ggml-medium-q5_0.bin` and `ggml-medium-encoder.mlmodelc`) are NOT currently present in the models directory.
