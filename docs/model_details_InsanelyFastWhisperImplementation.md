# InsanelyFastWhisperImplementation - Model Download & Cache Details

This document traces the execution flow, download URLs, and cache locations for the `InsanelyFastWhisperImplementation`.

## Benchmark Execution Flow

This section traces the execution flow for:
```bash
.venv/bin/python3 test_benchmark.py medium 1 InsanelyFastWhisperImplementation
```

### 1. Entry Point: `test_benchmark.py`

```
test_benchmark.py:78-80
↓
model = "medium"
runs = 1
implementations = "InsanelyFastWhisperImplementation"
↓
asyncio.run(main(model, runs, implementations))
```

### 2. Benchmark Runner: `benchmark.py:146`

```
benchmark.py:146
implementation.load_model("medium")
```

### 3. Insanely Fast Implementation: `insanely.py:76-148`

```python
def load_model(self, model_name: str) -> None:
    import torch
    from transformers.pipelines import pipeline

    # Use base class helper to get model repo ID
    self.model_name = self._map_model_name(model_name)
    # → calls _get_model_map() → returns "openai/whisper-medium"

    # Apple Silicon optimizations
    attn_implementation = "sdpa"  # Scaled Dot Product Attention for MPS
    model_kwargs = {
        "attn_implementation": attn_implementation,
        "use_cache": True,
        "low_cpu_mem_usage": True
    }

    # Load using transformers pipeline (NOT insanely_fast_whisper library!)
    self._model = pipeline(
        "automatic-speech-recognition",
        model=self.model_name,  # "openai/whisper-medium"
        torch_dtype=torch.float16,
        device="mps",  # Apple Silicon GPU
        model_kwargs=model_kwargs,
    )
```

### 4. Model Mapping: `insanely.py:150-164`

```python
def _get_model_map(self) -> Dict[str, str]:
    return {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",  # ← SELECTED
        "large": "openai/whisper-large-v3-turbo",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
    }
```

### 5. Transformers Pipeline Model Download

The `transformers.pipeline()` function automatically downloads the model from HuggingFace Hub:

```
https://huggingface.co/openai/whisper-medium/resolve/main/config.json
https://huggingface.co/openai/whisper-medium/resolve/main/model.safetensors
https://huggingface.co/openai/whisper-medium/resolve/main/generation_config.json
https://huggingface.co/openai/whisper-medium/resolve/main/preprocessor_config.json
https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json
https://huggingface.co/openai/whisper-medium/resolve/main/vocab.json
https://huggingface.co/openai/whisper-medium/resolve/main/merges.txt
https://huggingface.co/openai/whisper-medium/resolve/main/added_tokens.json
https://huggingface.co/openai/whisper-medium/resolve/main/normalizer.json
https://huggingface.co/openai/whisper-medium/resolve/main/special_tokens_map.json
```

### 6. Cache Behavior (Default HuggingFace Cache)

| Aspect | Location |
|--------|----------|
| **Default HF Cache** | `~/.cache/huggingface/hub/models--openai--whisper-medium/` |
| **Snapshot Directory** | `~/.cache/huggingface/hub/models--openai--whisper-medium/snapshots/{revision}/` |

**Flow:**
```
HuggingFace CDN (https://huggingface.co/openai/whisper-medium/...)
      ↓ transformers auto-download
~/.cache/huggingface/hub/models--openai--whisper-medium/blobs/
      ↓ symlinked to
~/.cache/huggingface/hub/models--openai--whisper-medium/snapshots/{revision}/model.safetensors
~/.cache/huggingface/hub/models--openai--whisper-medium/snapshots/{revision}/config.json
~/.cache/huggingface/hub/models--openai--whisper-medium/snapshots/{revision}/tokenizer.json
(... and other files)
```

**Note:** This implementation uses the DEFAULT HuggingFace cache (`~/.cache/huggingface/hub/`), NOT a custom cache directory.

### 7. Transcription-Time Model Usage: `insanely.py:166-233`

When `transcribe()` is called:

```python
async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    # Save audio to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        sf.write(temp_file.name, audio, 16000, format="WAV")

        # Apple Silicon optimization: reduced chunk length for memory efficiency
        chunk_length = 20  # vs 30 on other platforms

        # Run transcription using the loaded pipeline
        result = self._model(
            temp_file.name,
            chunk_length_s=chunk_length,
            batch_size=self.batch_size,  # Adaptive based on system memory
            return_timestamps=False,
            return_language=True,
            generate_kwargs={"task": "transcribe"}
        )

    return TranscriptionResult(text=result.get("text", ""), ...)
```

## Summary Table

| Aspect | Value |
|--------|-------|
| **Requested Model** | `medium` |
| **Actual Repo Used** | `openai/whisper-medium` (Official OpenAI repo) |
| **Download URL Pattern** | `https://huggingface.co/openai/whisper-medium/resolve/main/{file}` |
| **Default HF Cache Location** | `~/.cache/huggingface/hub/models--openai--whisper-medium/` |
| **Custom Cache Location** | NOT USED (uses default HF cache) |
| **Files Downloaded** | `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.json`, `merges.txt`, `preprocessor_config.json`, `generation_config.json`, `added_tokens.json`, `normalizer.json`, `special_tokens_map.json` |
| **Backend Library** | `transformers.pipeline` (PyTorch MPS) |
| **Attention Implementation** | SDPA (Scaled Dot Product Attention) |

## Model Mapping Reference

The `InsanelyFastWhisperImplementation` uses this model mapping in `_get_model_map()`:
`src/mac_whisper_speedtest/implementations/insanely.py:150-164`

| Model Name | HuggingFace Repo |
|------------|------------------|
| `tiny` | `openai/whisper-tiny` |
| `base` | `openai/whisper-base` |
| `small` | `openai/whisper-small` |
| `medium` | `openai/whisper-medium` |
| `large` | `openai/whisper-large-v3-turbo` |
| `large-v2` | `openai/whisper-large-v2` |
| `large-v3` | `openai/whisper-large-v3` |

## Notes

### Library Clarification

**Important:** Despite the name, this implementation does NOT use the `insanely_fast_whisper` Python library code for transcription. Instead:

1. The check for `insanely_fast_whisper` in `implementations/__init__.py` is just a proxy to verify that the necessary dependencies (transformers, torch, etc.) are installed.

2. The actual implementation uses `transformers.pipelines.pipeline` directly:
   ```python
   from transformers.pipelines import pipeline
   self._model = pipeline("automatic-speech-recognition", model=self.model_name, ...)
   ```

3. The `insanely-fast-whisper` package itself is also just a CLI wrapper around `transformers.pipeline` (see `.venv/lib/python3.12/site-packages/insanely_fast_whisper/cli.py`).

### Apple Silicon Optimizations

| Optimization | Value | Purpose |
|--------------|-------|---------|
| `attn_implementation` | `"sdpa"` | SDPA is more optimized for MPS than flash_attention_2 |
| `torch_dtype` | `torch.float16` | Half-precision for better GPU performance |
| `batch_size` | Adaptive (10-16) | Based on available system memory |
| `chunk_length_s` | 20 | Reduced from 30 for better memory efficiency on MPS |
| `use_cache` | True | Enable KV cache for better performance |
| `low_cpu_mem_usage` | True | Optimize for unified memory architecture |
| `device` | `"mps"` | Apple Metal Performance Shaders GPU backend |

### Quantization Note

The implementation attempts 4-bit quantization via BitsAndBytesConfig, but this is NOT supported on macOS:
```python
# Note: bitsandbytes is not supported on macOS, so 4-bit quantization will be skipped on Apple Silicon
```

The `quantization="4bit"` parameter is set but effectively ignored on Apple Silicon systems.

---

## check-models Command Flow

This section traces the execution flow for:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations InsanelyFastWhisperImplementation
```

### 1. CLI Entry Point: `cli.py:101-161`

```python
@app.command(name="check-models")
def check_models(model="medium", implementations="InsanelyFastWhisperImplementation", ...):
    # Filter to just InsanelyFastWhisperImplementation
    impls_to_check = [InsanelyFastWhisperImplementation]

    # Create checker and check all models
    checker = ModelChecker(verify_method=verify_method, verbose=verbose)
    statuses = checker.check_all_models("medium", impls_to_check)
```

### 2. ModelChecker.check_all_models(): `check_models.py:171-216`

```python
def check_all_models(self, model_size="medium", implementations):
    for impl_class in implementations:
        impl = InsanelyFastWhisperImplementation()

        # Get model info from implementation
        model_info = impl.get_model_info("medium")

        # Check HuggingFace cache (default location)
        hf_status, hf_size = self._check_hf_cache(model_info, impl, "medium")

        # Check local cache
        local_status, local_size = self._check_local_cache(model_info, impl, "medium")
```

### 3. InsanelyFastWhisperImplementation.get_model_info(): `insanely.py:245-258`

```python
def get_model_info(self, model_name: str) -> ModelInfo:
    # Use the same model mapping as load_model() to ensure consistency
    repo_id = self._map_model_name(model_name)
    # → "openai/whisper-medium"

    return ModelInfo(
        model_name=model_name,  # "medium"
        repo_id=repo_id,  # "openai/whisper-medium"
        cache_paths=[],
        expected_size_mb=None,  # Will be determined by HF verification
        verification_method="huggingface",
        download_trigger="auto",
        timeout_seconds=30 if "large" in model_name else 15  # 15 for medium
    )
```

### 4. HuggingFace Cache Check: `check_models.py:218-238`

```python
def _check_hf_cache(self, model_info, impl_instance, model_name):
    if model_info.verification_method == "huggingface":
        # Checks DEFAULT HF cache (~/.cache/huggingface/hub/)
        return self._verify_hf_model(
            repo_id="openai/whisper-medium",
            cache_dir=None,  # Default cache
            impl_instance=impl_instance,
            model_name="medium"
        )
```

### 5. Local Cache Check: `check_models.py:337-384`

```python
def _check_local_cache(self, model_info, impl_instance, model_name):
    # No custom hf_cache_dir specified for InsanelyFast
    # No cache_paths specified
    # Returns "n/a" since this impl uses default HF cache only
    if not model_info.cache_paths:
        return "n/a", None
```

### 6. HF Model Verification: `check_models.py:240-297`

```python
def _verify_hf_model(self, repo_id, cache_dir, impl_instance, model_name):
    from huggingface_hub import scan_cache_dir

    # Scan the default HF cache
    cache_info = scan_cache_dir()  # ~/.cache/huggingface/hub/

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:  # Looking for "openai/whisper-medium"
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
| **get_model_info() returns** | `openai/whisper-medium` |
| **load_model() uses** | `openai/whisper-medium` (SAME - no mismatch!) |
| **Default HF Cache Check** | `~/.cache/huggingface/hub/` → shows actual status |
| **Local Cache Check** | N/A (no custom cache directory) |
| **Verification Method** | `huggingface` (with load-based verification) |
| **Download Trigger** | `auto` (uses transformers auto-download) |
| **Timeout** | 15 seconds (30 for large models) |

---

## Consistency Note

`InsanelyFastWhisperImplementation` correctly follows the standardized pattern:

| Method | Repo for "medium" |
|--------|-------------------|
| `get_model_info()` via `_map_model_name()` | `openai/whisper-medium` |
| `load_model()` via `_map_model_name()` | `openai/whisper-medium` |

**Result:** Both methods use the same `_get_model_map()` dictionary, ensuring:
- `check-models` correctly identifies cached models
- No false "missing" reports for downloaded models
- Consistent verification with actual benchmark behavior

This implementation correctly follows the recommended pattern where `_get_model_map()` is the single source of truth for model mappings.

---

## Comparison with insanely-fast-whisper Library

The `insanely-fast-whisper` Python package (`.venv/lib/python3.12/site-packages/insanely_fast_whisper/cli.py`) is simply a CLI wrapper that does essentially the same thing:

```python
# From insanely_fast_whisper/cli.py
pipe = pipeline(
    "automatic-speech-recognition",
    model=args.model_name,
    torch_dtype=torch.float16,
    device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
    model_kwargs={"attn_implementation": "flash_attention_2"} if args.flash else {"attn_implementation": "sdpa"},
)
```

The key differences in our implementation:
1. Uses SDPA always on macOS (more optimized for MPS than flash_attention_2)
2. Adds Apple Silicon-specific optimizations (`use_cache`, `low_cpu_mem_usage`)
3. Adaptive batch sizing based on system memory
4. Reduced chunk length (20s vs 30s) for better memory efficiency on MPS
