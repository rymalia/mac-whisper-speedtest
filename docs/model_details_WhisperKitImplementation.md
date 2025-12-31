# Model Details: WhisperKitImplementation

This document traces the execution flow for `WhisperKitImplementation`, documenting how models are loaded, downloaded, and verified. This implementation uses a Swift bridge to the native WhisperKit framework by Argmax, providing CoreML-accelerated Whisper transcription on Apple Silicon.

## Architecture Overview

| Aspect | Value |
|--------|-------|
| **Backend** | WhisperKit (Swift framework by Argmax) |
| **Integration** | Swift bridge via subprocess |
| **Model Format** | CoreML compiled models (.mlmodelc) |
| **Download Source** | HuggingFace via Swift HubApi (not Python huggingface_hub) |
| **Model Repository** | `argmaxinc/whisperkit-coreml` |
| **Cache Location** | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/` |
| **Acceleration** | CoreML + Apple Neural Engine |

---

## Benchmark Execution Flow

Command traced:
```bash
.venv/bin/python3 test_benchmark.py medium 1 WhisperKitImplementation
```

### Step-by-Step Execution

1. **Entry Point** (`test_benchmark.py:76-82`)
   - Parses arguments: `model="medium"`, `runs=1`, `implementations="WhisperKitImplementation"`
   - Loads audio from `tools/whisperkit-bridge/.build/checkouts/WhisperKit/Tests/WhisperKitTests/Resources/jfk.wav`
   - Calls `asyncio.run(main(model, runs, implementations))`

2. **Implementation Discovery** (`test_benchmark.py:46-62`)
   - Calls `get_all_implementations()` from `implementations/__init__.py`
   - Filters to `[WhisperKitImplementation]` based on command-line argument

3. **Benchmark Runner** (`benchmark.py:135-146`)
   - Creates instance: `implementation = WhisperKitImplementation()`
   - `__init__()` finds Swift bridge at: `{project_root}/tools/whisperkit-bridge/.build/release/whisperkit-bridge`
   - Calls `implementation.load_model("medium")`

4. **Python load_model()** (`whisperkit.py:117-157`)
   - Uses `_map_model_name("medium")` → `_get_model_map()` → returns `"medium"`
   - Sets `self.model_name = "medium"`
   - Calls `_ensure_weights_directories()` to pre-create CoreML weights directories (workaround for WhisperKit download issue)
   - Tests bridge with `subprocess.run([bridge_path, "--help"])`

5. **Transcription** (`benchmark.py:155-167`)
   - Calls `await implementation.transcribe(config.audio_data)`

6. **Python transcribe()** (`whisperkit.py:159-239`)
   - Preprocesses audio (16kHz mono, float32, normalized)
   - Saves to temporary WAV file
   - Calls Swift bridge:
     ```
     whisperkit-bridge /tmp/audio.wav --format json --model medium
     ```
   - Parses JSON result with transcription text and timing

7. **Swift Bridge** (`tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift:23-76`)
   - Creates `WhisperKitConfig(model: "medium")`
   - Initializes `WhisperKit(config)` → triggers model download/load

8. **WhisperKit Initialization** (`WhisperKit.swift:53-90`)
   - Calls `setupModels(model: "medium", ...)`

9. **Model Setup** (`WhisperKit.swift:298-331`)
   - Sets `repo = modelRepo ?? "argmaxinc/whisperkit-coreml"` (default)
   - Calls `WhisperKit.download(variant: "medium", from: repo)`

10. **Model Download** (`WhisperKit.swift:243-295`)
    - Uses `HubApi` from swift-transformers package
    - Searches for models matching `*medium/*` in repo
    - If ambiguous, refines to `*openai*medium/*`
    - Finds folder: `openai_whisper-medium/`
    - Downloads via `hubApi.snapshot()`

11. **HubApi Download** (`HubApi.swift:175-178, 234-257`)
    - Default download location: `~/Documents/huggingface/` (from `FileManager.urls(for: .documentDirectory)`)
    - Full path: `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-medium/`
    - Downloads all model files to this location

12. **Model Loading** (`WhisperKit.swift:337-458`)
    - Loads CoreML models from downloaded folder:
      - `MelSpectrogram.mlmodelc` - Feature extractor
      - `AudioEncoder.mlmodelc` - Audio encoder
      - `TextDecoder.mlmodelc` - Text decoder
    - Loads tokenizer from `tokenizer.json`

13. **Transcription Processing**
    - WhisperKit processes audio through full pipeline
    - Returns JSON with `text`, `transcription_time`, `language`, `segments`

---

## Summary Table

| Requested Model | Mapped Model Name | Model Directory | Cache Path |
|-----------------|-------------------|-----------------|------------|
| `tiny` | `tiny` | `openai_whisper-tiny` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-tiny/` |
| `base` | `base` | `openai_whisper-base` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-base/` |
| `small` | `small` | `openai_whisper-small` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/` |
| `medium` | `medium` | `openai_whisper-medium` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-medium/` |
| `large` | `large-v3` | `openai_whisper-large-v3` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/` |
| `large-v3` | `large-v3` | `openai_whisper-large-v3` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/` |
| `large-v3-turbo` | `large-v3-turbo` | `openai_whisper-large-v3-turbo` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3-turbo/` |

### Files Downloaded

For each model variant, WhisperKit downloads:

| File/Directory | Description |
|----------------|-------------|
| `AudioEncoder.mlmodelc/` | CoreML-compiled audio encoder model |
| `TextDecoder.mlmodelc/` | CoreML-compiled text decoder model |
| `MelSpectrogram.mlmodelc/` | CoreML-compiled mel spectrogram extractor |
| `config.json` | Model configuration |
| `tokenizer.json` | Tokenizer vocabulary and settings |
| `generation_config.json` | Text generation configuration |
| `merges.txt` | BPE merge rules |
| `vocab.json` | Vocabulary mapping |

### Expected Sizes

| Model | Expected Size (MB) |
|-------|-------------------|
| `tiny` | 76 |
| `small` | 467 |
| `medium` | 1400 |
| `large-v3` | 2900 |
| `distil-large-v3` | 229 |

---

## Model Mapping Reference

From `whisperkit.py:_get_model_map()`:

```python
{
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3",        # Note: "large" maps to "large-v3"
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "large-turbo": "large-v3-turbo"
}
```

---

## Notes

### Cache Location Differences

WhisperKit uses a **custom cache location** that differs from the standard HuggingFace cache:

| Cache Type | Location |
|------------|----------|
| **Standard HF cache** | `~/.cache/huggingface/hub/` |
| **WhisperKit cache** | `~/Documents/huggingface/models/` |

This is because WhisperKit's Swift `HubApi` uses `FileManager.urls(for: .documentDirectory)` as the default download base, which resolves to `~/Documents/` on macOS.

### Swift Bridge Architecture

```
Python                          Swift
───────────────────────────────────────────────────────
WhisperKitImplementation        whisperkit-bridge
        │                              │
  load_model()                         │
        │                              │
  transcribe() ──────────────────► main.swift
        │                              │
        │                       WhisperKitConfig(model:)
        │                              │
        │                       WhisperKit(config)
        │                              │
        │                       setupModels()
        │                              │
        │                       download() via HubApi
        │                              │
        │                       loadModels()
        │                              │
   parse JSON ◄─────────────── transcribe() → JSON
```

### Pre-created Weights Directories

The Python implementation calls `_ensure_weights_directories()` before each transcription to work around a WhisperKit download issue:

- **Problem**: Incomplete downloads leave partial model structures without `weights/` subdirectories
- **Symptom**: Subsequent downloads succeed but fail to move files to non-existent directories
- **Workaround**: Pre-create `weights/` directories for all three CoreML model components

Directories created:
- `{model_cache_path}/AudioEncoder.mlmodelc/weights/`
- `{model_cache_path}/TextDecoder.mlmodelc/weights/`
- `{model_cache_path}/MelSpectrogram.mlmodelc/weights/`

### Internal Timing

The Swift bridge reports internal transcription time via `timings.fullPipeline` to exclude:
- Audio file loading overhead
- Subprocess communication overhead
- JSON serialization overhead

This provides more accurate benchmarking by measuring only the actual transcription processing.

---

## check-models Command Flow

Command traced:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations WhisperKitImplementation
```

### Step-by-Step Execution

1. **CLI Entry Point** (`cli.py:101-161`)
   - Parses arguments: `model="medium"`, `implementations="WhisperKitImplementation"`
   - Filters implementations to `[WhisperKitImplementation]`
   - Creates `ModelChecker()`

2. **Check All Models** (`check_models.py:171-216`)
   - Iterates through implementations
   - Creates instance: `impl = WhisperKitImplementation()`
   - Calls `model_info = impl.get_model_info("medium")`

3. **get_model_info()** (`whisperkit.py:290-338`)
   - Uses `_map_model_name("medium")` → returns `"medium"` (same as load_model)
   - Builds model directory name: `"openai_whisper-medium"`
   - Constructs cache path: `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-medium`
   - Returns `ModelInfo`:
     ```python
     ModelInfo(
         model_name="medium",
         repo_id=None,  # WhisperKit uses its own download mechanism
         cache_paths=[
             model_cache_path / "AudioEncoder.mlmodelc",
             model_cache_path / "TextDecoder.mlmodelc",
             model_cache_path / "MelSpectrogram.mlmodelc",
         ],
         expected_size_mb=1400,
         verification_method="size",
         download_trigger="bridge"
     )
     ```

4. **Check HF Cache** (`check_models.py:218-238`)
   - `model_info.repo_id` is `None`
   - Returns `("n/a", None)` - no HF cache check for WhisperKit

5. **Check Local Cache** (`check_models.py:337-384`)
   - No `hf_cache_dir` specified
   - Checks if all `cache_paths` exist:
     - `AudioEncoder.mlmodelc/`
     - `TextDecoder.mlmodelc/`
     - `MelSpectrogram.mlmodelc/`
   - If all exist, calls `_verify_by_loading(impl, "medium")`
   - Uses timeout-protected loading to verify model completeness
   - Verifies total size against `expected_size_mb` (±10% tolerance)

6. **Display Results** (`check_models.py:407-451`)
   - Prints status table with:
     - HF Hub Cache: "n/a" (WhisperKit doesn't use standard HF cache)
     - Local Cache: "complete"/"missing"/"incomplete"
     - Disk Usage (MB)

---

## check-models Summary Table

| Model | Verification Method | Expected Size (MB) | Cache Paths Checked |
|-------|--------------------|--------------------|---------------------|
| `tiny` | `size` | 76 | AudioEncoder, TextDecoder, MelSpectrogram |
| `base` | `size` | - | AudioEncoder, TextDecoder, MelSpectrogram |
| `small` | `size` | 467 | AudioEncoder, TextDecoder, MelSpectrogram |
| `medium` | `size` | 1400 | AudioEncoder, TextDecoder, MelSpectrogram |
| `large-v3` | `size` | 2900 | AudioEncoder, TextDecoder, MelSpectrogram |

---

## Consistency Analysis: No Variant Mismatch

Both `load_model()` and `get_model_info()` use the standardized base class pattern:

| Component | Method | Result for "medium" |
|-----------|--------|---------------------|
| `load_model("medium")` | Uses `_map_model_name()` → `_get_model_map()` | `"medium"` |
| `get_model_info("medium")` | Uses `_map_model_name()` → `_get_model_map()` | `"medium"` |

**Result**: Both methods produce the same model name. `check-models` correctly identifies cached models and verification is consistent with actual benchmark usage.

The implementation follows the standardized model mapping pattern documented in `CLAUDE.md`:
- Single source of truth in `_get_model_map()`
- Both `load_model()` and `get_model_info()` use `_map_model_name()` helper
- No divergent mappings between verification and loading

---

## Key Files Analyzed

| File | Purpose |
|------|---------|
| `src/mac_whisper_speedtest/implementations/whisperkit.py` | Python wrapper implementation |
| `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift` | Swift bridge executable |
| `tools/whisperkit-bridge/Package.swift` | Swift package manifest (depends on WhisperKit 0.13.1+) |
| `.build/checkouts/WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift` | WhisperKit main class |
| `.build/checkouts/WhisperKit/Sources/WhisperKit/Core/Configurations.swift` | WhisperKitConfig class |
| `.build/checkouts/swift-transformers/Sources/Hub/HubApi.swift` | Swift HubApi for downloads |
| `src/mac_whisper_speedtest/check_models.py` | Model verification logic |
| `src/mac_whisper_speedtest/benchmark.py` | Benchmark runner |
