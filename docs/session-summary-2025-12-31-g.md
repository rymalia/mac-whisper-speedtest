# Session Summary - 2025-12-31 (Session G)

## Overview

Documented the execution flow for `WhisperCppCoreMLImplementation`, continuing the implementation documentation series. This implementation uses pywhispercpp (Python bindings for whisper.cpp) with optional CoreML encoder acceleration.

## Key Decisions Made

1. **Followed existing template**: Used established documentation structure from previous sessions
2. **Documented dual-file architecture**: GGML models (auto-downloaded) + CoreML encoders (manual download)
3. **No variant mismatches found**: Implementation correctly uses `_map_model_name()` in both `load_model()` and `get_model_info()`

## Files Created

| File | Description |
|------|-------------|
| `docs/model_details_WhisperCppCoreMLImplementation.md` | Complete execution trace for WhisperCpp+CoreML implementation including benchmark flow, check-models flow, cache locations, model mapping, and consistency analysis |

## Files Modified

| File | Description |
|------|-------------|
| `docs/IMPLEMENTATION_DOCUMENTATION_TEMPLATE.md` | Updated completion checklist to mark WhisperCppCoreMLImplementation as done |

## Key Findings

### WhisperCpp+CoreML Architecture

| Aspect | Value |
|--------|-------|
| **Backend** | whisper.cpp (C++) via pywhispercpp Python bindings |
| **Download Source** | Direct HTTP from HuggingFace (not HF Hub API) |
| **Model Format** | GGML quantized (`.bin`) + CoreML encoders (`.mlmodelc`) |
| **Cache Location** | `{project_root}/models/` (custom, not HF cache) |
| **Threading** | 4 threads |
| **CoreML Enable** | Environment variable `WHISPER_COREML=1` |

### Model Mapping (from implementation's `_get_model_map()`)

| Model Name | GGML Model Name | Quantization |
|------------|-----------------|--------------|
| `tiny` | `tiny-q5_1` | 5-bit |
| `base` | `base-q5_1` | 5-bit |
| `small` | `small` | None (full precision) |
| `medium` | `medium-q5_0` | 5-bit |
| `large` | `large-v3-turbo-q5_0` | 5-bit |

### CoreML Encoder Mapping (from `_get_coreml_model_map()`)

| Model Name | CoreML Encoder Name |
|------------|---------------------|
| `tiny` | `tiny` |
| `base` | `base` |
| `small` | `small` |
| `medium` | `medium` |
| `large` | `large-v3-turbo` |

### Download URL Pattern

```
https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin
```

Example for "medium":
```
https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin
```

### No Variant Mismatch

Implementation correctly uses standardized pattern:

| Component | Behavior | Result |
|-----------|----------|--------|
| `load_model("medium")` | Uses `_map_model_name()` → `_get_model_map()` | `medium-q5_0` |
| `get_model_info("medium")` | Uses `_map_model_name()` → `_get_model_map()` | `medium-q5_0` |

**Result**: Both methods produce the same GGML model name. `check-models` correctly identifies cached models.

### Notable Architecture Details

1. **Two file types required**:
   - GGML model (`.bin`) - auto-downloaded by pywhispercpp
   - CoreML encoder (`.mlmodelc`) - requires manual download

2. **Fallback behavior**: If CoreML encoder not found, falls back to CPU-only inference

3. **Unique quantization choice**: "small" model uses full precision while others use 5-bit quantization

## Key Files Analyzed

- `src/mac_whisper_speedtest/implementations/coreml.py` - Main implementation
- `.venv/lib/python3.12/site-packages/pywhispercpp/model.py` - Library's Model class
- `.venv/lib/python3.12/site-packages/pywhispercpp/utils.py` - Library's download_model function
- `.venv/lib/python3.12/site-packages/pywhispercpp/constants.py` - Available models and URL patterns
- `src/mac_whisper_speedtest/check_models.py` - Model verification
- `src/mac_whisper_speedtest/utils.py` - `get_models_dir()` helper

## Summary Statistics

- **Files created**: 1
- **Files modified**: 1
- **Known issues documented**: 0 (none found)
- **Implementations documented**: 7 of 9 total

## Documentation Progress

- [x] `LightningWhisperMLXImplementation` - docs/model_details_LightningWhisperMLXImplementation.md
- [x] `MLXWhisperImplementation` - docs/model_details_MLXWhisperImplementation.md
- [x] `ParakeetMLXImplementation` - docs/model_details_ParakeetMLXImplementation.md
- [x] `InsanelyFastWhisperImplementation` - docs/model_details_InsanelyFastWhisperImplementation.md
- [x] `WhisperMPSImplementation` - docs/model_details_WhisperMPSImplementation.md
- [x] `FasterWhisperImplementation` - docs/model_details_FasterWhisperImplementation.md
- [x] `WhisperCppCoreMLImplementation` - docs/model_details_WhisperCppCoreMLImplementation.md
- [ ] `WhisperKitImplementation`
- [ ] `FluidAudioCoreMLImplementation`

## Next Steps

Continue documenting remaining 2 Swift bridge implementations using `docs/IMPLEMENTATION_DOCUMENTATION_TEMPLATE.md`:
1. `WhisperKitImplementation` - Swift bridge (WhisperKit)
2. `FluidAudioCoreMLImplementation` - Swift bridge (FluidAudio)
