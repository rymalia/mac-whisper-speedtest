# Session Summary: LightningWhisperMLXImplementation Documentation

**Date**: January 7, 2026
**Focus**: Tracing execution flow and model download behavior for `LightningWhisperMLXImplementation`

---

## Deliverables

### 1. Implementation Documentation
**Created**: `docs/model_details_LightningWhisperMLXImplementation.md`

Contents:
- 15-step execution flow from CLI to transcription
- Clear **[PROJECT]** vs **[LIBRARY]** file categorization
- Model mapping reference tables (library-level + project-level)
- 7 detailed notes with plain-English explanations
- Empirical test results with terminal output

### 2. CLAUDE.md Update
**Added**: "Model File Verification Guidelines" section

Key points captured:
- Never assume origin of model files in any folder
- Ask permission before deleting model files
- Rename trick using `__OFF__` pattern for safe testing
- User preference for common HuggingFace cache location

---

## Key Technical Findings

### Cache Behavior (Empirically Verified)

| Claim | Result |
|-------|--------|
| Initial assumption: "Dual Cache Problem" | **INCORRECT** |
| `lightning-whisper-mlx` duplicates to HF cache | **NO** — downloads only to `./mlx_models/` |
| Duplicate files in two locations | Caused by **different implementations**, not one library |

**Test Methodology**:
1. Deleted `./mlx_models/tiny/` and `~/.cache/huggingface/hub/models--mlx-community--whisper-tiny/`
2. Ran: `.venv/bin/python3 test_benchmark2.py tiny 1 LightningWhisperMLXImplementation`
3. Observed: Files only appeared in `./mlx_models/tiny/` (71MB)

### Cross-Implementation Cache Locations

| Implementation | Download Method | Cache Location |
|----------------|-----------------|----------------|
| `lightning-whisper-mlx` | `hf_hub_download(local_dir=...)` | `./mlx_models/{name}/` |
| `mlx-whisper` | `snapshot_download()` | `~/.cache/huggingface/hub/` |

**Implication**: Benchmarking both implementations downloads the same model twice to different locations.

### Model Mapping

| User Request | lightning-whisper-mlx Uses |
|--------------|---------------------------|
| `tiny` | `mlx-community/whisper-tiny` |
| `small` | `mlx-community/whisper-small-mlx` |
| `medium` | `mlx-community/whisper-medium-mlx` |
| `large` | `mlx-community/whisper-large-v3-mlx` (mapped via project code) |

**Note**: No `large-v3-turbo` support in this library.

---

## Lesson Learned

**Initial Incorrect Assumption**: Seeing identical model files in two locations led to the assumption that `lightning-whisper-mlx` was duplicating files.

**Actual Cause**: The files were downloaded on different dates by different implementations:
- `~/.cache/huggingface/hub/...whisper-small-mlx/` — Dec 24 (by `mlx-whisper`)
- `./mlx_models/small/` — Jan 6 (by `lightning-whisper-mlx`)

**Rule Established**: Always verify cache behavior empirically by:
1. Clearing/renaming expected folders (with permission)
2. Running the specific process
3. Observing which folders receive files

---

## Documentation Progress

| Implementation | Status | Documentation File |
|----------------|--------|-------------------|
| `LightningWhisperMLXImplementation` | **COMPLETE** | `model_details_LightningWhisperMLXImplementation.md` |
| `MLXWhisperImplementation` | Pending | — |
| `ParakeetMLXImplementation` | Pending | — |
| `InsanelyFastWhisperImplementation` | Pending | — |
| `WhisperMPSImplementation` | Pending | — |
| `FasterWhisperImplementation` | Pending | — |
| `WhisperCppCoreMLImplementation` | Pending | — |
| `WhisperKitImplementation` | Pending | — |
| `FluidAudioCoreMLImplementation` | Pending | — |

---

## Files Modified This Session

| File | Action |
|------|--------|
| `docs/model_details_LightningWhisperMLXImplementation.md` | Created (338 lines) |
| `CLAUDE.md` | Added "Model File Verification Guidelines" section |
| `docs/SESSION_SUMMARY_2026-01-07_LightningWhisperMLX.md` | Created (this file) |
