# MODEL_CACHING.md Analysis Report

**Date**: 2025-12-31
**Analyst**: Claude Opus 4.5
**Purpose**: Compare MODEL_CACHING.md against authoritative model_details files and provide revision recommendations

---

## Executive Summary

MODEL_CACHING.md contains significant inaccuracies regarding cache locations for 4 of 9 implementations, while its troubleshooting sections (FluidAudio, WhisperKit) remain accurate and valuable. Recommendation: Restructure the document to fix errors, add a quick reference table, and point to model_details files as the authoritative source.

---

## Methodology

1. **Document Review**: Read all 9 `docs/model_details_*.md` files and `docs/MODEL_CACHING.md`
2. **Comparison**: Systematically compared cache locations, model mappings, and verification methods
3. **Empirical Testing**: Verified FluidAudio download issue and fix_models.sh script functionality
4. **Root Cause Analysis**: Investigated why model_details missed the FluidAudio download issue

---

## Findings

### Critical Discrepancy #1: Cache Location Misclassification

MODEL_CACHING.md claims these implementations use default HF cache (`~/.cache/huggingface/hub/`):

| Implementation | MODEL_CACHING.md Says | Actual Location | Status |
|---------------|----------------------|-----------------|--------|
| **MLXWhisper** | Default HF cache | `{project_root}/models/` (custom) | WRONG |
| **FasterWhisper** | Default HF cache | `{project_root}/models/` (custom) | WRONG |
| **InsanelyFastWhisper** | Default HF cache | `~/.cache/huggingface/hub/` | Correct |
| **LightningWhisperMLX** | Default HF cache | `./mlx_models/{model}/` + HF cache | MISLEADING |
| **ParakeetMLX** | Default HF cache | `{project_root}/models/hub/` (via HF_HOME) | WRONG |

**Impact**: Users searching for cached models will look in wrong locations.

### Critical Discrepancy #2: Missing Known Issues

The model_details files document several issues that MODEL_CACHING.md omits:

#### LightningWhisperMLX Variant Mismatch
- `get_model_info()` returns 8-bit quantized repos
- `load_model()` downloads base (non-quantized) versions
- **Result**: `check-models` may report "missing" for downloaded models

#### WhisperMPS Dual Download Issue
- `load_model()` downloads to `{project_root}/models/`
- `transcribe()` will re-download to `~/.cache/whisper/` via library's `ModelHolder` singleton (empirically confirmed 2025-12-31)
- **Result**: Same model stored twice, wasting disk space
- **Note**: Despite library name "whisper-mps", uses MLX not MPS

#### WhisperMPS Large Model Filename Mismatch (Empirically Confirmed 2025-12-31)
- `load_model("large")` downloads `large-v3.pt`
- `get_model_info("large")` looks for `large.pt`
- **Result**: `check-models` reports "missing" for working models
- **Verified**: `test_benchmark.py large` succeeds while `check-models --model large` reports missing

#### ParakeetMLX Cache Location Verification Gap
- Downloads to `{project_root}/models/hub/` via HF_HOME override
- `get_model_info()` doesn't specify `hf_cache_dir`
- **Result**: May report "missing" for cached models

### Discrepancy #3: Outdated Verification Tables

The "Verification Details for Small Model" table contains outdated repo IDs that don't reflect:
- Current model mappings in implementations
- Known variant mismatches
- Fallback chain behavior

### Verified Accurate Sections

| Section | Verification Method | Status |
|---------|---------------------|--------|
| FluidAudio two-stage download issue | Empirical testing | ACCURATE |
| FluidAudio file structure (sizes) | File inspection | ACCURATE |
| fix_models.sh script | Execution test | WORKS |
| WhisperKit cache location | Code review | ACCURATE |
| WhisperKit incomplete download issue | Documentation | ACCURATE |
| WhisperMPS download source (Azure CDN) | Code review | ACCURATE |
| CoreML compilation explanation | N/A | USEFUL |

---

## FluidAudio Deep Dive

### Empirical Test Performed

1. **Disabled cache**: Renamed Application Support directories with "_OFF_" suffix
2. **Ran bridge**: Hung at "Starting model load..." for >30 seconds
3. **Ran fix_models.sh**: Successfully copied from HF cache to Application Support
4. **Ran bridge again**: Completed successfully (0.073s transcription)

### File Structure Verification

| File | MODEL_CACHING.md | Actual | Match |
|------|------------------|--------|-------|
| Encoder.mlmodelc | 425 MB | 425M | YES |
| Decoder.mlmodelc | 23 MB | 23M | YES |
| Preprocessor.mlmodelc | 520 KB | 520K | YES |
| JointDecision.mlmodelc | 12 MB | 12M | YES |
| parakeet_vocab.json | listed | 148K | YES |
| config.json | listed | 2 bytes (`{}`) | YES |

### Why model_details Missed This Issue

The model_details document traced the Swift code flow (DownloadUtils.swift), which shows:
- Direct download to Application Support via URLSession
- No intermediate HuggingFace cache step

However, **empirical testing** revealed:
- The Swift download can hang indefinitely
- Models may exist in HF cache from Python operations
- The fix_models.sh workaround is still needed

**Lesson**: Code tracing documents *intended behavior*; empirical testing documents *actual behavior*.

---

## Code Tracing vs. Empirical Testing

This analysis revealed an important documentation principle:

### Code Tracing (model_details approach)
- Documents what source code is **designed** to do
- Follows execution paths through functions
- Shows intended architecture and flow
- **Limitation**: May miss runtime failures, edge cases, race conditions

### Empirical Testing (MODEL_CACHING.md approach for troubleshooting)
- Documents what **actually happens** in practice
- Tests with clean/corrupted caches
- Discovers workarounds for real-world issues
- **Limitation**: May not explain why issues occur

### Recommendation
Both approaches are valuable. model_details files should include "Known Issues" sections based on empirical testing. MODEL_CACHING.md should focus on practical troubleshooting while pointing to model_details for authoritative technical details.

---

## Recommendations

### Option A: Minimal Fixes
- Fix the 4 incorrect cache locations
- Add missing known issues
- Keep current structure

**Pros**: Least work
**Cons**: Document remains 555+ lines and hard to maintain

### Option B: Replace with Simple Reference
- Create ~50-100 line quick reference
- Point to model_details for all details

**Pros**: Easy to maintain
**Cons**: Loses valuable troubleshooting content

### Option C: Hybrid Restructure (RECOMMENDED)
- Add Quick Reference table at top
- Point to model_details as authoritative source
- Keep troubleshooting sections (FluidAudio, WhisperKit)
- Remove redundant verification tables
- Add documentation philosophy note
- Target ~200-250 lines

**Pros**: Best of both worlds
**Cons**: Requires careful restructuring

---

## Proposed New Structure for MODEL_CACHING.md

```
# Model Caching Guide

## Quick Reference: Where Are My Models?
[Accurate lookup table - first thing novices need]

## Authoritative Documentation
[Point to model_details files]

## Troubleshooting

### FluidAudio Download Issue
[Keep - verified accurate and useful]

### WhisperKit Incomplete Downloads
[Keep - verified accurate]

### WhisperMPS Dual Download
[Add - documented in model_details but useful here]

## Background

### Cache Mechanisms Overview
[Simplified from current content]

### Code Tracing vs. Empirical Behavior
[New section explaining documentation philosophy]

## Appendix: Model File Locations
[Condensed from current detailed sections]
```

---

## Implementation Plan

### Phase 1: Immediate (Already Done)
- [x] Updated `model_details_FluidAudioCoreMLImplementation.md` with Known Issues section

### Phase 2: MODEL_CACHING.md Revision
- [ ] Add Quick Reference table with correct cache locations
- [ ] Add pointer to model_details files as authoritative
- [ ] Fix incorrect cache location claims
- [ ] Remove outdated verification tables
- [ ] Add "Code Tracing vs. Empirical Behavior" section
- [ ] Keep FluidAudio and WhisperKit troubleshooting sections

### Phase 3: Cleanup
- [ ] Remove redundant content
- [ ] Consolidate to ~200-250 lines
- [ ] Final review for consistency

---

## Correct Cache Locations Reference

For use in the Quick Reference table:

| Implementation | Primary Cache Location | Notes |
|---------------|------------------------|-------|
| FasterWhisperImplementation | `{project}/models/models--Systran--faster-whisper-*` | Custom HF cache via download_root |
| MLXWhisperImplementation | `{project}/models/models--mlx-community--whisper-*` | Custom HF cache via HF_HOME |
| InsanelyFastWhisperImplementation | `~/.cache/huggingface/hub/models--openai--whisper-*` | Default HF cache |
| LightningWhisperMLXImplementation | `./mlx_models/{model}/` | Project-relative, NOT HF cache |
| ParakeetMLXImplementation | `{project}/models/hub/models--nvidia--parakeet-*` | Custom HF cache via HF_HOME |
| WhisperMPSImplementation | `{project}/models/*.pt` | Direct download from Azure CDN; uses MLX not MPS |
| WhisperCppCoreMLImplementation | `{project}/models/ggml-*.bin` | Direct HTTP download |
| WhisperKitImplementation | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/` | Swift HubApi default |
| FluidAudioCoreMLImplementation | `~/Library/Application Support/FluidAudio/Models/` | Swift framework default |

---

## Files Modified This Session

1. **docs/model_details_FluidAudioCoreMLImplementation.md**
   - Added "Known Issues" section
   - Documented download reliability issue
   - Added workaround using fix_models.sh
   - Added verification steps
   - Added documentation philosophy note

2. **docs/MODEL_CACHING_ANALYSIS_2025-12-31.md** (this file)
   - Complete analysis report
   - Findings and recommendations
   - Implementation plan

---

## Next Steps

1. Review this analysis document
2. Decide on revision approach (Option A, B, or C)
3. Proceed with MODEL_CACHING.md revision
4. Consider adding Known Issues sections to other model_details files where applicable
