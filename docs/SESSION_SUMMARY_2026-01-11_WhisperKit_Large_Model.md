# Session Summary: WhisperKit Large Model Deep Dive

**Date**: January 11, 2026
**Duration**: ~2.5 hours (including ~1 hour of model download time)
**Model**: Claude Opus 4.5

---

## Objectives

1. Deep dive into `WhisperKitImplementation` with **large model** (not small)
2. Answer specific questions about download behavior, caching, timeouts, and completeness checks
3. Empirically verify all findings (not just code analysis)

---

## Key Questions Investigated

| Question | Answer | How Verified |
|----------|--------|--------------|
| Does Swift bridge re-download every call? | **NO** - uses file-exists caching | Code + empirical |
| Where are temp files during download? | `/private/var/folders/*/T/CFNetworkDownload_*.tmp` | Empirical |
| Download folder = runtime folder? | **YES** - same location | Code + empirical |
| Is 300s timeout sufficient for large? | **NO** - needs ~30+ minutes | Empirical (timeout hit) |
| Does partial download resume? | **NO** - creates new temp files | Empirical |
| Completeness check on existing files? | **NONE** - only `fileExists()` | Code + empirical |
| What happens with incomplete cache? | **SILENT FAILURE** - skipped | Empirical |

---

## Empirical Tests Performed

### Test 1: Fresh Download with 300s Timeout
- **Result**: TIMEOUT after exactly 300 seconds
- **Downloaded**: 611MB in 5 minutes (~2 MB/s)
- **Partial cache created**: MelSpectrogram complete, TextDecoder 4K (incomplete)

### Test 2: Retry with Partial Cache
- **Result**: TIMEOUT after 300 seconds
- **Critical finding**: 4K TextDecoder folder was **SKIPPED** (treated as complete!)
- **New orphaned temp file**: 477MB

### Test 3: Direct Swift Bridge (No Python Timeout)
- **First attempt**: 23:50 - CDN timeout during TextDecoder download
- **Second attempt**: 27:31 - SUCCESS, completed remaining files
- **Total download time**: ~51 minutes

### Test 4: Cached Model Benchmark
- **Result**: SUCCESS
- **Total time**: 6.97 seconds
- **Internal transcription**: 3.05 seconds
- **No re-download** - cache correctly used

---

## Critical Bugs Discovered

### P0: Timeout Insufficient for Large Model
- **Location**: `whisperkit.py:128`
- **Current**: 300 seconds (5 minutes)
- **Required**: ~1800 seconds (30 minutes) for reliable download
- **Impact**: Large model download **always fails** on first run

### P0: No Completeness Check
- **Location**: `swift-transformers/Sources/Hub/HubApi.swift:204-206`
- **Bug**: Only checks `FileManager.default.fileExists()`, not size/checksum
- **Impact**: Incomplete files cause **permanent failure state**

### P1: No Download Resume
- **Location**: `swift-transformers/Sources/Hub/Downloader.swift`
- **Bug**: Each attempt creates new temp file instead of resuming
- **Impact**: Bandwidth waste, orphaned temp files (~1.8GB observed)

---

## Model Sizes Discovered

| Component | Size |
|-----------|------|
| AudioEncoder.mlmodelc | 1.2GB |
| TextDecoder.mlmodelc | 1.7GB |
| MelSpectrogram.mlmodelc | 392K |
| Tokenizer (models/) | 2.6MB |
| **Total large-v3** | **2.9GB** |

---

## Performance Comparison

| Model | Total (cached) | Internal Transcription | Overhead | Model Size |
|-------|----------------|------------------------|----------|------------|
| small | 1.37s | 0.44s | ~0.93s | 487MB |
| large-v3 | 6.97s | 3.05s | ~3.9s | 2.9GB |

---

## Files Created/Modified

| File | Action |
|------|--------|
| `docs/model_details_WhisperKitImplementation_large.md` | **Created** - comprehensive documentation |
| `docs/SESSION_SUMMARY_2026-01-11_WhisperKit_Large_Model.md` | **Created** - this summary |

---

## Code Locations Referenced

### Project Files
- `src/mac_whisper_speedtest/implementations/whisperkit.py:128` - timeout setting
- `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift` - Swift bridge

### Library Files (Read-Only)
- `swift-transformers/Sources/Hub/HubApi.swift:204-206` - file-exists check (bug)
- `swift-transformers/Sources/Hub/Downloader.swift` - URLSession download
- `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:298-330` - model setup

---

## Recommended Immediate Fixes

1. **Increase timeout** in `whisperkit.py:128`:
   ```python
   timeout=1800  # 30 minute timeout for large model download
   ```

2. **Add documentation warning** about first-run download time for large models

3. **Cleanup script** for orphaned temp files:
   ```bash
   rm -f /private/var/folders/*/T/CFNetworkDownload_*.tmp
   ```

---

## Workarounds for Users

### Manual Large Model Download
```bash
# Run bridge directly (will complete download, ~30 min)
tools/whisperkit-bridge/.build/release/whisperkit-bridge tests/jfk.wav --model large-v3
```

### Clean Corrupt Cache
```bash
rm -rf ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/
```

### Use Turbo Model Instead
```bash
# Smaller (~500MB), may complete within timeout
.venv/bin/python3 test_benchmark2.py large-v3-turbo 1 WhisperKitImplementation
```

---

## Session Insights

1. **Empirical testing is essential** - Code analysis alone would have missed the CDN timeout issue and the exact download times

2. **Large model behavior differs significantly from small** - 300s timeout works for 487MB small model but fails for 2.9GB large model

3. **The Hub library's caching is fundamentally broken** for partial downloads - a 4K folder was treated as a complete 1.7GB model

4. **Subprocess overhead scales with model size** - 0.93s for small vs 3.9s for large (CoreML model loading time)

5. **Network conditions matter** - Even without our timeout, the HuggingFace CDN itself timed out after 24 minutes on first attempt

---

## Template Improvements

Based on findings from this session, updated `docs/IMPLEMENTATION_DOCUMENTATION_TEMPLATE.md` to ensure future Claude agents test both model sizes:

### Changes Made

| Section | Change |
|---------|--------|
| Section 2 intro | Added table explaining WHY test both sizes |
| Section 2 intro | Added "CRITICAL LESSON LEARNED" callout with WhisperKit example |
| Test Procedure | Split into **Phase A/B/C** (small, large, cached) |
| Rename examples | Fixed `large` → `lar__OFF__ge` (was incorrectly `ti__OFF__ny`) |
| New section | Added "What To Do When Large Model Times Out" with step-by-step recovery |
| Completion Criteria | Added explicit "BOTH sizes" requirements |
| Output Structure | Added subsection requirements for both sizes |
| Example Output | Added `WhisperKitImplementation_large.md` as gold standard for large model testing |

### Key Rationale

> **Without this change**: Future agents would only test `small` model (quick, always works) and miss timeout bugs that only manifest with `large` models.

Template grew from ~193 to ~239 lines. The added clarity should significantly improve agent compliance with thorough testing requirements.

---

## Files Created/Modified (Final)

| File | Action |
|------|--------|
| `docs/model_details_WhisperKitImplementation_large.md` | **Created** - comprehensive large model documentation |
| `docs/SESSION_SUMMARY_2026-01-11_WhisperKit_Large_Model.md` | **Created** - this summary |
| `docs/IMPLEMENTATION_DOCUMENTATION_TEMPLATE.md` | **Modified** - added dual-size testing requirements |

---

## Next Steps (Not Done This Session)

- [ ] Fix timeout in `whisperkit.py` (1 line change: 300 → 1800)
- [ ] Consider PR to swift-transformers for completeness checking
- [ ] Test `large-v3-turbo` model behavior
- [ ] Document other implementations with same rigor (using updated template)
