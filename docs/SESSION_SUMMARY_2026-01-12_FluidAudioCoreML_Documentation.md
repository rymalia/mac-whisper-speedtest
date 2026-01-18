# Session Summary: FluidAudioCoreMLImplementation Deep Dive

**Date:** 2026-01-12
**Duration:** ~1 hour
**Model:** Claude Opus 4.5

---

## Objective

Re-investigate and thoroughly document the `FluidAudioCoreMLImplementation` to answer specific questions about model downloading, caching behavior, CoreML configuration, and runtime behavior.

---

## Key Deliverable

**Created:** `docs/model_details_FluidAudioCoreMLImplementation.md`

A comprehensive ~600-line documentation file covering:
- Complete execution flow (Python → Swift bridge → FluidAudio library)
- 18 key questions answered with code evidence
- Empirical test results with actual terminal output
- Known issues with priority rankings (P0-P3)
- Recommended improvements with effort estimates

---

## Investigation Methodology

### Code Analysis
1. Traced execution from `test_benchmark2.py` through `fluidaudio_coreml.py` [PROJECT]
2. Analyzed Swift bridge at `tools/fluidaudio-bridge/Sources/main.swift` [BRIDGE]
3. Deep-dived into FluidAudio library: `AsrModels.swift`, `DownloadUtils.swift` [LIBRARY]

### Empirical Testing
1. **Fresh Download Test** - Started with empty cache, ran bridge directly
2. **Cached Performance Test** - Ran Python benchmark with pre-downloaded models
3. **Incomplete File Test** - Renamed required file to trigger recovery behavior

---

## Key Findings

### Model Download Behavior

| Finding | Detail |
|---------|--------|
| Download location | `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/` |
| Uses HF Hub cache? | **NO** - completely independent |
| Checks HF Hub cache? | **NO** - never looks there |
| Download method | Direct HTTP to `huggingface.co/*/resolve/main/*` (not HF Hub API) |
| Resume support | **NOT IMPLEMENTED** (code structure suggests it, but not used) |
| Per-file skip | **YES** - files with matching size are skipped |
| Total size | 2.4GB (13 files) |
| Download time | ~22 minutes |

### Model Name Handling

**HARDCODED at every level:**
- Python wrapper logs "ignores model_name parameter"
- Swift bridge calls `AsrModels.downloadAndLoad()` with no parameter
- Library uses `DownloadUtils.Repo.parakeet` enum (single option)

**Result:** `tiny`, `small`, `medium`, `large` all produce identical results.

### CoreML Configuration

```swift
config.computeUnits = isCI ? .cpuAndNeuralEngine : .all
config.allowLowPrecisionAccumulationOnGPU = true
```

- **Normal mode:** CPU + GPU + Neural Engine (full Apple Silicon)
- **CI mode:** CPU + Neural Engine only
- **SDPA:** Not relevant (PyTorch concept, not CoreML)
- **Quantization:** 4-bit variant exists but not used by default

### Critical Wipe-and-Restart Behavior

**Empirically confirmed:** Missing ONE required file triggers deletion of ALL 2.4GB:

```
Before: 2.4G (13 files)
After:  14M  (2 files, re-downloading from scratch)
```

Code location: `DownloadUtils.swift:46-50` [LIBRARY]

---

## Empirical Test Results

| Test | Command | Result |
|------|---------|--------|
| Fresh download | `./fluidaudio-bridge tests/jfk.wav` | 22 min, 2.4GB |
| Cached performance | `python test_benchmark2.py small 2` | 0.37s transcription |
| Missing file | Renamed RNNTJoint.mlmodelc | **COMPLETE WIPE** |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Internal transcription time | 0.37s |
| Real-time factor | ~30x |
| First-run overhead | ~1.0s (model loading) |
| Subsequent run overhead | ~0.3s |

---

## Issues Discovered

| Priority | Issue | Location | Effort |
|----------|-------|----------|--------|
| **P0** | Python timeout 300s < 22 min download | `fluidaudio_coreml.py:116` [PROJECT] | 1 line |
| **P1** | No pre-download warning | `fluidaudio_coreml.py` [PROJECT] | ~20 lines |
| **P2** | Resume not actually implemented | `DownloadUtils.swift:342` [LIBRARY] | ~50 lines |
| **P2** | Aggressive cache wipe on any failure | `DownloadUtils.swift:46-50` [LIBRARY] | ~100 lines |
| **P3** | Model size parameter ignored | Multiple locations | Large |

---

## File Classification (Per User Request)

| Tag | Path | Modifiable? |
|-----|------|-------------|
| **[PROJECT]** | `src/mac_whisper_speedtest/` | ✅ Yes |
| **[BRIDGE]** | `tools/fluidaudio-bridge/Sources/` | ✅ Yes |
| **[LIBRARY]** | `.build/checkouts/FluidAudio/` | ❌ No (external) |

---

## Documentation Checklist Status

- [x] Code analysis flow documented for both `small` and `large` model paths
- [x] Benchmark was ACTUALLY run (not just analyzed)
- [x] Terminal output from benchmark runs included
- [x] Model file locations verified with `ls` commands
- [x] "Empirical Test Results" section contains actual observed data
- [x] "Key Questions Answered" table included
- [x] "Recommended Improvements" section with proposals
- [x] Priority Summary table with effort estimates
- [x] Implementation Order Recommendation with phases

---

## Post-Session State

**Models Status:** WIPED during Test 3 (incomplete file handling test)

**To Restore:**
```bash
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json
# Takes ~22 minutes
```

---

## Questions Answered (Summary)

1. **Re-downloads every call?** NO - modelsExist() check prevents re-download
2. **Downloads to HF cache?** NO - uses Application Support folder
3. **Checks HF cache for models?** NO - never looks there
4. **Resume partial downloads?** NO - despite code structure
5. **Complete files skipped?** YES - file size verification
6. **Completeness checks?** YES - file size + coremldata.bin validation
7. **Uses HF Hub API?** NO - direct HTTP REST API
8. **Timeout sufficient?** NO - 300s Python timeout < 22 min download
9. **Fallback chains?** NO - single hardcoded model
10. **CPU or GPU?** BOTH + Neural Engine (.all compute units)
11. **Apple Silicon optimized?** YES - CoreML + ANE
12. **SDPA relevant?** NO - PyTorch concept
13. **Uses quantization?** 4-bit variant exists but not default

---

## Files Modified/Created

| File | Action |
|------|--------|
| `docs/model_details_FluidAudioCoreMLImplementation.md` | Created (~600 lines) |
| `docs/SESSION_SUMMARY_2026-01-12_FluidAudioCoreML_Documentation.md` | Created (this file) |

---

## Recommendations for Next Session

1. **Quick Win:** Apply P0 timeout fix (1 line change)
2. **Re-download models** if FluidAudio benchmarking needed
3. **Consider upstream PR** to FluidAudio for resume support and less aggressive wipe behavior
