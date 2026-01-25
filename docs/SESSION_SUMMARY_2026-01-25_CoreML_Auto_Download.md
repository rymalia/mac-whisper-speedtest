# Session Summary: CoreML Auto-Download Implementation

**Date:** 2026-01-25
**Primary Focus:** CoreML model auto-download and installation documentation updates

---

## Accomplishments

### 1. Installation Documentation Updates (Ready to Commit)

Updated three files to document the optional CoreML build for pywhispercpp:

| File | Change |
|------|--------|
| `pyproject.toml` | Pinned pywhispercpp to `v1.4.1` |
| `README.md` | Added "Optional: CoreML Acceleration" section with build commands and performance stats |
| `CLAUDE.md` | Added CoreML build instructions to Setup section |

**Key insight:** `uv sync` cannot pass environment variables to CMake, so CoreML build requires a separate manual step after initial install.

### 2. CoreML Auto-Download Feature (Ready to Commit)

Implemented automatic downloading of CoreML encoder models from HuggingFace:

**File:** `src/mac_whisper_speedtest/implementations/coreml.py`

New methods:
- `_has_coreml_support()` - Parses `Model.system_info()` to detect if pywhispercpp has CoreML compiled in
- `_download_coreml_model()` - Downloads `.mlmodelc.zip` from `huggingface.co/ggerganov/whisper.cpp`, extracts, validates

**Flow:**
1. `load_model()` checks if `.mlmodelc` exists
2. If missing, checks `_has_coreml_support()`
3. If CoreML supported, downloads and extracts model
4. Falls back gracefully to Metal GPU if download fails

### 3. Runtime CoreML Detection (Ready to Commit)

**File:** `src/mac_whisper_speedtest/cli.py`

Added `_check_coreml_availability()` function that:
- Runs on macOS only
- Checks `Model.system_info()` for `COREML = 0`
- Logs warning with link to build guide if CoreML not enabled
- Called at CLI startup in `main()`

### 4. Feature Plan Document Update

**File:** `docs/feature_plans_WhisperCppCoreMLImplementation_CoreML_Improvements.md`

Replaced 6 references to deleted `test_benchmark2.py` with `--batch` mode commands:
- `.venv/bin/python3 test_benchmark2.py small 3` → `.venv/bin/mac-whisper-speedtest -b -m small -n 3`

---

## Files Modified (Unstaged)

| File | Lines Changed | Description |
|------|---------------|-------------|
| `pyproject.toml` | +1 | Pin pywhispercpp to v1.4.1 |
| `README.md` | +20 | CoreML acceleration section |
| `CLAUDE.md` | +12 | CoreML build instructions |
| `src/mac_whisper_speedtest/implementations/coreml.py` | +102 | Auto-download, CoreML detection |
| `src/mac_whisper_speedtest/cli.py` | +30 | Runtime CoreML check |
| `docs/feature_plans_...CoreML_Improvements.md` | 6 lines | test_benchmark2.py → --batch |

---

## Pending Commits

### Commit 1: Installation Documentation
```bash
git add pyproject.toml README.md CLAUDE.md
```

**Suggested message:**
```
docs: add CoreML build instructions for pywhispercpp

Document the optional CoreML acceleration build for WhisperCppCoreML
implementation, providing 2-3x speedup on Apple Silicon.

Changes:
- Pin pywhispercpp to v1.4.1 in pyproject.toml
- Add "Optional: CoreML Acceleration" section to README.md
- Add CoreML build instructions to CLAUDE.md Setup section

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Commit 2: CoreML Auto-Download Feature
```bash
git add src/mac_whisper_speedtest/implementations/coreml.py \
        src/mac_whisper_speedtest/cli.py \
        docs/feature_plans_WhisperCppCoreMLImplementation_CoreML_Improvements.md
```

**Suggested message:**
```
feat: add CoreML model auto-download and runtime detection

Add automatic downloading of CoreML encoder models from HuggingFace
when missing, enabling 2-3x performance improvement without manual
model setup.

Changes:
- Add _download_coreml_model() to fetch and extract .mlmodelc.zip
  from huggingface.co/ggerganov/whisper.cpp
- Add _has_coreml_support() to detect pywhispercpp CoreML capability
- Integrate auto-download into load_model() with graceful fallback
- Add _check_coreml_availability() in CLI to warn on startup if
  CoreML not enabled
- Update feature plan to use --batch mode (test_benchmark2.py deleted)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Verification (Not Yet Performed)

```bash
# Test auto-download
mv models/ggml-small-encoder.mlmodelc models/ggml-small-encoder.mlmodelc.bak
.venv/bin/mac-whisper-speedtest -b -m small -n 1 -i "WhisperCppCoreMLImplementation"

# Restore
mv models/ggml-small-encoder.mlmodelc.bak models/ggml-small-encoder.mlmodelc
```

---

## Priority Queue Status

| Priority | Task | Status |
|----------|------|--------|
| 1 | CLI Batch Mode | ✅ Done (previous session) |
| 2 | CoreML Improvements | ✅ **Code complete, pending commit** |
| 3 | MLX 0.27→0.30 | 🔲 Next |
| 4 | WhisperKit 0.13→0.15 | 🔲 Planned |
| 5 | FluidAudio 0.1→0.10 | 🔲 Planned |

---

## Next Steps

1. **Test auto-download** by temporarily moving existing `.mlmodelc` and running benchmark
2. **Commit changes** (two commits as outlined above)
3. **Proceed to MLX upgrade** (Priority #3)

---

## Session Statistics

- **Context used:** ~94k/200k tokens (47%)
- **Files modified:** 6
- **Lines added:** ~165
- **Features implemented:** CoreML auto-download, runtime detection
