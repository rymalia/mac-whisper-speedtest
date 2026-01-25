# Session Summary: CLI Batch Mode Implementation

**Date:** 2026-01-24
**Duration:** Extended session
**Primary Focus:** Implementing CLI batch mode and prioritization planning

---

## Revised Prioritization (READ THIS FIRST)

| Priority | Task | Status | Rationale |
|----------|------|--------|-----------|
| 1 | CLI Batch Mode | ✅ **DONE** | Enables non-interactive testing for all future work |
| **2** | **CoreML Improvements** | 🔲 Ready | pywhispercpp is upgraded and working but needs a proper commit, delivers 2-3x speedup, independent of upgrades |
| 3 | MLX 0.27→0.30 | 🔲 Planned | Foundation for 4 implementations |
| 4 | WhisperKit 0.13→0.15 | 🔲 Planned | Medium risk, no code changes expected |
| 5 | FluidAudio 0.1→0.10 | 🔲 Planned | Highest risk, requires code changes |

**Note:** The CoreML Improvements plan (`docs/feature_plans_WhisperCppCoreMLImplementation_CoreML_Improvements.md`) needs updating — it references `test_benchmark2.py` which was deleted. Replace with `--batch` mode commands.

---

## Accomplishments

### 1. CLI Batch Mode Feature (Complete)

Implemented `--batch` and `--audio` flags to enable file-based benchmarking without microphone recording.

**New CLI Options:**
```bash
.venv/bin/mac-whisper-speedtest --batch                   # Uses tests/jfk.wav
.venv/bin/mac-whisper-speedtest --batch --audio tests/ted_60.wav
.venv/bin/mac-whisper-speedtest -b -m large -n 1 -i "MLXWhisperImplementation"
```

**Short flags added:** `-m` (model), `-n` (runs), `-i` (implementations), `-b` (batch), `-a` (audio)

**Auto-enable feature:** If `--audio` is provided with a non-default value, batch mode enables automatically.

### 2. Venv Stale Path Fix

**Problem:** After renaming folder from `mac-whisper-speedtest_MAIN` to `mac-whisper-speedtest`, 71 scripts in `.venv/bin/` had hardcoded shebangs pointing to the old path.

**Solution:** `rm -rf .venv && uv sync` — recreated the venv with correct paths.

### 3. pywhispercpp Build Issue

**Problem:** `uv sync` failed to build pywhispercpp from git due to CMake not finding Python in uv's isolated build environment.

**Workaround:** User installed pywhispercpp manually using the CoreML build guide (`docs/optimizations_2026-01-13_pywhispercpp_CoreML_Build_Guide.md`).

**Current state:** pywhispercpp is re-enabled in pyproject.toml and working.

---

## Files Modified

| File | Change |
|------|--------|
| `src/mac_whisper_speedtest/cli.py` | Added batch mode, audio loading, short flags, constants block |
| `CLAUDE.md` | Updated usage docs with batch mode |
| `README.md` | Added batch mode section, CLI options table, fixed "8"→"9" implementations |
| `docs/feature_plan_CLI_Batch_Mode.md` | Marked as ✅ Implemented |
| `pyproject.toml` | Temporarily disabled then re-enabled pywhispercpp |
| `uv.lock` | Regenerated (venv rebuild) |

## Files Deleted

| File | Reason |
|------|--------|
| `test_benchmark.py` | Replaced by `--batch` mode |
| `test_benchmark2.py` | Replaced by `--batch` mode |

---

## Pending Commit

The CLI batch mode changes are ready to commit. Proposed commit message:

```
feat(cli): add batch mode for non-interactive benchmarking

Add --batch/-b and --audio/-a flags to enable file-based benchmarking
without microphone recording. This consolidates three entry points into
one unified CLI.

New features:
- --batch/-b: Enable non-interactive mode using audio file
- --audio/-a: Specify audio file path (default: tests/jfk.wav)
- Auto-enable batch mode when --audio provided with non-default value
- Add short flags: -m (model), -n (runs), -i (implementations)
- Centralized defaults as constants at top of cli.py

Removed:
- test_benchmark.py (replaced by --batch)
- test_benchmark2.py (replaced by --batch)

Documentation:
- Updated CLAUDE.md with new CLI usage
- Updated README.md with batch mode section and CLI options table
- Marked feature_plan_CLI_Batch_Mode.md as implemented

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Key Discoveries

### 1. Missing Feature Plan in Original Prioritization

The `feature_plans_WhisperCppCoreMLImplementation_CoreML_Improvements.md` was not included in the initial prioritization review. This plan:
- Adds auto-download of CoreML models from HuggingFace
- Adds runtime CoreML availability detection
- Is independent of Swift bridge upgrades
- Provides 2-3x performance improvement

### 2. Feature Plan Staleness

The CoreML Improvements plan references `test_benchmark2.py` (deleted). Before implementing, update lines 97, 100, 357, 373, 384, 385 to use `--batch` mode instead.

### 3. uv Build Isolation Issue

When building packages from git that use CMake (like pywhispercpp), uv's isolated build environment can cause CMake to fail finding Python. Workaround: use `pip install --no-build-isolation` or build manually.

---

## Next Steps

1. **Commit the CLI batch mode changes** (staged, ready to go)
2. **Update CoreML Improvements plan** — replace `test_benchmark2.py` references with `--batch` commands
3. **Implement CoreML Improvements** (Priority #2)
4. **Continue with MLX upgrade** (Priority #3)

---

## Documents Referenced

| Document | Purpose |
|----------|---------|
| `docs/feature_plan_CLI_Batch_Mode.md` | Implementation guide (now marked complete) |
| `docs/feature_plans_WhisperCppCoreMLImplementation_CoreML_Improvements.md` | Next priority item |
| `docs/upgrade_plan_MLX_0.27_to_0.30.md` | MLX upgrade strategy |
| `docs/upgrade_plan_WhisperKit_0.13_to_0.15.md` | WhisperKit upgrade strategy |
| `docs/upgrade_plan_FluidAudio_0.1_to_0.10.md` | FluidAudio upgrade strategy |
| `docs/optimizations_2026-01-13_pywhispercpp_CoreML_Build_Guide.md` | pywhispercpp build with CoreML |

---

## Session Statistics

- **Context used:** ~143k/200k tokens (72%)
- **Files modified:** 6
- **Files deleted:** 2
- **Feature implemented:** CLI Batch Mode (complete)
- **Prioritization updated:** Added CoreML Improvements at #2
