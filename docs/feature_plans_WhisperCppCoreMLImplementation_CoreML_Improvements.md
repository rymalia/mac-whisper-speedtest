# Implementation Plan: CoreML Model Auto-Download and Project Updates

## Overview

Implement automatic CoreML model downloading for `WhisperCppCoreMLImplementation`, run comparative benchmarks, update installation process, documentation, and prepare for contribution.

**Key Finding**: CoreML `.mlmodelc` models are available as pre-compiled `.zip` files at:
- Repository: `https://huggingface.co/ggerganov/whisper.cpp/tree/main`
- Format: Zip archives containing `.mlmodelc` bundles
- Strategy: Download → Extract → Use

---

## Implementation Breakdown

### 1. Add CoreML Model Auto-Download to `coreml.py`

**File**: `src/mac_whisper_speedtest/implementations/coreml.py`

**Current Behavior** (Lines 65-70):
- Checks if `.mlmodelc` file exists
- Logs warning if missing
- Disables CoreML and falls back to Metal GPU

**New Behavior**:
- Check if `.mlmodelc` exists
- If missing AND CoreML is enabled (detected via system_info), attempt download
- Download `.zip` from HuggingFace
- Extract to `models/` directory
- Validate extraction succeeded
- Use CoreML model if available, fallback gracefully if download/extraction fails

**Implementation Details**:

```python
def _download_coreml_model(self, model_name: str) -> Path | None:
    """Download and extract CoreML model from HuggingFace.

    Args:
        model_name: Model name (tiny, base, small, medium, large-v3-turbo)

    Returns:
        Path to extracted .mlmodelc directory, or None if download failed
    """
    # 1. Construct download URL
    base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    zip_filename = f"ggml-{model_name}-encoder.mlmodelc.zip"
    download_url = f"{base_url}/{zip_filename}"

    # 2. Download with progress bar (use requests + tqdm like pywhispercpp does)
    # 3. Extract zip to models directory
    # 4. Validate .mlmodelc/coremldata.bin exists
    # 5. Clean up zip file
    # 6. Return path to extracted .mlmodelc directory
```

**Key Design Decisions**:
- Only download if `pywhispercpp` has `COREML = 1` in system_info
- Use same download pattern as existing implementations (requests + tqdm)
- Store in centralized `models/` directory via `get_models_dir()`
- Fail gracefully with informative logs if download fails
- Don't re-download if already present (check existence first)

**Model Name Mapping**:
Need to map internal model names to HuggingFace zip filenames:
- `tiny-q5_1` → `ggml-tiny-encoder.mlmodelc.zip`
- `base-q5_1` → `ggml-base-encoder.mlmodelc.zip`
- `small` → `ggml-small-encoder.mlmodelc.zip`
- `medium-q5_0` → `ggml-medium-encoder.mlmodelc.zip`
- `large-v3-turbo-q5_0` → `ggml-large-v3-turbo-encoder.mlmodelc.zip`

---

### 2. Run Comparative Benchmarks

**Goal**: Validate CoreML performance against all other implementations

**Test Configurations**:
- Model sizes: `small` and `large` (as agreed)
- Runs per implementation: 3 (for statistical consistency)
- Test audio: `tests/jfk.wav` (11 seconds, already validated)

**Implementations to Compare**:
1. WhisperCppCoreMLImplementation (with CoreML) ← NEW
2. WhisperCppCoreMLImplementation (without CoreML, fallback) ← Baseline
3. WhisperKitImplementation
4. MLXWhisperImplementation
5. FasterWhisperImplementation
6. LightningWhisperMLXImplementation
7. InsanelyFastWhisperImplementation
8. WhisperMPSImplementation
9. ParakeetMLXImplementation

**Command**:
```bash
# Small model comparison (all implementations)
.venv/bin/python3 test_benchmark2.py small 3

# Large model comparison (all implementations)
.venv/bin/python3 test_benchmark2.py large 3
```

**Expected Results to Document**:
- Average time per implementation
- Speedup vs baseline (Metal GPU fallback)
- Ranking of implementations by speed
- Transcription quality comparison (text output)

**Output Location**: Results will be written to a new file:
- `docs/BENCHMARK_RESULTS_2026-01-13_CoreML_Comparison.md`

---

### 3. Update Installation Process

**Goal**: Make CoreML-enabled build conditional and well-documented

#### A. Update `pyproject.toml`

**File**: `pyproject.toml`

**Change 1**: Pin pywhispercpp to v1.4.1
```toml
# Current:
pywhispercpp = { git = "https://github.com/absadiki/pywhispercpp" }

# Updated:
pywhispercpp = { git = "https://github.com/absadiki/pywhispercpp", tag = "v1.4.1" }
```

**Rationale**: Version pinning for stability and reproducibility

#### B. Add Runtime Detection to `cli.py`

**File**: `src/mac_whisper_speedtest/cli.py`

**New Function**:
```python
def _check_coreml_availability() -> None:
    """Check if pywhispercpp has CoreML support and warn if not.

    Only runs on macOS. Prints helpful message if CoreML not enabled.
    """
    import platform
    if platform.system() != "Darwin":
        return

    try:
        from pywhispercpp.model import Model
        info = Model.system_info()

        if "COREML = 0" in info:
            logger.warning(
                "pywhispercpp built without CoreML support. "
                "For 2-3x speedup, rebuild with CoreML flags. "
                "See docs/SESSION_SUMMARY_2026-01-13_pywhispercpp_CoreML_Build_Guide.md"
            )
    except ImportError:
        pass  # pywhispercpp not installed
```

**Integration Point**: Call in `main()` function before running benchmarks

**Rationale**:
- User requested "conditional (detect and recommend)" approach
- Non-intrusive: just logs a warning
- Provides actionable guidance with link to comprehensive guide

#### C. Update `CLAUDE.md`

**File**: `CLAUDE.md`

**New Section** (after line 15):
```markdown
## Optional: CoreML Acceleration for WhisperCpp

For 2-3x speedup with `WhisperCppCoreMLImplementation`, rebuild pywhispercpp with CoreML:

\`\`\`bash
# Uninstall current version
uv pip uninstall pywhispercpp
uv cache clean pywhispercpp

# Reinstall with CoreML support
WHISPER_COREML=ON WHISPER_COREML_ALLOW_FALLBACK=ON \\
  uv pip install --no-cache git+https://github.com/absadiki/pywhispercpp@v1.4.1
\`\`\`

**Performance gains**:
- Small model: 1.87x faster (0.50s vs 0.93s)
- Large model: 2.82x faster (1.15s vs 3.25s)

For detailed instructions, see [CoreML Build Guide](docs/SESSION_SUMMARY_2026-01-13_pywhispercpp_CoreML_Build_Guide.md).
```

---

### 4. Update README

**File**: `README.md`

**Section to Update**: Installation section (around lines 122-137)

**Addition 1**: Add note about CoreML after basic setup
```markdown
### Optional: Enable CoreML Acceleration

On Apple Silicon Macs, you can enable CoreML/Neural Engine acceleration for `WhisperCppCoreMLImplementation`:

\`\`\`bash
WHISPER_COREML=ON WHISPER_COREML_ALLOW_FALLBACK=ON \\
  uv pip install --no-cache git+https://github.com/absadiki/pywhispercpp@v1.4.1
\`\`\`

This provides 2-3x speedup. See [CoreML Build Guide](docs/SESSION_SUMMARY_2026-01-13_pywhispercpp_CoreML_Build_Guide.md) for details.
```

**Addition 2**: Update performance comparison table (if exists)
- Add CoreML-enabled results for small/large models
- Show comparative speedups

---

### 5. Commit Changes and Prepare for Contribution

**Branch Strategy**: As user requested "review commits first, decide later":
- Create feature branch locally
- Make atomic commits
- Review together before deciding on PR vs direct push

**Commit Structure**:

```
Commit 1: feat: add automatic CoreML model download to WhisperCppCoreMLImplementation
- Add _download_coreml_model() method to coreml.py
- Integrate download logic into load_model()
- Add error handling and logging

Commit 2: feat: add runtime CoreML availability check in CLI
- Add _check_coreml_availability() to cli.py
- Log helpful warning if CoreML not enabled
- Provide link to build guide

Commit 3: docs: update installation docs for CoreML build
- Update CLAUDE.md with CoreML build instructions
- Update README.md with optional CoreML setup
- Pin pywhispercpp to v1.4.1 in pyproject.toml

Commit 4: docs: add benchmark results comparing CoreML vs other implementations
- Add BENCHMARK_RESULTS_2026-01-13_CoreML_Comparison.md
- Document performance comparisons
- Include test methodology

Commit 5: chore: update documentation references
- Ensure all docs cross-reference correctly
- Update version numbers
- Add CoreML model download verification
```

**Commit Messages Follow**:
- Conventional Commits format
- Include context and rationale
- Reference related documentation

**PR Description** (if going with PR approach):
```markdown
## Summary

Adds automatic CoreML model downloading to `WhisperCppCoreMLImplementation` and documents the CoreML build process for 2-3x performance improvements.

## Changes

- **feat**: Auto-download CoreML `.mlmodelc` models from HuggingFace
- **feat**: Runtime detection and warning if CoreML not enabled
- **docs**: Comprehensive CoreML build guide with empirical benchmarks
- **docs**: Updated README and CLAUDE.md with installation instructions
- **chore**: Pin pywhispercpp to v1.4.1 for stability

## Performance Impact

With CoreML enabled (Apple Silicon):
- Small model: 1.87x faster (0.50s vs 0.93s)
- Large model: 2.82x faster (1.15s vs 3.25s)

## Testing

- [x] Tested CoreML model download and extraction
- [x] Validated graceful fallback when CoreML unavailable
- [x] Benchmarked against all 9 implementations
- [x] Verified on Apple M3 with macOS 14.x

## Documentation

- CoreML Build Guide: `docs/SESSION_SUMMARY_2026-01-13_pywhispercpp_CoreML_Build_Guide.md`
- Benchmark Results: `docs/BENCHMARK_RESULTS_2026-01-13_CoreML_Comparison.md`
```

---

### 6. Review Other Implementation Documentation

**Goal**: Identify similar optimization opportunities in other implementations

**Files to Review**:
- `docs/model_details_WhisperKitImplementation.md`
- `docs/model_details_MLXWhisperImplementation.md`
- `docs/model_details_FasterWhisperImplementation.md`
- `docs/model_details_LightningWhisperMLXImplementation.md`
- `docs/model_details_InsanelyFastWhisperImplementation.md`
- `docs/model_details_WhisperMPSImplementation.md`
- `docs/model_details_ParakeetMLXImplementation.md`
- `docs/model_details_FluidAudioCoreMLImplementation.md`

**Review Criteria**:
1. Are model downloads optimized?
2. Are there platform-specific optimizations being missed?
3. Are cache locations optimal?
4. Are there version pinning issues?
5. Are there known performance bottlenecks documented?

**Output**: Create priority list of next improvements based on:
- Performance impact potential
- Implementation effort
- User pain points documented

---

## Critical Files

### Files to Modify:
1. `src/mac_whisper_speedtest/implementations/coreml.py` (add download logic)
2. `src/mac_whisper_speedtest/cli.py` (add CoreML check)
3. `pyproject.toml` (pin version)
4. `CLAUDE.md` (add CoreML instructions)
5. `README.md` (add CoreML setup)

### Files to Create:
1. `docs/BENCHMARK_RESULTS_2026-01-13_CoreML_Comparison.md` (benchmark results)

### Files to Reference:
1. `docs/SESSION_SUMMARY_2026-01-13_pywhispercpp_CoreML_Build_Guide.md` (comprehensive guide)
2. `docs/model_details_WhisperCppCoreMLImplementation.md` (technical details)
3. `src/mac_whisper_speedtest/utils.py` (get_models_dir utility)
4. `src/mac_whisper_speedtest/implementations/mlx.py` (HuggingFace download pattern)
5. `src/mac_whisper_speedtest/implementations/__init__.py` (conditional import pattern)

---

## Verification Plan

### Phase 1: CoreML Download Verification
```bash
# Remove existing CoreML models
mv models/*.mlmodelc models/backup/

# Run benchmark (should trigger auto-download)
.venv/bin/python3 test_benchmark2.py small 1 "WhisperCppCoreMLImplementation"

# Verify:
# 1. Download initiated and completed
# 2. .mlmodelc extracted correctly
# 3. CoreML model loaded successfully
# 4. Transcription completes
```

### Phase 2: Fallback Verification
```bash
# Test with CoreML-disabled build
uv pip uninstall pywhispercpp
uv pip install git+https://github.com/absadiki/pywhispercpp@v1.4.1

# Run benchmark
.venv/bin/python3 test_benchmark2.py small 1 "WhisperCppCoreMLImplementation"

# Verify:
# 1. Warning logged about missing CoreML
# 2. Graceful fallback to Metal GPU
# 3. No download attempted (COREML = 0)
```

### Phase 3: Comprehensive Benchmark
```bash
# Run full comparison
.venv/bin/python3 test_benchmark2.py small 3
.venv/bin/python3 test_benchmark2.py large 3

# Verify:
# 1. All implementations complete successfully
# 2. CoreML implementation shows ~2-3x speedup
# 3. Results documented in benchmark file
```

### Phase 4: Documentation Verification
- [ ] All cross-references in docs resolve correctly
- [ ] Installation commands work on fresh setup
- [ ] CoreML build guide steps are accurate
- [ ] Benchmark results match observed performance

### Phase 5: Commit Review
- [ ] Each commit builds and tests pass
- [ ] Commit messages follow conventions
- [ ] No unrelated changes included
- [ ] Documentation updated consistently

---

## Dependencies and Prerequisites

### Required:
- Python 3.11+
- uv package manager
- macOS 14.0+ with Apple Silicon (for CoreML)
- Existing project dependencies from pyproject.toml

### For CoreML Support:
- Xcode command-line tools
- pywhispercpp v1.4.1 built with `WHISPER_COREML=ON WHISPER_COREML_ALLOW_FALLBACK=ON`

### For Benchmarking:
- All 9 Whisper implementations installed
- Test audio files (tests/jfk.wav)
- Sufficient disk space for model downloads (~2GB)

---

## Risk Assessment

### Low Risk:
- Adding download logic (isolated, well-tested pattern)
- Documentation updates (no code impact)
- Version pinning (explicit, controlled)

### Medium Risk:
- Benchmark consistency (environmental factors)
- Download reliability (network dependencies)

### Mitigations:
- Graceful fallback if download fails
- Retry logic with exponential backoff
- Clear error messages for troubleshooting
- Comprehensive logging of all steps

---

## Timeline Estimate

**Note**: Providing concrete implementation steps without time estimates as requested.

1. CoreML download implementation: Write, test, validate
2. CLI detection feature: Write, integrate, test
3. Documentation updates: Update all files, verify links
4. Run benchmarks: Execute and document results
5. Commit preparation: Atomic commits with clear messages
6. Documentation review: Read through other implementations

---

## Success Criteria

- [ ] CoreML models download automatically when missing
- [ ] Download fails gracefully with helpful error messages
- [ ] Fallback to Metal GPU works when CoreML unavailable
- [ ] Runtime detection warns users about missing CoreML
- [ ] Benchmarks show expected 2-3x speedup with CoreML
- [ ] Documentation is comprehensive and cross-referenced
- [ ] Installation process is well-documented
- [ ] Commits are atomic and well-messaged
- [ ] All tests pass
- [ ] No regressions in existing functionality
