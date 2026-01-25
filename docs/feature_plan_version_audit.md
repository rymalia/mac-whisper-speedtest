# Version Audit Report

**Generated:** 2026-01-14
**Purpose:** Comprehensive audit of all packages, Swift bridges, and dependencies with upgrade prioritization

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Overview](#current-state-overview)
3. [Implementation-Specific Dependencies](#implementation-specific-dependencies)
4. [Swift Bridge Dependencies](#swift-bridge-dependencies)
5. [Key Transitive Dependencies](#key-transitive-dependencies)
6. [Prioritized Upgrade Recommendations](#prioritized-upgrade-recommendations)
7. [Upgrade Plan & Best Practices](#upgrade-plan--best-practices)
8. [Appendix: Full uv Dry-Run Output](#appendix-full-uv-dry-run-output)

---

## Executive Summary

### Key Findings

| Category | Current | Latest | Gap | Urgency | Notes |
|----------|---------|--------|-----|---------|-------|
| **Swift: FluidAudio** | 0.1.0 | 0.10.0 | 9 minor versions | **CRITICAL** | API changes required (add `.v2`) |
| **Swift: WhisperKit** | 0.13.1 | 0.15.0 | 2 minor versions | **HIGH** | No code changes expected |
| **Python: MLX** | 0.27.1 | 0.30.3 | 3 minor versions | HIGH | Performance only, no API changes |
| **Python: transformers** | 4.51.3 | 4.57.5 | 6 patch versions | MEDIUM | |
| **Python: torch** | 2.6.0 | 2.9.1 | 3 minor versions | MEDIUM | |
| **Python: faster-whisper** | 1.1.1 | 1.2.1 | 1 minor version | MEDIUM | |
| **Python: parakeet-mlx** | 0.3.5 | 0.5.0 | 2 minor versions | MEDIUM | |

### Risk Assessment

- **183 packages** would be affected by a full upgrade
- **13 packages** would be removed (dependency cleanup)
- **12 new packages** would be added
- **1 anomalous version change** detected (transformers showing v2.3.0 in dry-run—likely a resolver artifact)

### Related Documents

Detailed upgrade plans with step-by-step instructions:

| Upgrade | Document | Risk | Phases |
|---------|----------|------|--------|
| FluidAudio 0.1→0.10 | [upgrade_plan_FluidAudio_0.1_to_0.10.md](upgrade_plan_FluidAudio_0.1_to_0.10.md) | HIGH (API changes required) | 3 |
| WhisperKit 0.13→0.15 | [upgrade_plan_WhisperKit_0.13_to_0.15.md](upgrade_plan_WhisperKit_0.13_to_0.15.md) | MEDIUM (no code changes expected) | 2 |
| MLX 0.27→0.30 | [upgrade_plan_MLX_0.27_to_0.30.md](upgrade_plan_MLX_0.27_to_0.30.md) | LOW-MEDIUM (performance only) | 2 |

### Hidden Native Code Warning

> **⚠️ Bundled Native Libraries**
>
> Some packages bundle native C/C++ code that is **invisible to pip/uv**:
>
> | Package | Bundles | Native Version | How to Check |
> |---------|---------|----------------|--------------|
> | `pywhispercpp` | whisper.cpp | 1.8.2 | `ls .venv/.../pywhispercpp/.dylibs/libwhisper.*.dylib` |
>
> When upgrading these packages, check BOTH the Python package changelog AND the underlying native library releases.

---

## Current State Overview

### Direct Dependencies (from pyproject.toml)

| Package | Declared Version | Locked Version | Latest Available | Update Needed? |
|---------|-----------------|----------------|------------------|----------------|
| `mlx` | >=0.5.0 | 0.27.1 | 0.30.3 | YES |
| `mlx-whisper` | >=0.4.2 | 0.4.2 | 0.4.3 | YES |
| `faster-whisper` | >=1.1.1 | 1.1.1 | 1.2.1 | YES |
| `parakeet-mlx` | >=0.3.5 | 0.3.5 | 0.5.0 | YES |
| `whisper-mps` | >=0.0.7 | 0.0.7 | 0.0.10 | YES |
| `lightning-whisper-mlx` | >=0.0.10 | (locked) | — | CHECK |
| `coremltools` | >=8.0.0 | 8.3.0 | 9.0 | YES |
| `huggingface-hub` | >=0.20.0 | 0.30.2 | 1.3.1 | YES (major!) |
| `numpy` | >=2.2.5 | 2.2.6 | 2.3.5 | YES |
| `typer` | >=0.9.0 | 0.16.0 | 0.21.1 | YES |
| `pywhispercpp` | git main | 1.3.1.dev38 | 1.4.1 | YES |

---

## Implementation-Specific Dependencies

Each of the 9 Whisper implementations has specific package dependencies. Here's how outdated packages affect each:

### 1. MLXWhisperImplementation (`mlx.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `mlx` | 0.27.1 | 0.30.3 | Missing M5 Neural Accelerator optimizations, CUDA backend, performance improvements |
| `mlx-metal` | 0.27.1 | 0.30.3 | Metal GPU optimizations |
| `mlx-whisper` | 0.4.2 | 0.4.3 | Bug fixes, potential new model support |

**Impact:** Medium-High. MLX 0.30+ includes significant performance improvements for Apple Silicon, especially with macOS 26.2+ and M5 chips.

> **Note:** MLX v0.30.0 introduced a breaking change: "Default strict mode for module `update` and `update_modules`". This does NOT affect our implementations because we don't call MLX APIs directly—wrapper libraries handle this internally.

### 2. LightningWhisperMLXImplementation (`lightning.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `lightning-whisper-mlx` | 0.0.10+ | — | Needs verification |
| `mlx` (transitive) | 0.27.1 | 0.30.3 | Same as above |

**Impact:** Medium. Benefits from MLX core updates.

### 3. ParakeetMLXImplementation (`parakeet_mlx.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `parakeet-mlx` | 0.3.5 | 0.5.0 | Missing 2 minor versions of improvements, possible new features |
| `mlx` (transitive) | 0.27.1 | 0.30.3 | Performance |

**Impact:** Medium. The Parakeet model family is actively developed; updates may include accuracy improvements.

### 4. InsanelyFastWhisperImplementation (`insanely.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `insanely-fast-whisper` | 0.0.15 | — | Check for updates |
| `transformers` | 4.51.3 | 4.57.5 | New model architectures, optimizations, bug fixes |
| `torch` | 2.6.0 | 2.9.1 | MPS backend improvements, memory optimizations |
| `accelerate` | 1.6.0 | 1.12.0 | Hardware acceleration improvements |

**Impact:** High. This implementation relies heavily on PyTorch's MPS backend; torch 2.9.1 has significant improvements.

### 5. WhisperMPSImplementation (`whisper_mps.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `whisper-mps` | 0.0.7 | 0.0.10 | 3 versions behind; bug fixes, optimizations |
| `mlx` (internal) | 0.27.1 | 0.30.3 | This package actually uses MLX internally despite the name |

**Impact:** Medium. Note: As documented in CODEBASE_EXPLORATION.md, whisper-mps actually uses MLX, not PyTorch MPS.

### 6. FasterWhisperImplementation (`faster.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `faster-whisper` | 1.1.1 | 1.2.1 | distil-large-v3.5 support, private HF models, revision downloads |
| `ctranslate2` | 4.6.0 | 4.6.3 | Bug fixes, optimizations |

**Impact:** Medium. The 1.2.0 release adds support for newer distil-whisper models.

### 7. WhisperCppCoreMLImplementation (`coreml.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| `pywhispercpp` | 1.3.1.dev38 | 1.4.1 | Using git dev version; stable release available |
| `coremltools` | 8.3.0 | 9.0 | Major version! New CoreML features, optimizations |
| `whisper.cpp` (bundled) | 1.8.2 | Check releases | **Hidden** - compiled into pywhispercpp |

**Impact:** High. coremltools 9.0 is a major release with potential breaking changes but also significant improvements for Apple Neural Engine.

> **⚠️ Hidden Dependency:** `pywhispercpp` bundles `whisper.cpp` as compiled native code.
> This is NOT visible in `uv.lock` or `pip list`.
>
> **Version mapping:**
> - pywhispercpp v1.3.0 → whisper.cpp v1.7.0
> - pywhispercpp v1.4.0 → whisper.cpp v1.8.2
>
> Check [whisper.cpp releases](https://github.com/ggml-org/whisper.cpp/releases) for breaking changes when upgrading pywhispercpp.

### 8. WhisperKitImplementation (`whisperkit.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| **WhisperKit (Swift)** | 0.13.1 | 0.15.0 | TranscriptionResult class, swift-transformers 1.1.2, Local Server feature |
| swift-transformers | 0.1.15 | 1.1.2+ | Major version jump in transitive dependency |

**Impact:** HIGH. Two minor versions behind. The 0.15.0 release includes important API changes (TranscriptionResult is now a class, not struct).

### 9. FluidAudioCoreMLImplementation (`fluidaudio_coreml.py`)

| Dependency | Current | Latest | Impact if Outdated |
|------------|---------|--------|-------------------|
| **FluidAudio (Swift)** | 0.1.0 | 0.10.0 | **9 versions behind!** Missing Sortformer diarization, Swift 6 support, streaming ASR |

**Impact:** CRITICAL. This is the most outdated dependency. FluidAudio has had extensive development including real-time speaker diarization (Sortformer), Parakeet EOU streaming, and Swift 6 compatibility.

---

## Swift Bridge Dependencies

### whisperkit-bridge (`tools/whisperkit-bridge/Package.swift`)

**Current Package.resolved:**
```
whisperkit: 0.13.1
swift-transformers: 0.1.15
swift-argument-parser: 1.6.1
swift-collections: 1.2.1
jinja: 1.2.1
```

**Latest Available:**
```
whisperkit: 0.15.0 (+2 minor)
swift-transformers: 1.1.2+ (major version jump!)
swift-argument-parser: ~1.6.1 (current)
```

**Update Complexity:** MEDIUM
- WhisperKit 0.15.0 promotes `TranscriptionResult` from struct to class (API change)
- swift-transformers major version bump may require code changes

### fluidaudio-bridge (`tools/fluidaudio-bridge/Package.swift`)

**Current Package.resolved:**
```
fluidaudio: 0.1.0
swift-argument-parser: 1.6.1
```

**Latest Available:**
```
fluidaudio: 0.10.0 (+9 minor versions!)
swift-argument-parser: ~1.6.1 (current)
```

**Update Complexity:** HIGH
- 9 minor versions likely include breaking API changes
- New features: Sortformer speaker diarization, Swift 6 support, Parakeet streaming
- May require significant bridge code updates

---

## Key Transitive Dependencies

These packages are not directly declared but critically affect functionality:

### Machine Learning Core

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| `torch` | 2.6.0 | 2.9.1 | Core ML backend for insanely-fast-whisper |
| `torchaudio` | 2.6.0 | 2.9.1 | Audio processing |
| `accelerate` | 1.6.0 | 1.12.0 | HuggingFace hardware acceleration |
| `safetensors` | 0.5.3 | 0.7.0 | Model loading speed/security |
| `tokenizers` | 0.21.1 | 0.22.2 | Text tokenization performance |

### Apple Silicon Specific

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| `mlx` | 0.27.1 | 0.30.3 | **Key package** - Apple's ML framework |
| `mlx-metal` | 0.27.1 | 0.30.3 | Metal GPU backend |
| `coremltools` | 8.3.0 | 9.0 | **Major version** - CoreML model tools |

### HuggingFace Ecosystem

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| `huggingface-hub` | 0.30.2 | 1.3.1 | **Major version jump!** Model download/caching |
| `hf-xet` | 1.0.3 | 1.2.0 | HuggingFace experimental features |
| `transformers` | 4.51.3 | 4.57.5 | Core NLP/ASR library |

### Scientific Computing

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| `numpy` | 2.2.6 | 2.3.5 | Array operations, breaking changes possible |
| `scipy` | 1.15.2 | 1.17.0 | Scientific computing |
| `scikit-learn` | 1.6.1 | 1.8.0 | ML utilities |

---

## Prioritized Upgrade Recommendations

### Priority 1: CRITICAL (Do First)

#### 1.1 FluidAudio Swift Bridge
- **Current:** 0.1.0 → **Target:** 0.10.0
- **Why Critical:** 9 versions behind, extensive new features including Sortformer speaker diarization
- **Complications:**
  - Major API changes likely
  - May require bridge code rewrite
  - Swift 6 migration may affect build process
- **Effort:** Medium-High (1-2 hours code review, testing)

#### 1.2 MLX Core Update
- **Current:** 0.27.1 → **Target:** 0.30.3
- **Why Critical:** Performance improvements for Apple Silicon, M5 support preparation
- **Complications:**
  - Affects exactly 4 implementations: MLXWhisperImplementation, WhisperMPSImplementation, LightningWhisperMLXImplementation, ParakeetMLXImplementation
  - **Must be done atomically with mlx-metal** (see warning below)
- **Effort:** Low (version bump, full test suite)

> **⚠️ CRITICAL:** `mlx` and `mlx-metal` MUST always have the same version. Never upgrade one without the other.
> ```bash
> # ALWAYS upgrade both together:
> uv lock --upgrade-package mlx --upgrade-package mlx-metal
> ```

### Priority 2: HIGH (Do Soon)

#### 2.1 WhisperKit Swift Bridge
- **Current:** 0.13.1 → **Target:** 0.15.0
- **Why High:** TranscriptionResult API change (struct→class), dependency updates
- **Complications:**
  - Need to verify bridge code handles the class change
  - swift-transformers major version jump
- **Effort:** Medium (test bridge behavior, possible code changes)

#### 2.2 coremltools Major Update
- **Current:** 8.3.0 → **Target:** 9.0
- **Why High:** Major version with significant CoreML improvements
- **Complications:**
  - Breaking changes possible
  - Affects pywhispercpp CoreML model conversion
- **Effort:** Medium (test CoreML model loading/conversion)

#### 2.3 huggingface-hub Major Update
- **Current:** 0.30.2 → **Target:** 1.3.1
- **Why High:** Major version jump, affects all HuggingFace model downloads
- **Complications:**
  - API changes in caching, downloads
  - May affect multiple implementations
- **Effort:** Medium-High (test all model download paths)

### Priority 3: MEDIUM (Planned Update)

#### 3.1 faster-whisper
- **Current:** 1.1.1 → **Target:** 1.2.1
- **Why Medium:** New model support (distil-large-v3.5), feature additions
- **Complications:** Minor, well-tested library
- **Effort:** Low

#### 3.2 parakeet-mlx
- **Current:** 0.3.5 → **Target:** 0.5.0
- **Why Medium:** 2 minor versions, likely improvements
- **Complications:** None expected
- **Effort:** Low

#### 3.3 whisper-mps
- **Current:** 0.0.7 → **Target:** 0.0.10
- **Why Medium:** Bug fixes, optimizations
- **Complications:** None expected
- **Effort:** Low

#### 3.4 torch/torchaudio
- **Current:** 2.6.0 → **Target:** 2.9.1
- **Why Medium:** MPS backend improvements, memory optimizations
- **Complications:**
  - Large download (~2GB)
  - May require coordinated update with transformers
- **Effort:** Medium (mostly download/test time)

### Priority 4: LOW (Opportunistic)

- `typer` 0.16.0 → 0.21.1 (CLI framework, no feature impact)
- `numpy` 2.2.6 → 2.3.5 (test thoroughly for breaking changes)
- Various utility packages (attrs, click, etc.)

---

## Upgrade Plan & Best Practices

### For a Novice: What You Need to Know

**Why version management matters:**
1. **Security:** Older versions may have known vulnerabilities
2. **Performance:** ML libraries often have significant speed improvements
3. **Compatibility:** Newer models may require newer library versions
4. **Bugs:** Bug fixes accumulate in newer versions

**Key concepts:**
- **Lock file (`uv.lock`):** Records exact versions installed; ensures reproducibility
- **Semantic versioning:** MAJOR.MINOR.PATCH (e.g., 4.51.3)
  - MAJOR: Breaking changes possible
  - MINOR: New features, backward compatible
  - PATCH: Bug fixes only
- **Dry-run:** Preview changes without making them

### Recommended Upgrade Sequence

```
Phase 1: Foundation (MLX ecosystem)
├── Step 1.1: Create upgrade branch
├── Step 1.2: Update mlx + mlx-metal together
├── Step 1.3: Run all 9 implementation tests
└── Step 1.4: Commit if passing

Phase 2: Swift Bridges
├── Step 2.1: Update WhisperKit bridge
├── Step 2.2: Test WhisperKit implementation
├── Step 2.3: Update FluidAudio bridge (may require code changes)
├── Step 2.4: Test FluidAudio implementation
└── Step 2.5: Commit if passing

Phase 3: HuggingFace Ecosystem
├── Step 3.1: Update huggingface-hub (major version!)
├── Step 3.2: Update transformers
├── Step 3.3: Test all implementations that download from HuggingFace
└── Step 3.4: Commit if passing

Phase 4: Individual Implementations
├── Step 4.1: Update faster-whisper
├── Step 4.2: Update parakeet-mlx
├── Step 4.3: Update whisper-mps
├── Step 4.4: Update coremltools (major version!)
└── Step 4.5: Full test suite, commit if passing

Phase 5: PyTorch Ecosystem
├── Step 5.1: Update torch + torchaudio together
├── Step 5.2: Update accelerate
├── Step 5.3: Test insanely-fast-whisper thoroughly
└── Step 5.4: Final commit
```

### Natural Phase Breakpoints

For complex upgrades, stop at these intermediate versions to isolate issues:

| Dependency | Phase 1 | Phase 2 | Phase 3 |
|------------|---------|---------|---------|
| **FluidAudio** | v0.4.1 (AsrModelVersion enum) | v0.8.2 (streaming features) | v0.10.0 (Swift 6, Sortformer) |
| **WhisperKit** | v0.14.1 (swift-transformers 1.0) | v0.15.0 (TranscriptionResult class) | — |
| **MLX** | v0.29.4 (mxfp4 quantization) | v0.30.3 (Neural Accelerator) | — |

See the [Related Documents](#related-documents) section for detailed upgrade plans with step-by-step instructions.

### Git Best Practices for Upgrades

#### Before Starting

```bash
# 1. Ensure you're on a clean main branch
git status
git checkout main
git pull origin main

# 2. Create a dedicated upgrade branch
git checkout -b chore/version-upgrades-2026-01

# 3. Verify current state works
.venv/bin/mac-whisper-speedtest -b  # Should pass for all implementations
```

#### During Upgrades

```bash
# After each phase, if tests pass:
git add uv.lock pyproject.toml
git commit -m "chore(deps): update [package-name] to v[version]

- Previous: [old-version]
- Current: [new-version]
- Reason: [brief why]

Tested: [which implementations]"

# If something breaks, you can revert just that change:
git revert HEAD
```

#### Handling Python Package Upgrades

```bash
# Preview what would change
uv lock --upgrade --dry-run

# Update a single package
uv lock --upgrade-package mlx

# Update multiple related packages together
uv lock --upgrade-package mlx --upgrade-package mlx-metal

# After updating lock file, sync your environment
uv sync

# Verify the installation
.venv/bin/python -c "import mlx; print(mlx.__version__)"
```

#### Handling Swift Bridge Upgrades

```bash
# 1. Update Package.swift version constraint
# In tools/whisperkit-bridge/Package.swift:
# Change: .package(url: "...", from: "0.13.1")
# To:     .package(url: "...", from: "0.15.0")

# 2. Resolve and update
cd tools/whisperkit-bridge
swift package resolve
swift package update

# 3. Rebuild
swift build -c release

# 4. Test
.venv/bin/mac-whisper-speedtest -b -m small -n 1 -i WhisperKitImplementation
```

#### After Completing All Upgrades

```bash
# Squash or keep commits based on preference
# Option A: Keep detailed history (recommended for learning)
git push origin chore/version-upgrades-2026-01

# Option B: Squash into single commit
git rebase -i main  # Mark all but first as 'squash'
git push origin chore/version-upgrades-2026-01

# Create Pull Request for review
gh pr create --title "chore(deps): version upgrades 2026-01" \
  --body "See docs/feature_plan_version_audit.md for details"
```

### Rollback Strategy

If an upgrade causes problems:

```bash
# Option 1: Revert specific commit
git log --oneline  # Find the problematic commit
git revert <commit-hash>

# Option 2: Reset to known good state
git checkout main -- uv.lock
uv sync

# Option 3: Full rollback to main
git checkout main
git branch -D chore/version-upgrades-2026-01  # Delete failed branch
```

### Testing Checklist

After each upgrade phase, verify:

- [ ] `uv sync` completes without errors
- [ ] `pytest tests/ -v` passes
- [ ] `.venv/bin/mac-whisper-speedtest -b` runs all 9 implementations
- [ ] No new deprecation warnings in output
- [ ] Model downloads still work (test with fresh cache if needed)

---

## Appendix: Full uv Dry-Run Output

<details>
<summary>Click to expand full `uv lock --upgrade --dry-run` output</summary>

```
Resolved 183 packages in 3.66s
Update accelerate v1.6.0 -> v1.12.0
Update aiohttp v3.11.16 -> v3.13.3
Update aiosignal v1.3.2 -> v1.4.0
Update alembic v1.15.2 -> v1.18.0
Remove antlr4-python3-runtime v4.9.3
Add anyio v4.12.1
Update attrs v25.3.0 -> v25.4.0
Update audioop-lts v0.2.1 -> v0.2.2
Update audioread v3.0.1 -> v3.1.0
Update av v14.3.0 -> v16.1.0
Add boto3 v1.42.27
Add botocore v1.42.27
Update cattrs v25.1.1 -> v25.3.0
Update certifi v2025.1.31 -> v2026.1.4
Update cffi v1.17.1 -> v2.0.0
Update charset-normalizer v3.4.1 -> v3.4.4
Update click v8.1.8 -> v8.3.1
Update colorlog v6.9.0 -> v6.10.1
Update contourpy v1.3.2 -> v1.3.3
Update coremltools v8.3.0 -> v9.0
Update ctranslate2 v4.6.0 -> v4.6.3
Update decorator v4.4.2 -> v5.2.1
Remove docopt v0.6.2
Update faster-whisper v1.1.1 -> v1.2.1
Update filelock v3.18.0 -> v3.20.3
Update flatbuffers v25.2.10 -> v25.12.19
Update fonttools v4.57.0 -> v4.61.1
Update frozenlist v1.6.0 -> v1.8.0
Update fsspec v2025.3.0 -> v2026.1.0
Add googleapis-common-protos v1.72.0
Update greenlet v3.2.0 -> v3.3.0
Add grpcio v1.76.0
Add h11 v0.16.0
Update hf-xet v1.0.3 -> v1.2.0
Add httpcore v1.0.9
Add httpx v0.28.1
Update huggingface-hub v0.30.2 -> v1.3.1
Remove hyperpyyaml v1.2.2
Update idna v3.10 -> v3.11
Update imageio v2.37.0 -> v2.37.2
Add importlib-metadata v8.7.1
Update iniconfig v2.1.0 -> v2.3.0
Add jmespath v1.0.1
Update joblib v1.4.2 -> v1.5.3
Update kiwisolver v1.4.8 -> v1.4.9
Update lightning v2.4.0 -> v2.6.0
Update lightning-utilities v0.14.3 -> v0.15.2
Update llvmlite v0.44.0 -> v0.46.0
Update markdown-it-py v3.0.0 -> v4.0.0
Update markupsafe v3.0.2 -> v3.0.3
Update matplotlib v3.10.1 -> v3.10.8
Update mlx v0.27.1 -> v0.30.3
Update mlx-metal v0.27.1 -> v0.30.3
Update mlx-whisper v0.4.2 -> v0.4.3
Update more-itertools v10.7.0 -> v10.8.0
Update moviepy v1.0.3 -> v2.2.1
Update msgpack v1.1.1 -> v1.1.2
Update multidict v6.4.3 -> v6.7.0
Update networkx v3.4.2 -> v3.6.1
Update numba v0.61.2 -> v0.63.1
Update numpy v2.2.6 -> v2.3.5
Update nvidia-cublas-cu12 v12.4.5.8 -> v12.8.4.1
Update nvidia-cuda-cupti-cu12 v12.4.127 -> v12.8.90
Update nvidia-cuda-nvrtc-cu12 v12.4.127 -> v12.8.93
Update nvidia-cuda-runtime-cu12 v12.4.127 -> v12.8.90
Update nvidia-cudnn-cu12 v9.1.0.70 -> v9.10.2.21
Update nvidia-cufft-cu12 v11.2.1.3 -> v11.3.3.83
Add nvidia-cufile-cu12 v1.13.1.3
Update nvidia-curand-cu12 v10.3.5.147 -> v10.3.9.90
Update nvidia-cusolver-cu12 v11.6.1.9 -> v11.7.3.90
Update nvidia-cusparse-cu12 v12.3.1.170 -> v12.5.8.93
Update nvidia-cusparselt-cu12 v0.6.2 -> v0.7.1
Update nvidia-nccl-cu12 v2.21.5 -> v2.27.3, v2.27.5
Update nvidia-nvjitlink-cu12 v12.4.127 -> v12.8.93
Add nvidia-nvshmem-cu12 v3.3.20
Update nvidia-nvtx-cu12 v12.4.127 -> v12.8.90
Remove omegaconf v2.3.0
Update onnxruntime v1.21.1 -> v1.23.2
Add opentelemetry-api v1.39.1
Add opentelemetry-exporter-otlp v1.39.1
Add opentelemetry-exporter-otlp-proto-common v1.39.1
Add opentelemetry-exporter-otlp-proto-grpc v1.39.1
Add opentelemetry-exporter-otlp-proto-http v1.39.1
Add opentelemetry-proto v1.39.1
Add opentelemetry-sdk v1.39.1
Add opentelemetry-semantic-conventions v0.60b1
Update optuna v4.3.0 -> v4.6.0
Update packaging v24.2 -> v25.0
Update pandas v2.2.3 -> v2.3.3
Update parakeet-mlx v0.3.5 -> v0.5.0
Update pillow v11.2.1 -> v11.3.0
Update platformdirs v4.3.7 -> v4.5.1
Update propcache v0.3.1 -> v0.4.1
Update protobuf v6.30.2 -> v6.33.4
Update psutil v7.0.0 -> v7.2.1
Update pyannote-audio v3.3.2 -> v4.0.1, v4.0.3
Update pyannote-core v5.0.0 -> v6.0.1
Update pyannote-database v5.1.3 -> v6.1.1
Update pyannote-metrics v3.2.1 -> v4.0.0
Update pyannote-pipeline v3.0.1 -> v4.0.0
Add pyannoteai-sdk v0.3.0
Update pycparser v2.22 -> v2.23
Update pygments v2.19.1 -> v2.19.2
Update pyparsing v3.2.3 -> v3.3.1
Update pytest v8.4.1 -> v9.0.2
Update pytest-asyncio v1.1.0 -> v1.3.0
Add python-dotenv v1.2.1
Update pytorch-lightning v2.5.1 -> v2.6.0
Update pytorch-metric-learning v2.8.1 -> v2.9.0
Update pywhispercpp v1.3.1.dev38+g8ecb35b (8ecb35b5) -> v1.4.1 (d8f202f4)
Update pyyaml v6.0.2 -> v6.0.3
Update regex v2024.11.6 -> v2025.11.3
Update requests v2.32.3 -> v2.32.5
Update rich v14.0.0 -> v14.2.0
Remove ruamel-yaml v0.18.10
Remove ruamel-yaml-clib v0.2.12
Add s3transfer v0.16.0
Add sacremoses v0.1.1
Update safetensors v0.5.3 -> v0.7.0
Update scikit-learn v1.6.1 -> v1.8.0
Update scipy v1.15.2 -> v1.17.0
Remove semver v3.0.4
Update sentencepiece v0.2.0 -> v0.2.1
Update setuptools v78.1.0 -> v80.9.0
Update soxr v0.5.0.post1 -> v1.0.0
Remove speechbrain v1.0.3
Update sqlalchemy v2.0.40 -> v2.0.45
Update structlog v25.2.0 -> v25.5.0
Update sympy v1.13.1 -> v1.14.0
Remove tabulate v0.9.0
Remove tensorboardx v2.6.2.2
Update tokenizers v0.21.1 -> v0.22.2
Update torch v2.6.0 -> v2.8.0, v2.9.1
Update torchaudio v2.6.0 -> v2.8.0, v2.9.1
Add torchcodec v0.7.0, v0.9.1
Update torchmetrics v1.7.1 -> v1.8.2
Update transformers v4.51.3 -> v2.3.0
Update triton v3.2.0 -> v3.4.0, v3.5.1
Update typer v0.16.0 -> v0.21.1
Add typer-slim v0.21.1
Update typing-extensions v4.13.2 -> v4.15.0
Update tzdata v2025.2 -> v2025.3
Update urllib3 v2.4.0 -> v2.6.3
Update whisper-mps v0.0.7 -> v0.0.10
Update yarl v1.20.0 -> v1.22.0
Add zipp v3.23.0
```

</details>

### Anomaly Notes

**`transformers v4.51.3 -> v2.3.0`**: This appears to be a resolver artifact or display bug. The actual latest stable transformers is v4.57.5. **Do NOT downgrade transformers to v2.3.0.**

**`torch v2.6.0 -> v2.8.0, v2.9.1`**: Multiple versions shown likely indicate platform-specific resolution. Target v2.9.1 for Apple Silicon.

---

## Sources

- [WhisperKit Releases](https://github.com/argmaxinc/WhisperKit/releases)
- [FluidAudio Releases](https://github.com/FluidInference/FluidAudio/releases)
- [MLX Releases](https://github.com/ml-explore/mlx/releases)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [faster-whisper Releases](https://github.com/SYSTRAN/faster-whisper/releases)
- [transformers PyPI](https://pypi.org/project/transformers/)
- [PyTorch PyPI](https://pypi.org/project/torch/)
- [Transformers v5 Blog](https://huggingface.co/blog/transformers-v5)

### Hidden Dependency References

- [whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases) — Native library bundled in pywhispercpp
- [pywhispercpp Releases](https://github.com/absadiki/pywhispercpp/releases) — Shows which whisper.cpp version is bundled

> **Tip:** When researching release notes for GitHub repos, try the `/releases` URL first (e.g., `github.com/org/repo/releases`). Many modern projects use GitHub Releases exclusively and don't maintain a separate `CHANGELOG.md` file.
