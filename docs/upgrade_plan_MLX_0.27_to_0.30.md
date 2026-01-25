# MLX Core Upgrade Plan: v0.27.1 → v0.30.3

**Created:** 2026-01-14
**Purpose:** Phased upgrade strategy for MLX core and dependent packages with discrete commits

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Version History & Key Changes](#version-history--key-changes)
4. [Phased Upgrade Strategy](#phased-upgrade-strategy)
5. [Implementation Details](#implementation-details)
6. [Coordinated Dependencies](#coordinated-dependencies)
7. [Git Workflow & Commit Strategy](#git-workflow--commit-strategy)
8. [Risk Mitigation](#risk-mitigation)
9. [Testing Checklist](#testing-checklist)

---

## Executive Summary

### The Gap

| Package | Current | Target | Gap | Risk Level |
|---------|---------|--------|-----|------------|
| `mlx` | 0.27.1 | 0.30.3 | 3 minor versions | **MEDIUM** |
| `mlx-metal` | 0.27.1 | 0.30.3 | 3 minor versions | **MEDIUM** |
| `mlx-whisper` | 0.4.2 | 0.4.3 | 1 patch version | LOW |

### Key Finding: Performance-Focused Upgrade

Unlike FluidAudio or WhisperKit, the MLX upgrade is primarily about **performance gains**, not API changes:

- **No breaking changes** identified in the 0.27 → 0.30 transition
- **M5 Neural Accelerator support** (macOS 26.2+) enables 19-27% performance boost
- **New quantization formats** (mxfp4, mxfp8) for memory efficiency
- APIs remain NumPy-compatible throughout

### Recommended Approach: 2-Phase Incremental Upgrade

| Phase | Target Version | Key Changes | Impact on Implementations |
|-------|---------------|-------------|---------------------------|
| **Phase 1** | 0.29.x | mxfp4 quantization, NCCL backend, stability | Performance improvements only |
| **Phase 2** | 0.30.3 | Neural Accelerator support, RDMA, nvfp4/mxfp8 | Performance + M5 optimization |

### Affected Implementations (4 of 9)

| Implementation | File | MLX Dependency Path |
|----------------|------|---------------------|
| MLXWhisperImplementation | `mlx.py` | Direct via `mlx-whisper` |
| WhisperMPSImplementation | `whisper_mps.py` | Indirect via `whisper-mps` (uses MLX internally) |
| LightningWhisperMLXImplementation | `lightning.py` | Indirect via `lightning-whisper-mlx` |
| ParakeetMLXImplementation | `parakeet_mlx.py` | Indirect via `parakeet-mlx` |

---

## Current State Analysis

### Package Versions (from uv.lock)

| Package | Locked Version | pyproject.toml Constraint |
|---------|---------------|---------------------------|
| `mlx` | 0.27.1 | `>=0.5.0` |
| `mlx-metal` | 0.27.1 | (transitive) |
| `mlx-whisper` | 0.4.2 | `>=0.4.2` |
| `parakeet-mlx` | 0.3.5 | `>=0.3.5` |
| `lightning-whisper-mlx` | 0.0.10 | `>=0.0.10` |
| `whisper-mps` | 0.0.7 | `>=0.0.7` |

### Dependency Graph

```
pyproject.toml
├── mlx >=0.5.0 ─────────────────► mlx 0.27.1
│                                      │
│                                      └── mlx-metal 0.27.1 (always same version)
│
├── mlx-whisper >=0.4.2 ─────────► mlx-whisper 0.4.2
│                                      │
│                                      └── requires: mlx
│
├── parakeet-mlx >=0.3.5 ────────► parakeet-mlx 0.3.5
│                                      │
│                                      └── requires: mlx
│
├── lightning-whisper-mlx >=0.0.10 ► lightning-whisper-mlx 0.0.10
│                                      │
│                                      └── requires: mlx
│
└── whisper-mps >=0.0.7 ─────────► whisper-mps 0.0.7
                                       │
                                       └── requires: mlx (internally)
```

### Critical Constraint: mlx and mlx-metal Must Match

**IMPORTANT:** The `mlx` and `mlx-metal` packages must ALWAYS have the same version. The Metal backend (`mlx-metal`) provides GPU acceleration for MLX on Apple Silicon and is versioned in lockstep.

```python
# This will fail:
mlx==0.30.3
mlx-metal==0.27.1  # Version mismatch!

# This is correct:
mlx==0.30.3
mlx-metal==0.30.3  # Same version
```

### Implementation Usage Patterns

#### MLXWhisperImplementation (`mlx.py:136-146`)
```python
from mlx_whisper import transcribe

result = transcribe(
    audio=audio,
    path_or_hf_repo=self._model_path,
    temperature=0.0,
    language=self.language,
    task="transcribe"
)
```
**MLX Usage:** Indirect via `mlx_whisper` — no direct MLX API calls.

#### WhisperMPSImplementation (`whisper_mps.py:95-106`)
```python
from whisper_mps.whisper.transcribe import transcribe

result = transcribe(
    audio=audio,
    model=self.model_name,
    temperature=0.0,
    language=self.language,
)
```
**MLX Usage:** Indirect via `whisper_mps` — the library uses `import mlx.core as mx` internally.

#### LightningWhisperMLXImplementation (`lightning.py:56,115`)
```python
from lightning_whisper_mlx import LightningWhisperMLX

self.whisper_model = LightningWhisperMLX(model=mapped_model_name)
result = self.whisper_model.transcribe(audio_path=temp_path)
```
**MLX Usage:** Indirect via `lightning_whisper_mlx` — no direct MLX API calls.

#### ParakeetMLXImplementation (`parakeet_mlx.py:37,132`)
```python
from parakeet_mlx import from_pretrained

self._model = from_pretrained(self._hf_repo)
result = self._model.transcribe(temp_file.name)
```
**MLX Usage:** Indirect via `parakeet_mlx` — no direct MLX API calls.

### Conclusion: Low Risk for Our Code

None of our implementations call MLX APIs directly. They all use higher-level wrapper libraries (`mlx-whisper`, `whisper-mps`, `lightning-whisper-mlx`, `parakeet-mlx`) that abstract MLX operations.

**This means:** As long as the wrapper libraries remain compatible with MLX 0.30.x, our code requires no changes.

---

## Version History & Key Changes

### MLX Release Timeline (0.27.1 → 0.30.3)

| Version | Date | Category | Key Changes |
|---------|------|----------|-------------|
| **0.27.1** | Jul 25, 2025 | Milestone | Initial PyPI CUDA backend, graph support, complex numbers |
| **0.28.0** | Aug 7, 2025 | Performance | Fused SDPA for CUDA, convolutions, vectorized kernels, LRU cache |
| **0.29.0** | Aug 29, 2025 | Feature | **mxfp4 quantization (Metal, CPU)**, NCCL backend, Pathlib support |
| 0.29.1 | Sep 12, 2025 | Stability | Bug fixes |
| 0.29.2 | Sep 26, 2025 | Stability | Bug fixes |
| 0.29.3 | Oct 17, 2025 | Stability | Bug fixes |
| 0.29.4 | Nov 11, 2025 | Stability | Bug fixes |
| **0.30.0** | Nov 20, 2025 | **Major** | **Neural Accelerators on M5 (macOS >= 26.2)**, `mx.median`, compiled function improvements |
| **0.30.1** | Dec 18, 2025 | Feature | **RDMA over Thunderbolt (JACCL)**, NAX JIT for MLX Swift, faster quant ops, QQMM tensor cores |
| **0.30.3** | Jan 14, 2026 | Feature | nvfp4/mxfp8 quantized ops (Metal), faster memory transfers |

### Detailed Release Notes

<details>
<summary><strong>v0.30.3</strong> (January 13, 2025) - Click to expand</summary>

**New Features:**
- Support for nvfp4 and mxfp8 quantized operations on Metal
- Support for nvfp4 and mxfp8 quantized-quantized matrix multiplication on CUDA
- Added `asarray` to array namespace

**Bug Fixes:**
- Fixed CUDA release build issues (multiple patches)
- Corrected `grid_dim_x` calculations
- Fixed `RandomBits::is_equivalent` to include width parameter
- Resolved non-row-contiguous scales issues
- Fixed numpy dtype conversion bugs

**Performance:**
- Faster column-contiguous to row-contiguous copy operations
</details>

<details>
<summary><strong>v0.30.1</strong> (December 18, 2024) - Click to expand</summary>

**New Features:**
- RDMA over Thunderbolt with JACCL backend (macOS >= 26.2)
- NAX with JIT support for MLX Swift compatibility
- Enhanced SDPA with masking and T_q != T_kv support
- QQMM implementation for faster tensor core utilization
- `mx.depends` exposed to Python

**Bug Fixes:**
- Fixed CUDA allocator copy conditions
- Corrected large-size random generation for CUDA
- Resolved graph update issues
- Fixed attention for large tensor sizes

**Performance:**
- Faster CUDA quantize/dequantize operations
- Improved column reduce speeds for training
- Enhanced RMS norm performance for small dimensions
</details>

<details>
<summary><strong>v0.30.0</strong> (November 19, 2024) - Click to expand</summary>

**New Features:**
- **Support for Neural Accelerators on M5 (macOS >= 26.2)**
- Added `mx.median` operation
- FP8 conversion capabilities
- Quantize/dequantize for mxfp8 and nvfp4
- Export with callback functionality
- `mx.depends` operation

**Bug Fixes:**
- Fixed AdamW weight_decay documentation
- Corrected dequantize Python signature
- Resolved SDPA copy issues
- Fixed cross-entropy axis parameter

**Performance:**
- Faster fully depthwise-separable 1D convolutions
- Improved complex matmul operations
- Better contiguous gather for first-axis indices
- Enhanced row-reduce implementation

**⚠️ Breaking Change:**
- Default strict mode for module `update` and `update_modules`
</details>

<details>
<summary><strong>v0.29.0</strong> (August 29, 2024) - Click to expand</summary>

**New Features:**
- **Support for mxfp4 quantization (Metal, CPU)**
- NCCL backend for `mx.distributed` on CUDA
- Custom CUDA kernel support
- Enhanced SDPA with sinks
- Dynamic Slice/DynamicSliceUpdate for CUDA
- Mode parameter for quantization

**Bug Fixes:**
- Fixed unsigned dtype interpretation in reduce operations
- Corrected convolution gradients with groups
- Fixed lapack SVD
- Resolved power operation issues

**Performance:**
- Faster general unary operations
- Optimized GEMM-based fallback convolution kernels
- Improved CUDA copy performance
</details>

<details>
<summary><strong>v0.28.0</strong> (August 7, 2024) - Click to expand</summary>

**New Features:**
- Initial fused SDPA vector implementation for CUDA
- Convolution support in CUDA backend
- Tree flatten/unflatten destination-based support

**Bug Fixes:**
- Fixed CUDA segfault on exit
- Corrected graph key issues with concurrent context
- Fixed arctan2 gradients
- Corrected type promotion in Adam with bias correction

**Performance:**
- Optimized CUDA normalization layers
- Faster softmax operations
- Reduced compiled kernel overheads
- Improved vectorized kernels for CUDA
</details>

<details>
<summary><strong>v0.27.1</strong> (July 25, 2024) - Click to expand</summary>

**New Features:**
- **Initial PyPI release of CUDA backend**
- RoPE support for CUDA
- Affine quantize for CUDA
- CUDA graph implementation
- Scan kernel for CUDA
- Parameter deletion support from modules

**Bug Fixes:**
- Fixed resource leaks in matmul and graphs
- Corrected complex reduce and NaN propagation
- Fixed compilation with CUDA 11
- Resolved layernorm race conditions

**Performance:**
- Vectorized store/load in binary and contiguous elementwise operations
- Enhanced matmul performance with CUBLAS GEMM
</details>

### Breaking Changes Analysis

**One breaking change identified in v0.30.0:**

> Default strict mode for module `update` and `update_modules`

**Impact Assessment:**
- This affects code that directly uses `mlx.nn.Module.update()` or `update_modules()`
- Our implementations do NOT call MLX APIs directly—they use wrapper libraries
- The wrapper libraries (mlx-whisper, parakeet-mlx, etc.) handle this internally
- **Conclusion: No impact on our code**

The [MLX GitHub Issues](https://github.com/ml-explore/mlx/issues/1426) show deprecation warnings related to macOS 15 SDK (BNNSLayerParametersBroadcastMatMul), but these are handled internally by MLX and don't affect user code.

### Performance Improvements Worth Noting

| Feature | Version | Benefit |
|---------|---------|---------|
| mxfp4 quantization | 0.29.0 | 4-bit quantized models, memory savings |
| Neural Accelerator | 0.30.0 | 19-27% faster on M5 chips |
| Faster quant/dequant | 0.30.1 | Improved model loading times |
| QQMM tensor cores | 0.30.1 | Faster quantized matrix multiplication |
| nvfp4/mxfp8 | 0.30.3 | More quantization format options |

### macOS Version Requirements

| MLX Version | Minimum macOS | Notes |
|-------------|---------------|-------|
| 0.27.x - 0.29.x | macOS 14.0 | Standard support |
| 0.30.0+ | macOS 14.0 | Neural Accelerator requires macOS **26.2** for M5 |

**Note:** M5 Neural Accelerator features are only available on macOS 26.2+. On older macOS versions, MLX 0.30.x falls back gracefully to standard Metal acceleration.

---

## Phased Upgrade Strategy

### Phase 1: Stability (v0.27.1 → v0.29.4)

**Goal:** Get mxfp4 quantization and stability improvements without M5-specific features.

**What changes:**
- mxfp4 quantization format support (Metal, CPU)
- NCCL backend for distributed (not used by us)
- Pathlib.Path compatibility for save/load
- Multiple stability fixes (0.29.1-0.29.4)

**Why stop at v0.29.4?**
- Last version before Neural Accelerator changes
- Maximum stability before M5-specific code paths
- Tests compatibility with all downstream packages

**Bridge code changes required:**
- **None** — our implementations don't call MLX directly

**Commits:**
1. `chore(deps): bump mlx 0.27.1 → 0.29.4`
2. `test(benchmark): verify all 4 MLX implementations`

---

### Phase 2: M5 Optimization (v0.29.4 → v0.30.3)

**Goal:** Get Neural Accelerator support and latest performance features.

**What changes:**
- Neural Accelerator support on M5 (macOS 26.2+)
- RDMA over Thunderbolt (JACCL backend)
- Faster quantization/dequantization
- nvfp4/mxfp8 quantized operations

**Why this is the final phase:**
- Latest stable version
- Full M5 performance benefits
- All new quantization formats available

**Bridge code changes required:**
- **None** — performance benefits are automatic

**Commits:**
1. `chore(deps): bump mlx 0.29.4 → 0.30.3`
2. `test(benchmark): verify Neural Accelerator benefits (if on M5)`

---

## Implementation Details

### Phase 1 Step-by-Step

#### Step 1.1: Create Upgrade Branch
```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_MAIN
git checkout main
git pull origin main
git checkout -b chore/mlx-upgrade-phase1
```

#### Step 1.2: Preview Changes
```bash
# See what would change
uv lock --upgrade-package mlx --upgrade-package mlx-metal --dry-run
```

**Expected output should show:**
- `mlx: 0.27.1 → 0.29.4`
- `mlx-metal: 0.27.1 → 0.29.4`

#### Step 1.3: Perform Atomic Upgrade
```bash
# Update BOTH packages together (critical!)
uv lock --upgrade-package mlx --upgrade-package mlx-metal

# Sync environment
uv sync
```

#### Step 1.4: Verify Installation
```bash
.venv/bin/python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
# Expected: MLX version: 0.29.4
```

#### Step 1.5: Test All MLX Implementations
```bash
# Test each implementation individually
.venv/bin/mac-whisper-speedtest -b -i MLXWhisperImplementation -n 2

# Or test all implementations (batch mode uses tests/jfk.wav by default)
.venv/bin/mac-whisper-speedtest -b
```

**Focus on these 4 implementations:**
- [ ] MLXWhisperImplementation
- [ ] WhisperMPSImplementation
- [ ] LightningWhisperMLXImplementation
- [ ] ParakeetMLXImplementation

#### Step 1.6: Commit
```bash
git add uv.lock
git commit -m "$(cat <<'EOF'
chore(deps): bump mlx 0.27.1 → 0.29.4

Phase 1 of MLX upgrade (v0.27.1 → v0.30.3)

Changes:
- mlx: 0.27.1 → 0.29.4
- mlx-metal: 0.27.1 → 0.29.4
- Adds mxfp4 quantization format support
- Multiple stability improvements

Tested:
- MLXWhisperImplementation: ✓
- WhisperMPSImplementation: ✓
- LightningWhisperMLXImplementation: ✓
- ParakeetMLXImplementation: ✓

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Phase 2 Step-by-Step

#### Step 2.1: Upgrade to Latest
```bash
uv lock --upgrade-package mlx --upgrade-package mlx-metal
uv sync
```

#### Step 2.2: Verify Version
```bash
.venv/bin/python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
# Expected: MLX version: 0.30.3
```

#### Step 2.3: Test All Implementations
```bash
.venv/bin/mac-whisper-speedtest -b
```

#### Step 2.4: Benchmark Comparison (Optional)
```bash
# Record baseline timing before commit
.venv/bin/mac-whisper-speedtest -b > benchmark_mlx_0.30.3.txt

# Compare with previous results if available
diff benchmark_mlx_0.29.4.txt benchmark_mlx_0.30.3.txt
```

#### Step 2.5: Commit
```bash
git add uv.lock
git commit -m "$(cat <<'EOF'
chore(deps): bump mlx 0.29.4 → 0.30.3

Phase 2 (final) of MLX upgrade (v0.27.1 → v0.30.3)

Changes:
- mlx: 0.29.4 → 0.30.3
- mlx-metal: 0.29.4 → 0.30.3
- Neural Accelerator support on M5 (macOS 26.2+)
- RDMA over Thunderbolt (JACCL backend)
- nvfp4/mxfp8 quantized operations

Performance Notes:
- 19-27% improvement expected on M5 chips with macOS 26.2+
- Graceful fallback on older hardware/OS

Tested:
- All 4 MLX-based implementations verified
- Benchmark times comparable or improved

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Coordinated Dependencies

### Package Upgrade Matrix

When upgrading MLX, consider upgrading these related packages in the same session:

| Package | Current | Latest | Upgrade With MLX? |
|---------|---------|--------|-------------------|
| `mlx` | 0.27.1 | 0.30.3 | **YES (primary)** |
| `mlx-metal` | 0.27.1 | 0.30.3 | **YES (mandatory, same version)** |
| `mlx-whisper` | 0.4.2 | 0.4.3 | Recommended (1 patch) |
| `parakeet-mlx` | 0.3.5 | 0.5.0 | Optional (2 minor) |
| `lightning-whisper-mlx` | 0.0.10 | Check | Optional |
| `whisper-mps` | 0.0.7 | 0.0.10 | Optional (3 patches) |

### Upgrade Order Recommendation

```
1. mlx + mlx-metal (MUST be atomic, same version)
   └── These are the foundation

2. mlx-whisper (optional, but low-risk)
   └── Our primary MLX implementation uses this

3. Other MLX-dependent packages (per audit plan)
   └── parakeet-mlx, whisper-mps, etc.
```

### Version Constraint Compatibility

Check that downstream packages accept MLX 0.30.x:

```bash
# Check mlx-whisper constraints
pip show mlx-whisper | grep -i require

# Check parakeet-mlx constraints
pip show parakeet-mlx | grep -i require
```

Most MLX-based packages use loose constraints like `mlx>=0.10.0` to allow major upgrades.

---

## Git Workflow & Commit Strategy

### Branch Naming

```
chore/mlx-upgrade-phase1   # Phase 1: v0.27.1 → v0.29.4
chore/mlx-upgrade-phase2   # Phase 2: v0.29.4 → v0.30.3
```

**Or single branch:**
```
chore/mlx-upgrade-v0.30
```

### Commit Message Format

```
chore(deps): bump mlx X.Y.Z → A.B.C

Phase N of MLX upgrade (v0.27.1 → v0.30.3)

Changes:
- List key changes from release notes

Tested:
- List implementations tested

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Coordinating with Other Upgrades

**Recommended order if doing multiple upgrades:**

```
1. MLX Core (foundation)
   └── Affects: mlx-whisper, whisper-mps, lightning-whisper-mlx, parakeet-mlx

2. FluidAudio Swift Bridge
   └── Independent of MLX

3. WhisperKit Swift Bridge
   └── Independent of MLX

4. Python package upgrades (faster-whisper, etc.)
   └── Independent of MLX
```

**Why MLX first?** It's the foundation for 4 implementations. If MLX breaks something, it's better to know before adding more changes.

---

## Risk Mitigation

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Downstream package incompatibility | Low | Medium | Test each implementation after upgrade |
| Performance regression | Very Low | Low | Benchmark before/after |
| API changes in MLX | Very Low | High | Our code doesn't call MLX directly |
| mlx-metal version mismatch | Medium | High | **Always upgrade both together** |
| macOS version incompatibility | Low | Medium | MLX falls back gracefully |

### Critical Warning: Version Synchronization

```bash
# NEVER do this:
uv lock --upgrade-package mlx
# This might leave mlx-metal at old version!

# ALWAYS do this:
uv lock --upgrade-package mlx --upgrade-package mlx-metal
```

### Rollback Procedures

**If Phase 1 fails:**
```bash
git checkout main -- uv.lock
uv sync
```

**If Phase 2 fails (but Phase 1 worked):**
```bash
# Pin to Phase 1 version
uv add mlx==0.29.4 mlx-metal==0.29.4
uv sync
```

**Nuclear option:**
```bash
git checkout main
git branch -D chore/mlx-upgrade-phase1
```

### Known Compatibility Notes

From [GitHub Issues](https://github.com/ml-explore/mlx/issues/2197):
- parakeet-mlx may crash on macOS 14 with MLX 0.25.2+ due to `newResidencySetWithDescriptor` API
- This project targets macOS 14.0+, but full features require macOS 15.0+
- M5 Neural Accelerator requires macOS 26.2+

---

## Testing Checklist

### Pre-Upgrade Baseline

- [ ] Record current benchmark: `.venv/bin/mac-whisper-speedtest -b > baseline_mlx_0.27.1.txt`
- [ ] Note transcription text output for comparison
- [ ] Verify all 4 MLX implementations work

### Phase 1 Verification

- [ ] `uv sync` completes without errors
- [ ] `.venv/bin/python -c "import mlx; print(mlx.__version__)"` shows 0.29.x
- [ ] MLXWhisperImplementation transcription works
- [ ] WhisperMPSImplementation transcription works
- [ ] LightningWhisperMLXImplementation transcription works
- [ ] ParakeetMLXImplementation transcription works
- [ ] No new deprecation warnings
- [ ] Benchmark time within 20% of baseline (or faster)

### Phase 2 Verification

- [ ] All Phase 1 checks pass
- [ ] `.venv/bin/python -c "import mlx; print(mlx.__version__)"` shows 0.30.3
- [ ] No runtime errors related to Neural Accelerator (graceful fallback expected on non-M5)
- [ ] Full benchmark suite completes

### Post-Upgrade Validation

- [ ] Compare benchmark times to baseline
- [ ] Compare transcription text (should be identical)
- [ ] Run `pytest tests/ -v` to ensure test suite passes
- [ ] Document any performance changes

---

## Summary

### Quick Reference: Upgrade Commands

```bash
# === PHASE 1 ===
git checkout -b chore/mlx-upgrade-phase1
uv lock --upgrade-package mlx --upgrade-package mlx-metal
# Verify: mlx should be 0.29.x
uv sync
.venv/bin/python -c "import mlx; print(mlx.__version__)"
.venv/bin/mac-whisper-speedtest -b
git add uv.lock && git commit -m "chore(deps): mlx 0.27.1 → 0.29.4"

# === PHASE 2 ===
uv lock --upgrade-package mlx --upgrade-package mlx-metal
# Verify: mlx should be 0.30.3
uv sync
.venv/bin/python -c "import mlx; print(mlx.__version__)"
.venv/bin/mac-whisper-speedtest -b
git add uv.lock && git commit -m "chore(deps): mlx 0.29.4 → 0.30.3"
```

### Key Takeaways

1. **This is a performance upgrade** — No API changes, no code modifications needed
2. **mlx and mlx-metal MUST be upgraded together** — Same version always
3. **4 implementations affected** — MLX, whisper-mps, lightning-whisper, parakeet
4. **M5 benefits require macOS 26.2+** — Graceful fallback on older systems
5. **Test all MLX implementations after each phase** — Even though APIs are stable

### Comparison: MLX vs Other Upgrades

| Aspect | MLX | FluidAudio | WhisperKit |
|--------|-----|------------|------------|
| Version gap | 3 minor | 9 minor | 2 minor |
| Breaking changes | None | API enum | Struct→Class |
| Code changes required | None | Required | None |
| Risk level | Low-Medium | High | Medium |
| Primary benefit | Performance | Features | Features |
| Implementations affected | 4 | 1 | 1 |

---

## Sources

- [MLX GitHub Releases](https://github.com/ml-explore/mlx/releases)
- [MLX PyPI](https://pypi.org/project/mlx/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Apple Machine Learning Research - M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [MLX GitHub Issues - macOS Compatibility](https://github.com/ml-explore/mlx/issues/2197)
