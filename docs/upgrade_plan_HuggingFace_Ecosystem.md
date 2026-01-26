# Plan: Upgrade HuggingFace Ecosystem Packages

**Created:** 2026-01-26
**Completed:** 2026-01-26
**Status:** ✅ DONE
**Risk Level:** MEDIUM (coordinated multi-package upgrade)

> **Note:** Upgrading huggingface-hub to 1.3.x triggered a cascading upgrade of transformers from 4.51.3 to 5.0.0. This required a code fix in `insanely.py` to handle bitsandbytes removal on macOS and the deprecated `torch_dtype` parameter.

---

## Executive Summary

Four HuggingFace ecosystem packages should be upgraded together due to tight coupling:

| Package | Current | Target | Gap |
|---------|---------|--------|-----|
| `transformers` | 4.51.3 | 4.57.5 | 6 patch |
| `accelerate` | 1.6.0 | 1.12.0 | 6 minor |
| `safetensors` | 0.5.3 | 0.7.0 | 2 minor |
| `tokenizers` | 0.21.1 | 0.22.2 | 1 minor |

**Key insight:** These are ALL **transitive dependencies** — none are declared in `pyproject.toml`. They're pulled in by `insanely-fast-whisper` and `faster-whisper`.

---

## Dependency Graph

```
pyproject.toml
├── insanely-fast-whisper >= 0.0.15
│   ├── transformers ──┬── tokenizers
│   │                  ├── safetensors
│   │                  └── huggingface-hub
│   │
│   └── accelerate ────┬── safetensors
│                      └── huggingface-hub
│
└── faster-whisper >= 1.1.1
    └── tokenizers
```

**Only ONE implementation is affected:** `InsanelyFastWhisperImplementation`
- Directly imports: `transformers.pipelines.pipeline`, `transformers.BitsAndBytesConfig`
- Indirectly uses: accelerate, safetensors, tokenizers

---

## Codebase Usage

### Direct Imports (insanely.py only)

```python
# Line 81
from transformers.pipelines import pipeline

# Line 83 (with fallback)
from transformers import BitsAndBytesConfig

# Lines 89-95 (with fallback)
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.utils import is_flash_attn_2_available
```

### No Direct Imports

| Package | Direct imports? | Used by |
|---------|-----------------|---------|
| `transformers` | YES (insanely.py) | InsanelyFastWhisperImplementation |
| `accelerate` | NO | Used internally by insanely-fast-whisper |
| `safetensors` | NO | Used internally by transformers/accelerate |
| `tokenizers` | NO | Used internally by transformers, faster-whisper |

---

## Upgrade Strategy

### Option A: Add Direct Constraints (Recommended)

Add all 4 packages to `pyproject.toml` for explicit version control:

```toml
dependencies = [
    # ... existing deps ...

    # HuggingFace ecosystem (coordinated versions)
    "transformers>=4.57.0,<5.0",
    "accelerate>=1.12.0,<2.0",
    "safetensors>=0.7.0,<1.0",
    "tokenizers>=0.22.0,<1.0",
]
```

**Pros:**
- Explicit version control
- Documents the dependency clearly
- Prevents accidental downgrades

**Cons:**
- More deps to maintain
- May conflict with future insanely-fast-whisper updates (unlikely)

### Option B: Targeted uv Upgrade

Use uv to upgrade specific packages:

```bash
uv lock --upgrade-package transformers \
        --upgrade-package accelerate \
        --upgrade-package safetensors \
        --upgrade-package tokenizers
uv sync
```

**Pros:**
- Minimal pyproject.toml changes
- Let resolver handle compatibility

**Cons:**
- Less explicit
- May not upgrade if constrained by other packages

---

## Upgrade Plan

### Phase 1: Pre-Upgrade Baseline

```bash
# Record current versions
.venv/bin/python -c "
import transformers, accelerate, safetensors, tokenizers
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'safetensors: {safetensors.__version__}')
print(f'tokenizers: {tokenizers.__version__}')
"
# Expected:
# transformers: 4.51.3
# accelerate: 1.6.0
# safetensors: 0.5.3
# tokenizers: 0.21.1

# Test InsanelyFastWhisper (the only affected implementation)
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "InsanelyFastWhisperImplementation"
```

### Phase 2: Dry Run

```bash
# Check what would change
uv lock --upgrade-package transformers \
        --upgrade-package accelerate \
        --upgrade-package safetensors \
        --upgrade-package tokenizers \
        --dry-run
```

Review the output for:
- Any unexpected cascading changes
- Version conflicts
- Packages being removed or added

### Phase 3: Upgrade

**Option A: Add direct constraints**

Edit `pyproject.toml` to add after existing dependencies:

```toml
    # HuggingFace ecosystem (coordinated versions)
    "transformers>=4.57.0,<5.0",
    "accelerate>=1.12.0,<2.0",
    "safetensors>=0.7.0,<1.0",
    "tokenizers>=0.22.0,<1.0",
```

Then:
```bash
uv sync
```

**Option B: Use uv upgrade**

```bash
uv lock --upgrade-package transformers \
        --upgrade-package accelerate \
        --upgrade-package safetensors \
        --upgrade-package tokenizers
uv sync
```

### Phase 4: Verify Versions

```bash
.venv/bin/python -c "
import transformers, accelerate, safetensors, tokenizers
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'safetensors: {safetensors.__version__}')
print(f'tokenizers: {tokenizers.__version__}')
"
# Expected:
# transformers: 4.57.x
# accelerate: 1.12.x
# safetensors: 0.7.x
# tokenizers: 0.22.x
```

### Phase 5: Test InsanelyFastWhisper

This is the **only implementation** that uses these packages:

```bash
# Basic test
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "InsanelyFastWhisperImplementation"

# Test with different model sizes
.venv/bin/mac-whisper-speedtest --batch -n 1 -m tiny -i "InsanelyFastWhisperImplementation"
.venv/bin/mac-whisper-speedtest --batch -n 1 -m large-v3-turbo -i "InsanelyFastWhisperImplementation"
```

### Phase 6: Test FasterWhisper (uses tokenizers)

Though it only uses tokenizers transitively, verify it still works:

```bash
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "FasterWhisperImplementation"
```

### Phase 7: Run Full Test Suite

```bash
pytest tests/ -v
```

### Phase 8: Update Documentation

Update `docs/feature_plan_version_audit.md`:
- Mark transformers as DONE
- Mark accelerate as DONE
- Mark safetensors as DONE
- Mark tokenizers as DONE
- Record completion date

---

## Coordination with Other Upgrades

### With huggingface-hub Upgrade

If upgrading `huggingface-hub` 0.30.2 → 1.3.1 at the same time:
- `transformers` 4.57.x supports huggingface-hub 1.x
- `accelerate` 1.12.x supports huggingface-hub 1.x
- **Recommendation:** Do huggingface-hub FIRST, then these packages

### With torch Upgrade

If upgrading `torch` 2.6.0 → 2.9.1 at the same time:
- `transformers` 4.57.x supports torch 2.9.x
- `accelerate` 1.12.x supports torch 2.9.x
- Can be done together or separately
- InsanelyFastWhisper uses both, test thoroughly if upgrading together

### Suggested Order

1. `huggingface-hub` (foundation)
2. HuggingFace ecosystem (this plan)
3. `torch` / `torchaudio` (last)

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| transformers API changes | LOW | Patch version, backward compatible |
| accelerate API changes | LOW | Well-maintained, stable APIs |
| safetensors format changes | VERY LOW | Binary format is stable |
| tokenizers API changes | VERY LOW | Rust core, stable API |
| Cascading dep conflicts | MEDIUM | Run dry-run first, verify versions |

---

## Rollback Plan

**If using Option A:**
```bash
# Remove the 4 added lines from pyproject.toml
uv sync
# Should restore previous transitive versions
```

**If using Option B:**
```bash
uv lock  # Re-lock without upgrade flags
uv sync
```

---

## Files to Modify

### Option A (Recommended)
1. **`pyproject.toml`** - Add 4 dependency constraints
2. **`docs/feature_plan_version_audit.md`** - Mark all 4 as completed

### Option B
1. **`uv.lock`** - Auto-updated
2. **`docs/feature_plan_version_audit.md`** - Mark all 4 as completed

---

## Verification Commands

```bash
# Version check (all 4 packages)
.venv/bin/python -c "
import transformers, accelerate, safetensors, tokenizers
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'safetensors: {safetensors.__version__}')
print(f'tokenizers: {tokenizers.__version__}')
"

# Primary affected implementation
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "InsanelyFastWhisperImplementation"

# Secondary affected implementation (tokenizers only)
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "FasterWhisperImplementation"

# Full test suite
pytest tests/ -v
```
