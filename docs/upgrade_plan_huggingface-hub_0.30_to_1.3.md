# Plan: Upgrade huggingface-hub 0.30.2 → 1.3.1

**Created:** 2026-01-26
**Completed:** 2026-01-26
**Status:** ✅ DONE

---

## Background: CLI vs Library (Important Distinction)

**Your question about using `uvx hf` instead of the installed package has an important nuance:**

```
huggingface-hub package
├── Python Library API (imported in code)
│   ├── snapshot_download()  ← used by mlx.py
│   ├── hf_hub_download()    ← used by parakeet-mlx, transformers, etc.
│   └── HfApi class          ← not used in this project
│
└── CLI Tool (`hf` command)
    ├── hf download
    ├── hf cache ls
    └── hf env
```

| Use Case | `uvx hf` works? | Installed package needed? |
|----------|-----------------|---------------------------|
| CLI commands (`hf download`, `hf cache ls`) | Yes | No |
| Python imports (`from huggingface_hub import...`) | No | Yes |
| Used by transformers, mlx-whisper, faster-whisper | No | Yes |

**Bottom line:** `uvx hf` is perfect for CLI operations and pre-upgrade testing, but the library dependency in `pyproject.toml` is unavoidable because:
1. `mlx.py` directly imports `snapshot_download()`
2. 5+ other packages (transformers, faster-whisper, mlx-whisper, etc.) depend on it internally

---

## Codebase Analysis

### Direct huggingface_hub Usage (1 file)

| File | Import | Function |
|------|--------|----------|
| `implementations/mlx.py` | `from huggingface_hub import snapshot_download` | Downloads MLX whisper models |

### Indirect Usage (via dependent libraries)

| Implementation | Library | HF Function Used Internally |
|----------------|---------|----------------------------|
| ParakeetMLX | parakeet_mlx | `hf_hub_download()` |
| InsanelyFast | transformers | `snapshot_download()` |
| FasterWhisper | faster-whisper | `hf_hub_download()` |
| LightningMLX | lightning-whisper-mlx | `hf_hub_download()` |

### Not Using huggingface_hub

| Implementation | Download Method |
|----------------|-----------------|
| WhisperCppCoreML | Direct HTTP via `requests.get()` |
| WhisperMPS | Azure CDN (OpenAI's CDN) |
| WhisperKit | Swift HubApi (not Python) |
| FluidAudio | Swift (internal) |

---

## Upgrade Plan

### Phase 1: Pre-Upgrade Exploration with `uvx hf`

```bash
# 1. Check what version uvx runs (should be latest ~1.3.x)
uvx hf version

# 2. Explore new CLI features
uvx hf cache ls                    # List all cached models
uvx hf env                         # Show cache paths

# 3. Test download behavior with 1.3.x
uvx hf download mlx-community/whisper-tiny-mlx --dry-run

# 4. Verify cache is shared (same location as project uses)
uvx hf env | grep HF_HUB_CACHE
# Should show: ~/.cache/huggingface/hub
```

### Phase 2: Capture Baseline

```bash
# Record current version
.venv/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"
# → 0.30.2

# Run baseline tests
pytest tests/test_model_params.py -v

# Run affected implementations (save output)
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small \
  -i "MLXWhisperImplementation,ParakeetMLXImplementation,LightningWhisperMLXImplementation,InsanelyFastWhisperImplementation,FasterWhisperImplementation"
```

### Phase 3: Upgrade

**File: `pyproject.toml`**

```toml
# Line 24 - Before
"huggingface-hub>=0.20.0",
# Line 24 - After
"huggingface-hub>=1.3.0,<2.0",

# Line ~26 - Before
"hf-xet>=1.0.3",
# Line ~26 - After
"hf-xet>=1.2.0",
```

```bash
# Resolve and sync
uv sync

# Verify both packages
.venv/bin/python -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')"
# → 1.3.1 (or higher)

.venv/bin/python -c "import hf_xet; print(f'hf_xet: {hf_xet.__version__}')"
# → 1.2.0 (or higher)
```

### Phase 4: Test All Affected Implementations

```bash
# Run each affected implementation
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "MLXWhisperImplementation"
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "ParakeetMLXImplementation"
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "LightningWhisperMLXImplementation"
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "InsanelyFastWhisperImplementation"
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small -i "FasterWhisperImplementation"

# Run full test suite
pytest tests/ -v
```

### Phase 5: Update Documentation

Update `docs/feature_plan_version_audit.md`:
- Mark huggingface-hub as DONE
- Mark hf-xet as DONE
- Record the completion date

---

## Best Practice Recommendation: When to Use `uvx hf`

**Add to project workflow (CLAUDE.md or docs):**

```bash
# Pre-download models before benchmarking (useful for CI/agents)
uvx hf download mlx-community/whisper-small-mlx-4bit
uvx hf download openai/whisper-small

# Inspect cache
uvx hf cache ls

# Verify model integrity (new in 1.x)
uvx hf cache verify mlx-community/whisper-small-mlx-4bit
```

**Why this works:** The cache directory (`~/.cache/huggingface/hub/`) is shared between `uvx hf` and your project's `.venv`. Models downloaded via `uvx hf` are immediately available to your Python code.

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Breaking API changes | LOW | Core functions (`snapshot_download`, `hf_hub_download`) unchanged |
| Cache format changes | NONE | Cache structure unchanged in 1.x |
| Dependent package compatibility | LOW | transformers, faster-whisper, mlx-whisper all support 1.x |

---

## Rollback Plan

```bash
# Revert pyproject.toml to: "huggingface-hub>=0.20.0", "hf-xet>=1.0.3"
uv sync
# Verify: should return to 0.30.2
```

---

## Files to Modify

1. **`pyproject.toml`** - Update version constraints:
   - `huggingface-hub>=1.3.0,<2.0` (line 24)
   - `hf-xet>=1.2.0` (line ~26)
2. **`docs/feature_plan_version_audit.md`** - Mark both as completed

## Verification Commands

```bash
# Version check
.venv/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"

# Test suite
pytest tests/ -v

# Batch benchmark (all affected implementations)
.venv/bin/mac-whisper-speedtest --batch -n 1 -m small
```
