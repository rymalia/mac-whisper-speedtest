# Model Details: MLXWhisperImplementation

This document traces the model download, caching, and loading behavior for the `MLXWhisperImplementation` class, which uses the `mlx-whisper` library for Apple Silicon-optimized Whisper inference.

---

## File Reference Legend

| Tag | Meaning |
|-----|---------|
| [PROJECT] | Files in this repository |
| [LIBRARY] | Files in `.venv/lib/python3.12/site-packages/mlx_whisper/` |

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| **What HuggingFace repo is used?** | Depends on model size - see Model Mapping below | `mlx.py:49-57` [PROJECT] |
| **Where are models cached?** | Project-local `models/` directory (NOT standard HF cache) | `mlx.py:76`, `utils.py:28-38` [PROJECT] |
| **Who handles the download?** | `huggingface_hub.snapshot_download()` | `mlx.py:81-84` [PROJECT] |
| **Does it prefer quantized models?** | Yes - tries 4-bit/8-bit first, falls back to non-quantized | `mlx.py:49-68` [PROJECT] |
| **Missing symlink behavior?** | Auto-recreates symlink to existing blob (no re-download) | Empirical test |
| **Corrupted blob behavior?** | HF doesn't detect; MLX load fails; falls back to non-quantized model | Empirical test |
| **Is corrupted file cleaned up?** | **NO** - remains in cache, causing repeated fallbacks | Empirical test |
| **Timeout issues?** | None observed - both small and large models download within reasonable time | Empirical test |

---

## Benchmark Execution Flow

### Command: `test_benchmark2.py small 1 MLXWhisperImplementation`

```
1. test_benchmark2.py: Parse args → model_name="small", num_runs=1
2. test_benchmark2.py: Load audio from tests/jfk.wav
3. test_benchmark2.py: Create BenchmarkConfig(model_name="small", implementations=[MLXWhisperImplementation])
4. benchmark.py: run_benchmark() → instantiate MLXWhisperImplementation()
5. benchmark.py: Call implementation.load_model("small")
   │
   ├─► mlx.py:44: self.model_name = "small"
   ├─► mlx.py:71: model_map["small"] = "mlx-community/whisper-small-mlx-4bit"
   ├─► mlx.py:76: models_dir = get_models_dir() → "{project_root}/models"
   ├─► mlx.py:81-84: snapshot_download(
   │       repo_id="mlx-community/whisper-small-mlx-4bit",
   │       cache_dir="{project_root}/models"
   │   )
   │   → Downloads to: models/models--mlx-community--whisper-small-mlx-4bit/
   │
   ├─► mlx.py:88: load_model(model_path) [LIBRARY]
   │   └─► load_models.py:22-24: Open config.json, load weights.npz
   │
   └─► mlx.py:90: Model loaded successfully

6. benchmark.py: Time transcription → implementation.transcribe(audio_data)
   │
   ├─► mlx.py:136: from mlx_whisper import transcribe
   └─► mlx.py:140-146: transcribe(audio, path_or_hf_repo=model_path, ...)
       └─► transcribe.py:147: ModelHolder.get_model() → reuses loaded model

7. benchmark.py: Return BenchmarkResult with timing and transcription
```

### Command: `test_benchmark2.py large 1 MLXWhisperImplementation`

Same flow, except:
- `model_map["large"]` = `"mlx-community/whisper-large-v3-turbo"` (NOT large-v3)
- Downloads to: `models/models--mlx-community--whisper-large-v3-turbo/`
- Model is fp16 (not quantized), despite log saying "quantized"

---

## Summary Table

| Requested Model | Actual HuggingFace Repo | Quantization | Cache Location | Size |
|-----------------|-------------------------|--------------|----------------|------|
| `tiny` | `mlx-community/whisper-tiny-mlx` | None | `models/models--mlx-community--whisper-tiny-mlx/` | ~75MB |
| `base` | `mlx-community/whisper-base-mlx` | None | `models/models--mlx-community--whisper-base-mlx/` | ~145MB |
| `small` | `mlx-community/whisper-small-mlx-4bit` | 4-bit | `models/models--mlx-community--whisper-small-mlx-4bit/` | ~187MB |
| `medium` | `mlx-community/whisper-medium-mlx-8bit` | 8-bit | `models/models--mlx-community--whisper-medium-mlx-8bit/` | ~750MB |
| `large` | `mlx-community/whisper-large-v3-turbo` | None (fp16) | `models/models--mlx-community--whisper-large-v3-turbo/` | ~1.5GB |
| `large-v2` | `mlx-community/whisper-large-v2-mlx-4bit` | 4-bit | `models/models--mlx-community--whisper-large-v2-mlx-4bit/` | ~800MB |
| `large-v3` | `mlx-community/whisper-large-v3-mlx-8bit` | 8-bit | `models/models--mlx-community--whisper-large-v3-mlx-8bit/` | ~1.5GB |

---

## Model Mapping Reference

```python
# Primary model map (prefers quantized) - mlx.py:49-57
model_map = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx-4bit",      # ← 4-bit quantized
    "medium": "mlx-community/whisper-medium-mlx-8bit",    # ← 8-bit quantized
    "large": "mlx-community/whisper-large-v3-turbo",      # ← Turbo variant, NOT large-v3
    "large-v2": "mlx-community/whisper-large-v2-mlx-4bit",
    "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
}

# Fallback model map (non-quantized) - mlx.py:60-68
fallback_model_map = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",           # ← Non-quantized fallback
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-turbo",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
```

---

## Notes

### Quantization Strategy
MLXWhisperImplementation prefers quantized models for better Apple Silicon performance:
- `small` → 4-bit (saves ~60% memory vs fp16)
- `medium` → 8-bit (saves ~50% memory vs fp16)
- `large-v2` → 4-bit
- `large-v3` → 8-bit

If the quantized model fails to load, it automatically falls back to the non-quantized version.

### Cache Location Issue
Unlike other implementations that use the standard HuggingFace cache (`~/.cache/huggingface/hub/`), MLXWhisperImplementation uses a **project-local** `models/` directory. This means:
- Models are NOT shared across projects
- Each project downloads its own copies
- Disk space is wasted with duplicate models

### "Large" Model Confusion
The `"large"` model name maps to `whisper-large-v3-turbo`, which is a distilled/faster variant, NOT the full `whisper-large-v3`. Users expecting the full large model should explicitly use `"large-v3"`.

---

## Key Source Files

| File | Role |
|------|------|
| `src/mac_whisper_speedtest/implementations/mlx.py` [PROJECT] | Implementation wrapper |
| `src/mac_whisper_speedtest/utils.py` [PROJECT] | `get_models_dir()` returns project-local cache path |
| `.venv/lib/python3.12/site-packages/mlx_whisper/load_models.py` [LIBRARY] | Model loading logic |
| `.venv/lib/python3.12/site-packages/mlx_whisper/transcribe.py` [LIBRARY] | Transcription with ModelHolder caching |

---

## Empirical Test Results

**Test Date:** 2026-01-11

### Small Model Tests

#### Test 1: Fresh Download (No Local Model)

**Pre-condition:**
```bash
ls -la models/
# Only .DS_Store present - no mlx-whisper models
```

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 MLXWhisperImplementation
```

**Terminal Output:**
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
...
[2026-01-11 16:51:04] [info] Downloading quantized model mlx-community/whisper-small-mlx-4bit to /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
Fetching 4 files: 100%|██████████| 4/4 [00:01<00:00,  2.39it/s]
[2026-01-11 16:51:05] [info] Loading quantized model from .../snapshots/f1da4c67f2ee8b6e763b974e149aa65d5b7658b7
[2026-01-11 16:51:06] [info] Successfully loaded quantized MLX Whisper model: mlx-community/whisper-small-mlx-4bit
[2026-01-11 16:51:06] [info] Run 1 completed in 0.7604 seconds

=== Benchmark Summary for 'small' model ===
mlx-whisper            0.7604          model=whisper-small-mlx-4bit, quantization=4bit
    "And so my fellow Americans, ask not what your country can do for you..."
```

**Cache after download:**
```bash
du -sh models/models--mlx-community--whisper-small-mlx-4bit/
# 187M

ls -la models/models--mlx-community--whisper-small-mlx-4bit/snapshots/*/
# .gitattributes -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
# config.json -> ../../blobs/0d7f9864df4795ae1b6841c0b283ed8398ad5f94
# README.md -> ../../blobs/abe673221f02f5d3f5f7be6cf02c6567b1b647bb
# weights.npz -> ../../blobs/ca6659298fe7550468ff0fc49dea7442615d9a53d1ce087aaded1b7627451998
```

**Key metrics:**
- Download time: ~2 seconds
- Model size: 187MB (4-bit quantized)
- Transcription time: 0.76 seconds

---

#### Test 2: Cached Behavior (Model Already Downloaded)

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 MLXWhisperImplementation
```

**Terminal Output:**
```
[2026-01-11 16:54:08] [info] Downloading quantized model mlx-community/whisper-small-mlx-4bit to .../models
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 29228.60it/s]   # ← Instant verification
[2026-01-11 16:54:09] [info] Loading quantized model from .../snapshots/...
[2026-01-11 16:54:10] [info] Run 1 completed in 0.7037 seconds
```

**Key observation:** The `29228 it/s` rate indicates cache verification only - no actual download.

---

#### Test 3: Missing Symlink (Partial Download Simulation)

**Setup:** Renamed `weights.npz` symlink to `weights__OFF__.npz`

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 MLXWhisperImplementation
```

**Terminal Output:**
```
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 24.81it/s]   # ← Slightly slower
[2026-01-11 16:57:10] [info] Loading quantized model from .../snapshots/...
[2026-01-11 16:57:11] [info] Successfully loaded quantized MLX Whisper model
```

**Result:** HuggingFace detected the missing symlink and recreated it pointing to the existing blob. **No re-download of the 187MB weights was needed.**

```bash
ls -la .../snapshots/*/
# weights__OFF__.npz -> ../../blobs/ca6659...  (our renamed file)
# weights.npz -> ../../blobs/ca6659...         (newly created symlink to SAME blob)
```

**Conclusion:** Excellent partial download recovery - blobs are preserved, only symlinks recreated.

---

#### Test 4: Corrupted Blob (Truncated Weights File)

**Setup:** Truncated the weights blob from 187MB to 1MB:
```bash
truncate -s 1000000 models/.../blobs/ca6659298fe7550468ff0fc49dea7442615d9a53d1ce087aaded1b7627451998
```

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 MLXWhisperImplementation
```

**Terminal Output:**
```
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 16743.73it/s]   # ← HF thinks it's fine!
[2026-01-11 16:58:57] [info] Loading quantized model from .../snapshots/...
[2026-01-11 16:58:57] [warning] Failed to load quantized model mlx-community/whisper-small-mlx-4bit:
    [load_npz] Input must be a zip file or a file-like object that can be opened with zipfile.ZipFile
[2026-01-11 16:58:57] [info] Falling back to non-quantized model: mlx-community/whisper-small-mlx
Fetching 4 files: 100%|██████████| 4/4 [00:33<00:00,  8.47s/it]   # ← Downloads fallback model
[2026-01-11 16:59:31] [info] Successfully loaded fallback model: mlx-community/whisper-small-mlx
[2026-01-11 16:59:32] [info] Run 1 completed in 1.1763 seconds

=== Benchmark Summary for 'small' model ===
mlx-whisper            1.1763          model=whisper-small-mlx, quantization=none   # ← Using fallback!
```

**Cache state after:**
```bash
du -sh models/models--mlx-community--whisper-small-mlx*/
# 996K    models/models--mlx-community--whisper-small-mlx-4bit/   # ← Corrupted, NOT cleaned up
# 459M    models/models--mlx-community--whisper-small-mlx/        # ← Fallback model downloaded
```

**Critical findings:**
1. HuggingFace does NOT verify file integrity (no SHA256 hash check)
2. Corruption detected only when MLX tries to load the file
3. **Silent fallback**: User gets slower non-quantized model without clear warning
4. **No cleanup**: Corrupted file persists, causing repeated fallback behavior
5. Transcription time degraded: 1.18s vs 0.70s (68% slower)

---

### Large Model Tests

#### Test 1: Fresh Download

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 MLXWhisperImplementation
```

**Terminal Output:**
```
[2026-01-11 16:51:38] [info] Downloading quantized model mlx-community/whisper-large-v3-turbo to .../models
Fetching 4 files: 100%|██████████| 4/4 [01:53<00:00, 28.30s/it]   # ← 1 min 53 sec download
[2026-01-11 16:53:32] [info] Loading quantized model from .../snapshots/beea265c324f07ba1e347f3c8a97aec454056a86
[2026-01-11 16:53:33] [info] Successfully loaded quantized MLX Whisper model: mlx-community/whisper-large-v3-turbo
[2026-01-11 16:53:37] [info] Run 1 completed in 3.7708 seconds

=== Benchmark Summary for 'large' model ===
mlx-whisper            3.7708          model=whisper-large-v3-turbo, quantization=none
```

**Cache after download:**
```bash
du -sh models/models--mlx-community--whisper-large-v3-turbo/
# 1.5G

ls -la .../snapshots/*/
# weights.safetensors -> ../../blobs/951ed3fc...   # ← Uses safetensors format
```

**Key metrics:**
- Download time: ~113 seconds (1:53)
- Model size: 1.5GB (fp16, NOT quantized despite log message)
- Transcription time: 3.77 seconds

---

#### Test 2: Cached Behavior

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 MLXWhisperImplementation
```

**Terminal Output:**
```
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 28387.84it/s]   # ← Instant verification
[2026-01-11 16:54:12] [info] Run 1 completed in 2.7551 seconds
```

**Key observation:** Cached run is faster (2.76s vs 3.77s) - likely JIT compilation warmup on first run.

---

## Known Issues / Conflicts Discovered

### Issue 1: Project-Local Cache Instead of Standard HF Cache

**Problem:** Uses `{project_root}/models/` instead of `~/.cache/huggingface/hub/`.

**Impact:** Models not shared across projects; duplicate downloads waste disk space.

**Location:** `utils.py:28-38` [PROJECT], `mlx.py:76` [PROJECT]

### Issue 2: No Integrity Verification on Corrupted Files

**Problem:** HuggingFace's `snapshot_download()` only checks file existence, not integrity. Corrupted files pass validation.

**Impact:** Corruption detected only at load time, triggering silent fallback to larger/slower model.

**Location:** `mlx.py:81-84` [PROJECT]

### Issue 3: Silent Fallback Degrades Performance

**Problem:** When quantized model fails to load, it silently falls back to non-quantized model without prominent warning.

**Impact:** Users unknowingly get 68% slower transcription (1.18s vs 0.70s for small model).

**Location:** `mlx.py:92-112` [PROJECT]

### Issue 4: Corrupted Files Not Cleaned Up

**Problem:** After fallback, corrupted model files remain in cache.

**Impact:** Every subsequent run attempts the corrupted model first, fails, and falls back - wasting time.

**Location:** `mlx.py:92-112` [PROJECT]

### Issue 5: Misleading Log Message

**Problem:** Log says "Downloading quantized model" even for non-quantized models like `whisper-large-v3-turbo`.

**Impact:** User confusion about what model is actually being used.

**Location:** `mlx.py:80` [PROJECT]

### Issue 6: "large" Maps to Turbo Variant

**Problem:** `model_name="large"` maps to `whisper-large-v3-turbo` (distilled variant), not the full `whisper-large-v3`.

**Impact:** Users expecting full large model accuracy get turbo variant instead.

**Location:** `mlx.py:54` [PROJECT]

---

## Recommended Improvements

### Improvement 1: Use Standard HuggingFace Cache (P2 - Medium)

**Problem:** Project-local cache prevents model sharing across projects.

**Current code:**
```python
# mlx.py:76, utils.py:28-38
models_dir = str(get_models_dir())  # Returns {project_root}/models/
model_path = snapshot_download(repo_id=self.hf_repo, cache_dir=models_dir)
```

**Recommended Fix:**
```python
# Option A: Remove cache_dir to use default HF cache
model_path = snapshot_download(repo_id=self.hf_repo)  # Uses ~/.cache/huggingface/hub/

# Option B: Allow user configuration via environment variable
import os
cache_dir = os.environ.get("MLX_WHISPER_CACHE", None)  # None = use HF default
model_path = snapshot_download(repo_id=self.hf_repo, cache_dir=cache_dir)
```

**Effort:** ~5 lines
**Impact:** Eliminates duplicate downloads across projects; saves disk space.

---

### Improvement 2: Add Integrity Verification (P1 - High)

**Problem:** Corrupted files not detected until load failure.

**Recommended Fix:**
```python
# After snapshot_download, verify the weights file can be opened
import zipfile

weights_path = Path(model_path) / "weights.npz"
if weights_path.exists():
    try:
        with zipfile.ZipFile(weights_path, 'r') as zf:
            zf.testzip()  # Verify integrity
    except (zipfile.BadZipFile, Exception) as e:
        self.log.warning(f"Corrupted weights file detected: {e}")
        # Delete corrupted file and re-download
        weights_path.unlink()
        model_path = snapshot_download(repo_id=self.hf_repo, cache_dir=models_dir, force_download=True)
```

**Effort:** ~15 lines
**Impact:** Prevents silent fallback to slower model; auto-recovers from corruption.

---

### Improvement 3: Make Fallback More Visible (P2 - Medium)

**Problem:** Fallback to non-quantized model is easy to miss in logs.

**Current code:**
```python
self.log.warning(f"Failed to load quantized model {self.hf_repo}: {e}")
self.log.info(f"Falling back to non-quantized model: {fallback_repo}")
```

**Recommended Fix:**
```python
self.log.warning(
    f"⚠️  FALLBACK ACTIVATED: Failed to load {self.hf_repo}, "
    f"using slower non-quantized model {fallback_repo}. "
    f"To fix: delete models/{self.hf_repo.replace('/', '--')} and re-run."
)
```

**Effort:** ~3 lines
**Impact:** Users aware of degraded performance; given actionable fix.

---

### Improvement 4: Fix Misleading Log Message (P3 - Low)

**Problem:** Says "Downloading quantized model" for non-quantized models.

**Current code:**
```python
self.log.info(f"Downloading quantized model {self.hf_repo} to {models_dir}")
```

**Recommended Fix:**
```python
quant_status = "quantized" if self._is_quantized else "non-quantized"
self.log.info(f"Downloading {quant_status} model {self.hf_repo} to {models_dir}")
```

**Effort:** 2 lines
**Impact:** Accurate logging; reduces user confusion.

---

### Improvement 5: Document "large" vs "large-v3" Distinction (P3 - Low)

**Problem:** Users don't know "large" maps to turbo variant.

**Recommended Fix:** Add comment in model_map and update docstring:
```python
model_map = {
    ...
    "large": "mlx-community/whisper-large-v3-turbo",  # NOTE: Turbo variant for speed; use "large-v3" for full model
    ...
}
```

**Effort:** 1 line comment + docstring update
**Impact:** Users can make informed model choice.

---

## Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| P1 | Add integrity verification | ~15 lines | Prevents silent performance degradation | ⬜ Not started |
| P2 | Use standard HF cache | ~5 lines | Saves disk space across projects | ⬜ Not started |
| P2 | Make fallback more visible | ~3 lines | Users aware of degraded performance | ⬜ Not started |
| P3 | Fix misleading log message | 2 lines | Accurate logging | ⬜ Not started |
| P3 | Document large vs large-v3 | 1 line | Informed model choice | ⬜ Not started |

---

## Implementation Order Recommendation

### Phase 1: Quick Wins (< 10 lines each)
- [ ] Fix misleading "quantized" log message
- [ ] Add comment documenting large vs large-v3
- [ ] Make fallback warning more prominent

### Phase 2: Medium Effort
- [ ] Add integrity verification for weights files
- [ ] Auto-cleanup corrupted files before fallback

### Phase 3: Architecture Change
- [ ] Switch to standard HuggingFace cache (requires testing across projects)
- [ ] Add CLI option to configure cache location
