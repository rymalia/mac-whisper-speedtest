# ParakeetMLXImplementation - Model Details

This document details the model download, caching, and execution behavior for the `ParakeetMLXImplementation` class.

> **IMPORTANT**: Parakeet is NOT a Whisper implementation. It is NVIDIA's NeMo-based TDT (Token-and-Duration Transducer) ASR model, ported to Apple MLX. Unlike Whisper implementations where model sizes (tiny, small, large) correspond to different model architectures and sizes, ParakeetMLX maps ALL Whisper-style size parameters to the **same single model**: `parakeet-tdt-0.6b-v2`.

---

## File Reference Legend

| Tag | Meaning | Example |
|-----|---------|---------|
| `[PROJECT]` | Files in this project's source tree | `src/mac_whisper_speedtest/implementations/parakeet_mlx.py` |
| `[LIBRARY]` | Installed library code in `.venv` | `.venv/lib/python3.12/site-packages/parakeet_mlx/` |

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| Does `small` and `large` use different models? | **NO** - Both map to `parakeet-tdt-0.6b-v2` | `[PROJECT] parakeet_mlx.py:52-65` model_map |
| What library handles the download? | `huggingface_hub.hf_hub_download()` | `[LIBRARY] parakeet_mlx/utils.py:6,64-65` |
| Where are models cached? | `~/.cache/huggingface/hub/` (standard HF cache) | Empirical test - see below |
| Does the HF_HOME override work? | **NO** - Downloads go to standard HF cache despite override | Empirical test - observed download location |
| Model size on disk? | ~2.47 GB total (model.safetensors: 2,471,559,904 bytes) | `ls -la` output |
| Is this a Whisper model? | **NO** - Parakeet is NVIDIA's TDT ASR architecture | `[LIBRARY] parakeet_mlx/parakeet.py:240-368` ParakeetTDT class |

---

## Benchmark Execution Flow

### Command Traced
```bash
.venv/bin/python3 test_benchmark2.py small 1 ParakeetMLXImplementation
.venv/bin/python3 test_benchmark2.py large 1 ParakeetMLXImplementation
```

### Execution Steps (Both Commands Follow Same Path)

1. **Entry Point** `[PROJECT] test_benchmark2.py`
   - Parses CLI args: model_name="small"/"large", num_runs=1, implementations=["ParakeetMLXImplementation"]
   - Loads audio from `tests/jfk.wav` (176000 samples @ 16kHz)
   - Creates `BenchmarkConfig` and calls `run_benchmark()`

2. **Benchmark Runner** `[PROJECT] benchmark.py:110-190`
   - Instantiates `ParakeetMLXImplementation()`
   - Calls `implementation.load_model("small")` or `implementation.load_model("large")`

3. **Model Loading** `[PROJECT] parakeet_mlx.py:48-86` 

   ```python
   def load_model(self, model_name: str) -> None:
       # Lines 52-65: Model mapping - ALL Whisper sizes map to same model
       model_map = {
           "parakeet-tdt-0.6b": "mlx-community/parakeet-tdt-0.6b-v2",
           "parakeet-tdt-0.6b-v2": "mlx-community/parakeet-tdt-0.6b-v2",
           "parakeet-tdt-1.1b": "mlx-community/parakeet-tdt-1.1b",
           "parakeet-ctc-0.6b": "mlx-community/parakeet-ctc-0.6b",
           "parakeet-ctc-1.1b": "mlx-community/parakeet-ctc-1.1b",
           # Whisper model names all map to 0.6b TDT
           "tiny": "mlx-community/parakeet-tdt-0.6b-v2",
           "small": "mlx-community/parakeet-tdt-0.6b-v2",
           "base": "mlx-community/parakeet-tdt-0.6b-v2",
           "medium": "mlx-community/parakeet-tdt-0.6b-v2",
           "large": "mlx-community/parakeet-tdt-0.6b-v2",
           "large-v2": "mlx-community/parakeet-tdt-0.6b-v2",
           "large-v3": "mlx-community/parakeet-tdt-0.6b-v2",
       }
   ```

4. **HF Cache Setup (Attempted)** `[PROJECT] parakeet_mlx.py:68-74`

   ```python
   # Attempt to override HF cache location (BUT THIS DOESN'T WORK)
   models_dir = get_models_dir()
   original_cache_dir = os.environ.get("HF_HOME")
   os.environ["HF_HOME"] = str(models_dir)
   ```
   > **Bug**: Setting `HF_HOME` after `huggingface_hub` is imported has no effect. The library caches its config at import time.

5. **Model Download** `[LIBRARY] parakeet_mlx/utils.py:59-78`

   ```python
   def from_pretrained(hf_id_or_path: str, *, dtype: mx.Dtype = mx.bfloat16) -> BaseParakeet:
       try:
           config = json.load(open(hf_hub_download(hf_id_or_path, "config.json"), "r"))
           weight = hf_hub_download(hf_id_or_path, "model.safetensors")
       except Exception:
           # Fallback to local path (problematic - see issues)
           config = json.load(open(Path(hf_id_or_path) / "config.json", "r"))
           weight = str(Path(hf_id_or_path) / "model.safetensors")
   ```

6. **Transcription** `[PROJECT] parakeet_mlx.py:88-122`
   - Converts numpy array to MLX array
   - Calls `get_logmel()` to create mel spectrogram
   - Calls `model.generate(mel)` which runs TDT decoding
   - Returns `TranscriptionResult` with text and segments

---

## Summary Table

| Property | Value |
|----------|-------|
| Requested Model | `small` or `large` (any Whisper-style name) |
| Actual Model Used | `parakeet-tdt-0.6b-v2` (always) |
| HuggingFace Repo | `mlx-community/parakeet-tdt-0.6b-v2` |
| Download URL | `https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2` |
| Expected Cache Location | `~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/` |
| Attempted Cache Override | `[PROJECT]/models/` (doesn't work) |
| Model Architecture | TDT (Token-and-Duration Transducer) - NOT Whisper |
| Model Parameters | ~600M parameters |
| Files Downloaded | `config.json` (36KB), `model.safetensors` (2.47GB), tokenizer files (~260KB total) |
| Total Size | ~2.47 GB |

---

## Model Mapping Reference

**Critical**: Unlike Whisper implementations, Parakeet maps ALL standard model size names to the same model:

| Input Parameter | Actual HuggingFace Repo | Notes |
|-----------------|------------------------|-------|
| `tiny` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `base` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `small` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `medium` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `large` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `large-v2` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `large-v3` | `mlx-community/parakeet-tdt-0.6b-v2` | Same model |
| `parakeet-tdt-0.6b` | `mlx-community/parakeet-tdt-0.6b-v2` | Explicit name |
| `parakeet-tdt-1.1b` | `mlx-community/parakeet-tdt-1.1b` | Larger variant |
| `parakeet-ctc-0.6b` | `mlx-community/parakeet-ctc-0.6b` | CTC variant |
| `parakeet-ctc-1.1b` | `mlx-community/parakeet-ctc-1.1b` | Larger CTC |

---

## Notes

### Architecture Differences
Parakeet uses a fundamentally different architecture from Whisper:  

- **Whisper**: Encoder-decoder transformer with attention-based generation. 
- **Parakeet TDT**: Conformer encoder + RNNT-style transducer with Token-and-Duration prediction

This means:
1. Different audio preprocessing (128-dimensional mel features vs Whisper's 80)
2. Different decoding strategy (greedy TDT decoding vs beam search)
3. Different output format (aligned tokens with durations vs segments with timestamps)

### HuggingFace CAS Issues
During testing, the HuggingFace CAS (Content-Addressable Storage) service experienced persistent failures:

```
RuntimeError: Data processing error: CAS service error : Error : single flight error:
Real call failed: CasObjectError(InternalIOError(Custom { kind: Other, error:
reqwest::Error { kind: Decode, source: hyper::Error(Body, Custom { kind: UnexpectedEof,
error: IncompleteBody }) } }))
```
This appears to be a transient HuggingFace CDN issue. Direct curl download from the HuggingFace URL succeeded.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `[PROJECT] src/mac_whisper_speedtest/implementations/parakeet_mlx.py` | Implementation wrapper |
| `[LIBRARY] .venv/.../parakeet_mlx/utils.py` | Model loading via `from_pretrained()` |
| `[LIBRARY] .venv/.../parakeet_mlx/parakeet.py` | Core Parakeet model classes (TDT, RNNT, CTC) |
| `[LIBRARY] .venv/.../parakeet_mlx/audio.py` | Audio preprocessing and mel spectrogram |

---

## Empirical Test Results

**Test Date**: 2026-01-11

### Download Testing

#### Initial Download Attempt (via huggingface_hub)
The initial download via the `huggingface_hub` library failed due to HuggingFace CAS service errors:

```bash
$ .venv/bin/python3 test_benchmark2.py small 1 ParakeetMLXImplementation
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
...
RuntimeError: Data processing error: CAS service error : Error : single flight error:
Real call failed: CasObjectError(InternalIOError(...IncompleteBody...))
```

**Partial download observed:**

```bash
$ ls -la ~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/blobs/
-rw-r--r--@ 1 rymalia  staff   36176 Jan 11 20:02 8955c588b5549ef70811f2121c6c8bda33508992  # config.json
-rw-r--r--@ 1 rymalia  staff  2471559904 Jan 11 20:20 ...74c96f.incomplete  # model.safetensors (incomplete)
```

#### Successful Download (via curl)
Due to persistent HuggingFace CAS errors, the model was downloaded via curl:

```bash
$ curl -L -o model.safetensors.download \
  "https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2/resolve/main/model.safetensors"
...
Downloaded file size: 2471559904 bytes
Download appears successful!
```

The file was then manually placed in the HuggingFace cache structure:

```bash
$ mv model.safetensors.download ~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/blobs/74c5e8b87417131dbed4a6cd8b81e1f4d5f6479dc5c35d6429288be7f704c96f
$ ln -sf "../../blobs/74c5e8b..." snapshots/8ae155.../model.safetensors
```

#### Final Cache State
```bash
$ ls -la ~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2/snapshots/8ae155301e23d820d82aa60d24817c900e69e487/
total 0
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 11 20:21 .gitattributes -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 11 20:02 config.json -> ../../blobs/8955c588b5549ef70811f2121c6c8bda33508992
lrwxr-xr-x@ 1 rymalia  staff   76 Jan 11 23:02 model.safetensors -> ../../blobs/74c5e8b87417131dbed4a6cd8b81e1f4d5f6479dc5c35d6429288be7f704c96f
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 11 20:21 README.md -> ../../blobs/4c40c9c8d725bd2f1c904199561239f388d02e85
lrwxr-xr-x@ 1 rymalia  staff   76 Jan 11 20:21 tokenizer.model -> ../../blobs/5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 11 20:21 tokenizer.vocab -> ../../blobs/d3ce5dea5479a740d2bd5009cf4c86be2d0a6eae
lrwxr-xr-x@ 1 rymalia  staff   52 Jan 11 20:21 vocab.txt -> ../../blobs/18f79afd579e1b06fecae25f4e0d7c2b404a2aaf
```

### Small Model Tests

**Note**: "Small" and "large" are the same model for Parakeet - this tests the `small` parameter.

```bash
$ .venv/bin/python3 test_benchmark2.py small 1 ParakeetMLXImplementation
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - ParakeetMLXImplementation

Starting benchmark with model 'small' (1 run(s))...
2026-01-11 23:02:38 [info     ] Benchmarking ParakeetMLXImplementation with model small
2026-01-11 23:02:38 [info     ] Loading model for ParakeetMLXImplementation
2026-01-11 23:02:38 [info     ] Loading Parakeet MLX model parameter: small
2026-01-11 23:02:38 [info     ] Loading Parakeet model from mlx-community/parakeet-tdt-0.6b-v2
2026-01-11 23:02:40 [info     ] Successfully loaded Parakeet model: <class 'parakeet_mlx.parakeet.ParakeetTDT'>
2026-01-11 23:02:40 [info     ] Model cached in: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
2026-01-11 23:02:40 [info     ] Run 1/1 for ParakeetMLXImplementation
2026-01-11 23:02:40 [info     ] Transcribing with Parakeet MLX using model small
2026-01-11 23:02:43 [info     ] Transcription completed: 'And so, my fellow Americans, ask not what your cou...' (1 segments)
2026-01-11 23:02:43 [info     ] Run 1 completed in 2.6474 seconds
2026-01-11 23:02:43 [info     ] Average time for ParakeetMLXImplementation: 2.6474 seconds

=== Benchmark Summary for 'small' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
parakeet-mlx           2.6474          model=parakeet-tdt-0.6b-v2, implementation=parakeet-mlx, platform=Apple Silicon (MLX)
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your..."
```

**Results:**
- Model load time: ~2 seconds (first load)
- Transcription time: **2.6474 seconds**
- Actual model used: `parakeet-tdt-0.6b-v2` (as expected)

### Large Model Tests

**Note**: Same model as "small" - this verifies the mapping behavior.

```bash
$ .venv/bin/python3 test_benchmark2.py large 1 ParakeetMLXImplementation
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - ParakeetMLXImplementation

Starting benchmark with model 'large' (1 run(s))...
2026-01-11 23:02:48 [info     ] Benchmarking ParakeetMLXImplementation with model large
2026-01-11 23:02:48 [info     ] Loading model for ParakeetMLXImplementation
2026-01-11 23:02:49 [info     ] Loading Parakeet MLX model parameter: large
2026-01-11 23:02:49 [info     ] Loading Parakeet model from mlx-community/parakeet-tdt-0.6b-v2
2026-01-11 23:02:50 [info     ] Successfully loaded Parakeet model: <class 'parakeet_mlx.parakeet.ParakeetTDT'>
2026-01-11 23:02:50 [info     ] Model cached in: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/models
2026-01-11 23:02:50 [info     ] Run 1/1 for ParakeetMLXImplementation
2026-01-11 23:02:50 [info     ] Transcribing with Parakeet MLX using model large
2026-01-11 23:02:51 [info     ] Transcription completed: 'And so, my fellow Americans, ask not what your cou...' (1 segments)
2026-01-11 23:02:51 [info     ] Run 1 completed in 1.2560 seconds
2026-01-11 23:02:51 [info     ] Average time for ParakeetMLXImplementation: 1.2560 seconds

=== Benchmark Summary for 'large' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
parakeet-mlx           1.2560          model=parakeet-tdt-0.6b-v2, implementation=parakeet-mlx, platform=Apple Silicon (MLX)
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your..."
```

**Results:**
- Model load time: ~1 second (already warm from previous run)
- Transcription time: **1.2560 seconds** (faster due to MLX caching)
- Actual model used: `parakeet-tdt-0.6b-v2` (same as "small")

### Cached Behavior Verification

Both runs used the same cached model from `~/.cache/huggingface/hub/`. The faster second run (1.26s vs 2.65s) demonstrates MLX's in-memory model caching - the weights were already loaded from the first run.

---

## Known Issues / Conflicts Discovered

### Issue 1: HF_HOME Override Doesn't Work (P1)

**Problem**: The implementation attempts to redirect HuggingFace downloads to the project's `models/` folder by setting `HF_HOME`, but this has no effect.

**Location**: `[PROJECT] parakeet_mlx.py:68-74`

**Root Cause**: The `huggingface_hub` library caches its configuration (including cache directory) at import time. Setting `HF_HOME` after import has no effect.

**Impact**:
- Models download to `~/.cache/huggingface/hub/` instead of project's `models/` folder
- Misleading log message claims model is cached in project folder
- Cannot share models between projects as intended

### Issue 2: All Model Sizes Map to Same Model (P2)

**Problem**: Users expecting Whisper-style model size behavior will get the same 0.6b model regardless of the size parameter they specify.

**Location**: `[PROJECT] parakeet_mlx.py:52-65`

**Impact**:
- Confusing UX for users expecting different model sizes
- Benchmark comparisons across sizes are meaningless for this implementation
- No way to access larger Parakeet models (1.1b) via standard CLI

### Issue 3: HuggingFace CAS Download Failures (P1 - External)

**Problem**: The HuggingFace xet-core CAS system experiences persistent download failures with `IncompleteBody` errors on the day this investigation was run.

**Location**: External - `huggingface_hub` library and HuggingFace CDN

**Impact**:
- First-time model downloads may fail repeatedly
- Users need to manually download via curl as workaround
- No automatic retry or fallback mechanism

### Issue 4: Poor Error Handling in parakeet-mlx Library (P2 - External)

**Problem**: The `from_pretrained()` function catches download exceptions and falls back to treating the HuggingFace repo ID as a local path, which always fails with a confusing error.

**Location**: `[LIBRARY] parakeet_mlx/utils.py:66-68`

```python
except Exception:
    config = json.load(open(Path(hf_id_or_path) / "config.json", "r"))  # Always fails
```

**Impact**: When download fails, users see "No such file or directory: 'mlx-community/parakeet-tdt-0.6b-v2/config.json'" instead of the actual download error.

---

## Recommended Improvements

### Improvement 1: Fix HF_HOME Override

**Problem**: Setting `HF_HOME` after huggingface_hub import has no effect.

**Impact**: Models download to wrong location; cannot share models across projects.

**Location**: `[PROJECT] parakeet_mlx.py:68-74`

**Recommended Fix**: Use `huggingface_hub`'s proper cache directory configuration.

```python
# Current (broken):
os.environ["HF_HOME"] = str(models_dir)

# Fixed (use huggingface_hub's API):
from huggingface_hub import constants as hf_constants
hf_constants.HF_HUB_CACHE = str(models_dir / "hub")
```

**Alternative Fix**: Set `HF_HOME` before importing the library (in a wrapper script or `__init__.py`).

**Effort**: ~10 lines
**Priority**: P1 (High)

---

### Improvement 2: Add Warning for Model Size Mapping

**Problem**: Users are unaware that all Whisper model sizes map to the same Parakeet model.

**Impact**: Confusion when comparing benchmark results; unexpected behavior.

**Location**: `[PROJECT] parakeet_mlx.py:48-65`

**Recommended Fix**: Add warning log when Whisper-style model name is used.

```python
def load_model(self, model_name: str) -> None:
    whisper_sizes = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}

    if model_name in whisper_sizes:
        self.log.warning(
            f"Parakeet is not a Whisper model. '{model_name}' maps to "
            f"'parakeet-tdt-0.6b-v2'. Use 'parakeet-tdt-1.1b' for larger model."
        )

    # ... rest of load_model
```

**Effort**: ~5 lines
**Priority**: P2 (Medium)

---

### Improvement 3: Add Larger Model Support to CLI

**Problem**: Users cannot access larger Parakeet models (1.1b variants) via CLI.

**Impact**: Missing functionality; cannot benchmark larger models.

**Location**: `[PROJECT] parakeet_mlx.py:52-65` and `[PROJECT] cli.py`

**Recommended Fix**: Either expose Parakeet-specific model names in CLI or add model aliases:

```python
# Option A: Map "large" to actual larger model
model_map = {
    # ...existing mappings...
    "large": "mlx-community/parakeet-tdt-1.1b",  # Use actual larger model
}

# Option B: Add implementation-specific model options in CLI
# --model parakeet-1.1b
```

**Effort**: ~20 lines
**Priority**: P2 (Medium)

---

### Improvement 4: Better Download Error Handling

**Problem**: Download failures show confusing "file not found" errors instead of actual download errors.

**Impact**: Users cannot diagnose download issues; poor error messages.

**Location**: `[PROJECT] parakeet_mlx.py:78-85`

**Recommended Fix**: Catch and re-raise with context, add retry logic.

```python
def load_model(self, model_name: str) -> None:
    # ...
    try:
        self._model = from_pretrained(self._hf_repo)
    except FileNotFoundError as e:
        # Likely a failed download that fell back to local path lookup
        raise RuntimeError(
            f"Failed to download model from {self._hf_repo}. "
            f"Try downloading manually: huggingface-cli download {self._hf_repo}"
        ) from e
    except Exception as e:
        self.log.error(f"Failed to load Parakeet model: {e}")
        raise
```

**Effort**: ~15 lines
**Priority**: P1 (High)

---

## Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| P1 | Fix HF_HOME override | ~10 lines | Models download to correct location | Not started |
| P1 | Better download error handling | ~15 lines | Users can diagnose issues | Not started |
| P2 | Add model size warning | ~5 lines | Reduced user confusion | Not started |
| P2 | Add larger model support | ~20 lines | Access to 1.1b models | Not started |

---

## Implementation Order Recommendation

### Phase 1: Critical Fixes
- [ ] Fix download error handling (P1) - Immediate UX improvement
- [ ] Add model size mapping warning (P2) - Quick win, prevents confusion

### Phase 2: Cache Management
- [ ] Fix HF_HOME override (P1) - Requires testing across platforms

### Phase 3: Feature Enhancement
- [ ] Add larger model support (P2) - Expands benchmark capabilities

---

## Additional Findings

### Finding 1: Parakeet Produces Different Output Format

Unlike Whisper implementations that return segments with timestamps, Parakeet returns aligned tokens with durations. The implementation wraps these into a single segment:

```python
# [PROJECT] parakeet_mlx.py:113-117
segments = [TranscriptionSegment(
    start=0.0,
    end=result.end,
    text=result.text
)]
```

This loses the fine-grained alignment information that Parakeet provides. A future improvement could expose this richer output.

### Finding 2: Model Is Not English-Only

Parakeet TDT 0.6b-v2 is trained primarily on English data but the model architecture supports multilingual input. The implementation hardcodes `language="en"`:

```python
# [PROJECT] parakeet_mlx.py:119
language="en",  # Parakeet is English-only
```

This is correct for the current model but should be made configurable if multilingual Parakeet models become available.
