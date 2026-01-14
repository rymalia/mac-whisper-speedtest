# WhisperCppCoreMLImplementation - Deep Dive Documentation

**Implementation**: `WhisperCppCoreMLImplementation`
**Backend**: whisper.cpp via pywhispercpp Python bindings
**Test Date**: 2026-01-12
**Status**: Empirically Verified

---

## File Reference Legend

| Tag | Description |
|-----|-------------|
| **[PROJECT]** | Files in this project's source tree |
| **[LIBRARY]** | Files in `.venv/lib/python3.12/site-packages/` |

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| **HF Hub Cache Usage?** | **NO** - Does not use `~/.cache/huggingface/hub/` | [LIBRARY] `pywhispercpp/utils.py:54` uses direct `requests.get()` |
| **Download Method?** | Direct HTTP via `requests` library | [LIBRARY] `pywhispercpp/utils.py:47-68` |
| **Check HF Cache First?** | **NO** - Only checks its own cache location | [LIBRARY] `pywhispercpp/utils.py:50` - simple `file_path.exists()` |
| **File Conversions?** | **NO** - Uses GGML `.bin` files as-is | Files downloaded as `ggml-{model}.bin`, used directly |
| **Why No HF Hub API?** | whisper.cpp uses custom GGML format, not HuggingFace Transformers | GGML is a custom tensor format for CPU inference |
| **Fallback Chains?** | **NO** - Single download source, no fallbacks | [LIBRARY] `pywhispercpp/utils.py:47` - single URL construction |
| **~/Library/Application Support Usage?** | Default location, but **overridden by project** | [PROJECT] `utils.py:36` sets `models_dir` to `{project}/models` |
| **CPU or GPU?** | **GPU (Metal)** + CPU with BLAS | `ggml_metal_init` in output confirms Metal usage |
| **CoreML Enabled?** | **NO** - Binary compiled without CoreML support | `whisper_print_system_info()` shows `COREML = 0` |
| **Apple Silicon Optimizations?** | **YES** - Metal GPU, ARM FMA, FP16 VA, Accelerate BLAS | System info: `ARM_FMA = 1`, `FP16_VA = 1` |
| **Quantization Used?** | **YES** - Some models use q5_0/q5_1/q8_0 GGML quantization | Model mapping selects quantized variants |
| **SDPA Relevance?** | **NOT APPLICABLE** - whisper.cpp is C/C++, not PyTorch | SDPA is a PyTorch-specific optimization |

---

## Technical Background

### What is CoreML?

**CoreML** is Apple's machine learning framework that enables models to run on Apple's Neural Engine (ANE), GPU, and CPU. Key points:

- **Neural Engine**: Dedicated ML accelerator chip on Apple Silicon (M1/M2/M3/M4) that can deliver 3x+ speedup vs CPU-only
- **CoreML Format**: Models must be converted to `.mlmodelc` (compiled CoreML) format
- **whisper.cpp CoreML Status**: The pywhispercpp binary from PyPI is **NOT compiled with CoreML support** (`COREML = 0`)
- **What This Means**: This implementation **cannot use the Neural Engine** despite the class name suggesting CoreML support

### What is BLAS Backend?

**BLAS (Basic Linear Algebra Subprograms)** is a standard API for performing basic linear algebra operations like matrix multiplication. On macOS:

- **Apple Accelerate Framework**: Provides highly optimized BLAS implementation for Apple Silicon
- **whisper.cpp Usage**: Uses BLAS for matrix operations in transformer layers
- **Output Message**: `whisper_backend_init: using BLAS backend` confirms Accelerate is active
- **Performance Impact**: Provides significant CPU speedup via vectorized operations and multi-threading

### What is Metal (GPU)?

**Metal** is Apple's low-level GPU API:

- `ggml_metal_init` messages show Metal GPU acceleration is being used
- Skipped BF16 kernels indicate some newer optimizations require M3+ chips or newer macOS
- **This is the PRIMARY acceleration** being used in this implementation

### What is SDPA (Scaled Dot-Product Attention)?

**SDPA** is a PyTorch optimization that's **NOT relevant to this implementation**:

- PyTorch 2.0+ provides fused attention kernels (FlashAttention, Memory-Efficient Attention)
- whisper.cpp is written in pure **C/C++**, not PyTorch
- whisper.cpp has its own optimized attention using ARM NEON SIMD intrinsics

### Quantization Formats

whisper.cpp uses **GGML quantization** to reduce model size:

| Format | Description | Size Reduction |
|--------|-------------|----------------|
| `q5_0` | 5-bit quantization (method 0) | ~60% smaller |
| `q5_1` | 5-bit quantization (method 1) | ~60% smaller |
| `q8_0` | 8-bit quantization | ~50% smaller |
| (none) | Full FP16 precision | Baseline |

---

## Benchmark Execution Flow

### Command: Small Model
```bash
.venv/bin/python3 test_benchmark2.py small 1 "WhisperCppCoreMLImplementation"
```

### Command: Large Model
```bash
.venv/bin/python3 test_benchmark2.py large 1 "WhisperCppCoreMLImplementation"
```

### Execution Steps

1. **Entry Point**: `test_benchmark2.py:88-94`
   - Parses CLI args: `model="small"` or `"large"`, `num_runs=1`

2. **Audio Loading**: `test_benchmark2.py:36-55`
   - Loads `tests/jfk.wav` (11 seconds, 16kHz mono)
   - Converts to float32 numpy array

3. **Implementation Discovery**: `test_benchmark2.py:58-74`
   - Filters to `WhisperCppCoreMLImplementation` from available implementations

4. **Benchmark Run**: `benchmark.py:124-134`
   - Creates `WhisperCppCoreMLImplementation()` instance
   - Calls `implementation.load_model("small")` or `load_model("large")`

5. **Model Loading** [PROJECT] `coreml.py:26-77`:
   ```python
   # Model name mapping
   models_map = {
       "tiny": "tiny-q5_1",
       "base": "base-q5_1",
       "small": "small",        # Full precision for small
       "medium": "medium-q5_0",
       "large": "large-v3-turbo-q5_0",  # Quantized turbo version
   }
   self.model_name = models_map.get(model_name, model_name)
   ```

6. **CoreML Check** [PROJECT] `coreml.py:48-70`:
   - Sets `WHISPER_COREML=1` environment variable (has no effect without CoreML binary)
   - Checks for `.mlmodelc` file (never found - not provided)
   - Falls back to `coreml_enabled = False`

7. **pywhispercpp Model Initialization** [LIBRARY] `model.py:68-96`:
   ```python
   # Called with models_dir={project}/models
   self.model_path = utils.download_model(model, models_dir)
   self._ctx = pw.whisper_init_from_file(self.model_path)
   ```

8. **Model Download** [LIBRARY] `utils.py:29-73`:
   - Constructs URL: `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin`
   - Downloads via `requests.get(url, stream=True)` with tqdm progress bar
   - Saves to `{project}/models/ggml-{model}.bin`

9. **Transcription** [PROJECT] `coreml.py:79-106`:
   ```python
   segments = self._model.transcribe(audio, language=None, translate=False)
   text = " ".join([segment.text for segment in segments])
   ```

10. **Result**: Returns `TranscriptionResult` with text, segments, and language

---

## Summary Table

### Small Model (`small`)

| Attribute | Value |
|-----------|-------|
| Requested Model | `small` |
| Actual Model | `small` (full precision, not quantized) |
| Download URL | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin` |
| Cache Location | `{project}/models/ggml-small.bin` |
| File Size | 465 MB (487,601,967 bytes) |
| Download Method | Direct HTTP via `requests` |
| HF Hub API | **NOT USED** |

### Large Model (`large`)

| Attribute | Value |
|-----------|-------|
| Requested Model | `large` |
| Actual Model | `large-v3-turbo-q5_0` (quantized turbo variant) |
| Download URL | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin` |
| Cache Location | `{project}/models/ggml-large-v3-turbo-q5_0.bin` |
| File Size | 547 MB (574,041,195 bytes) |
| Download Method | Direct HTTP via `requests` |
| HF Hub API | **NOT USED** |

---

## Model Mapping Reference

### Project-Level Mapping [PROJECT] `coreml.py:37-44`

| Input | Output | Notes |
|-------|--------|-------|
| `tiny` | `tiny-q5_1` | 5-bit quantized |
| `base` | `base-q5_1` | 5-bit quantized |
| `small` | `small` | Full precision |
| `medium` | `medium-q5_0` | 5-bit quantized |
| `large` | `large-v3-turbo-q5_0` | Turbo variant, 5-bit quantized |

### CoreML Model Mapping (Unused) [PROJECT] `coreml.py:53-58`

The implementation has a separate mapping for CoreML models, but since CoreML is not enabled, this is never used:

| Input | CoreML Model Checked |
|-------|---------------------|
| `tiny` | `ggml-tiny-encoder.mlmodelc` |
| `base` | `ggml-base-encoder.mlmodelc` |
| `small` | `ggml-small-encoder.mlmodelc` |
| `medium` | `ggml-medium-encoder.mlmodelc` |
| `large` | `ggml-large-v3-turbo-encoder.mlmodelc` |

---

## Cache Behavior Analysis

### Download Location Hierarchy

1. **Project-Level Override**: `{project}/models/` (used by this implementation)
2. **pywhispercpp Default**: `~/Library/Application Support/pywhispercpp/models/` (NOT used)
3. **HuggingFace Cache**: `~/.cache/huggingface/hub/` (**NEVER checked**)

### File Existence Check [LIBRARY] `utils.py:49-51`

```python
file_path = Path(download_dir) / os.path.basename(url)
if file_path.exists():
    logger.info(f"Model {model_name} already exists in {download_dir}")
```

**No checksums, no partial download resume, no validation** - just a simple file existence check.

### Error Recovery [LIBRARY] `utils.py:69-72`

```python
except Exception as e:
    os.remove(file_path)  # Delete partial file on error
    raise e
```

Partial downloads are deleted on error, but there's no retry mechanism.

---

## Notes

### Misleading Class Name

The class is named `WhisperCppCoreMLImplementation`, but **CoreML is NOT actually used**:

1. The pywhispercpp binary from PyPI is compiled **without CoreML support**
2. `whisper_print_system_info()` returns `COREML = 0`
3. The implementation checks for `.mlmodelc` files that don't exist and wouldn't work anyway

**The actual acceleration comes from Metal GPU and Accelerate BLAS, not CoreML.**

### Actual Acceleration Stack

```
whisper.cpp
├── Metal GPU (ggml_metal_init)      ← PRIMARY GPU acceleration
├── Accelerate BLAS                   ← Optimized linear algebra
├── ARM NEON SIMD                     ← Vector instructions
└── Multi-threaded CPU (n_threads=4) ← Parallelization
```

### Why Not HuggingFace Hub API?

whisper.cpp uses a custom **GGML tensor format** that is incompatible with HuggingFace's safetensors/PyTorch formats. The models are hosted on HuggingFace but downloaded via direct HTTP, not the Hub API.

Benefits of current approach:
- Simple, dependency-light (just `requests`)
- No authentication required for public models
- Direct download without Hub client overhead

Drawbacks:
- No caching interoperability with other HF-based tools
- No resumable downloads
- No checksum verification

---

## Key Source Files

| File | Type | Purpose |
|------|------|---------|
| `src/mac_whisper_speedtest/implementations/coreml.py` | [PROJECT] | Main implementation class |
| `src/mac_whisper_speedtest/utils.py` | [PROJECT] | `get_models_dir()` - sets cache location |
| `.venv/lib/python3.12/site-packages/pywhispercpp/model.py` | [LIBRARY] | Model class, transcription logic |
| `.venv/lib/python3.12/site-packages/pywhispercpp/utils.py` | [LIBRARY] | `download_model()` - HTTP download |
| `.venv/lib/python3.12/site-packages/pywhispercpp/constants.py` | [LIBRARY] | Available models, default cache path |

---

## Empirical Test Results

### Test Environment

- **Date**: 2026-01-12
- **Machine**: Apple Silicon Mac
- **Python**: 3.12 (via uv venv)
- **pywhispercpp version**: 1.3.1 (from PyPI)

### Small Model Tests

#### Fresh Download Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 "WhisperCppCoreMLImplementation"
```

**Terminal Output (excerpt):**
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - WhisperCppCoreMLImplementation

Starting benchmark with model 'small' (1 run(s))...
[info] CoreML support is enabled for whisper.cpp
[warning] CoreML model not found at .../models/ggml-small-encoder.mlmodelc. Will use CPU fallback.
Downloading Model small ...: 100%|██████████| 465M/465M [03:09<00:00, 2.58MiB/s]
...
[info] Run 1 completed in 1.0678 seconds
[info] Average time for WhisperCppCoreMLImplementation: 1.0678 seconds

=== Benchmark Summary for 'small' model ===
whisper.cpp            1.0678          model=small, coreml=False, n_threads=4
```

**Downloaded File:**
```bash
$ ls -la models/ggml-small.bin
-rw-r--r--@ 1 rymalia  staff  487601967 Jan 12 17:07 models/ggml-small.bin

$ du -sh models/ggml-small.bin
480M	models/ggml-small.bin
```

#### Cached Run Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 "WhisperCppCoreMLImplementation"
```

**Terminal Output (excerpt):**
```
ggml_metal_init: found device: Apple M2 Max
ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128 (not supported)
whisper_backend_init: using BLAS backend
whisper_init_state: kv self size  =   18.87 MB
whisper_init_state: compute buffer (encode) =  128.01 MB
[info] Run 1 completed in 0.9359 seconds
[info] Average time for WhisperCppCoreMLImplementation: 0.9359 seconds

=== Benchmark Summary for 'small' model ===
whisper.cpp            0.9359          model=small, coreml=False, n_threads=4
    "And so my fellow Americans, ask not what your country can do for you..."
```

**Observations:**
- No re-download occurred (file existed)
- Metal GPU initialization confirmed
- Transcription time: ~0.94 seconds
- CoreML disabled as expected

### Large Model Tests

#### Fresh Download Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 "WhisperCppCoreMLImplementation"
```

**Terminal Output (excerpt):**
```
[info] CoreML support is enabled for whisper.cpp
[warning] CoreML model not found at .../models/ggml-large-v3-turbo-encoder.mlmodelc. Will use CPU fallback.
Downloading Model large-v3-turbo-q5_0 ...: 100%|██████████| 547M/547M [03:49<00:00, 2.51MiB/s]
...
[info] Run 1 completed in 3.2398 seconds
[info] Average time for WhisperCppCoreMLImplementation: 3.2398 seconds

=== Benchmark Summary for 'large' model ===
whisper.cpp            3.2398          model=large-v3-turbo-q5_0, coreml=False, n_threads=4
```

**Downloaded File:**
```bash
$ ls -la models/ggml-large-v3-turbo-q5_0.bin
-rw-r--r--@ 1 rymalia  staff  574041195 Jan 12 17:12 models/ggml-large-v3-turbo-q5_0.bin

$ du -sh models/ggml-large-v3-turbo-q5_0.bin
560M	models/ggml-large-v3-turbo-q5_0.bin
```

#### Cached Run Test

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 "WhisperCppCoreMLImplementation"
```

**Terminal Output:**
```
ggml_metal_init: found device: Apple M2 Max
whisper_backend_init: using BLAS backend
whisper_init_state: compute buffer (encode) =  212.29 MB
[info] Run 1 completed in 3.2450 seconds

=== Benchmark Summary for 'large' model ===
whisper.cpp            3.2450          model=large-v3-turbo-q5_0, coreml=False, n_threads=4
    "And so, my fellow Americans, ask not what your country can do for you..."
```

**Observations:**
- Model maps to turbo variant with q5_0 quantization (547MB instead of ~3GB)
- No download timeout issues (turbo model is smaller)
- Transcription time: ~3.24 seconds

### System Info Verification

```bash
$ .venv/bin/python3 -c "from pywhispercpp.model import Model; print(Model.system_info())"
AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = -1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | COREML = 0 | OPENVINO = 0 |
```

**Key Findings:**
- `COREML = 0` - **CoreML is NOT compiled in**
- `ARM_FMA = 1` - ARM Fused Multiply-Add enabled
- `FP16_VA = 1` - FP16 vector arithmetic enabled (Apple Silicon optimization)
- `NEON = -1` - NEON status unclear (possibly runtime-detected)

---

## Known Issues / Conflicts Discovered

### Issue 1: CoreML Support Not Compiled (So CoreML Check is Dead Code)  

**Problem**: The binary as-is does not support CoreML, But the implementation sets `WHISPER_COREML=1` and checks for `.mlmodelc` files (which aren't there and couldn't be used if they were)  
**Locations**: [LIBRARY] `pywhispercpp` build configuration & [PROJECT] `coreml.py:24, 48-70`  
**Impact**: Missing a MAJOR Performance Enhancer, misleading warnings, unnecessary file system checks  
**Recommendation**: Either rebuild pywhispercpp with CoreML support or remove CoreML code in `coreml.py`  

### Issue 2: Misleading Class Name

**Problem**: Class is named `WhisperCppCoreMLImplementation` but CoreML is not used  
**Impact**: User confusion about actual acceleration being used  
**Status**: Documentation-only (name change would break backward compatibility)  

### Issue 3: No Download Resume/Checksum

**Problem**: If download fails partway, the partial file is deleted and must restart from scratch  
**Location**: [LIBRARY] `pywhispercpp/utils.py:69-72`  
**Impact**: Wasted bandwidth on unstable connections  
**Status**: Library-level issue  

### Issue 4: No HF Cache Integration

**Problem**: Cannot share models with other HuggingFace-based implementations  
**Location**: [LIBRARY] `pywhispercpp/utils.py:47`  
**Impact**: Duplicate disk space usage  
**Status**: Design choice (GGML format is incompatible anyway)  

---

## Recommended Improvements

### Improvement 1: Consider Building pywhispercpp with CoreML

**Problem**: CoreML support is disabled in the PyPI binary  
**Impact**: Missing 3x+ speedup from Neural Engine  
**Location**: Library build configuration  

**Option A**: Request CoreML build from pywhispercpp maintainers  
**Option B**: Build pywhispercpp from source with `-DWHISPER_COREML=1`:  

```bash
# Requires Xcode command-line tools and coremltools
pip install git+https://github.com/abdeladim-s/pywhispercpp.git \
    --install-option="--cmake-args=-DWHISPER_COREML=1"
```

**Effort**: Large (build system changes, model conversion)  
**Priority**: P2 (Medium) - Significant performance improvement potential  

---

### Improvement 2: Rename Class or Add Disclaimer

**Problem**: Class name `WhisperCppCoreMLImplementation` is misleading since CoreML is not enabled  
**Impact**: User confusion; expectations of Neural Engine acceleration that doesn't happen  
**Location**: [PROJECT] `coreml.py:13`  

**Quick Fix** (docstring update):  
```python
class WhisperCppCoreMLImplementation(WhisperImplementation):
    """Whisper implementation using pywhispercpp (whisper.cpp).

    NOTE: Despite the class name, CoreML/Neural Engine acceleration is NOT enabled.
    The pywhispercpp binary from PyPI is compiled without CoreML support.
    Actual acceleration uses Metal GPU and Accelerate BLAS.
    """
```

**Better Fix** (rename class):  
```python
class WhisperCppImplementation(WhisperImplementation):
    """Whisper implementation using pywhispercpp (whisper.cpp) with Metal GPU acceleration."""
```

**Effort**: ~5 lines (docstring) or ~20 lines (rename with backward compat alias)  
**Priority**: P2 (Medium) - Improves clarity but doesn't affect functionality  

---

### Improvement 3: Add Model Size Selection Option

**Problem**: `large` model always maps to `large-v3-turbo-q5_0`, user cannot choose full `large-v3`  
**Impact**: Users wanting highest quality must modify code  
**Location**: [PROJECT] `coreml.py:37-44`  

**Recommended Fix**: Add variant parameter or expand mapping:  

```python
models_map = {
    "tiny": "tiny-q5_1",
    "base": "base-q5_1",
    "small": "small",
    "medium": "medium-q5_0",
    "large": "large-v3-turbo-q5_0",
    # Full-size variants
    "large-v3": "large-v3",
    "large-v3-q5": "large-v3-q5_0",
    "large-v3-turbo": "large-v3-turbo-q5_0",
}
```

**Effort**: ~10 lines  
**Priority**: P3 (Low) - Nice-to-have flexibility  

---

### Improvement 4: Remove Dead CoreML Code

**Problem**: Code checks for `.mlmodelc` files that will never be used  
**Impact**: Unnecessary warnings, confusing log messages  
**Location**: [PROJECT] `coreml.py:48-70`  

**Recommended Fix**: Remove CoreML-specific code since it has no effect:  

```python
def load_model(self, model_name: str) -> None:
    import pywhispercpp.model
    self._pywhispercpp = pywhispercpp
    pywhispercpp.model.logging = self.log

    models_map = {
        "tiny": "tiny-q5_1",
        "base": "base-q5_1",
        "small": "small",
        "medium": "medium-q5_0",
        "large": "large-v3-turbo-q5_0",
    }
    self.model_name = models_map.get(model_name, model_name)

    # Load the model (Metal GPU + BLAS acceleration is automatic)
    self._model = self._pywhispercpp.model.Model(
        self.model_name,
        models_dir=str(self.models_dir),
        n_threads=self.n_threads,
    )
```

**Effort**: ~20 lines removed  
**Priority**: P3 (Low) - Cleanup, no functional impact  

---

## Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| P2 | Build pywhispercpp with CoreML support | Large | ~3x speedup | Future |
| P3 | Add model variant selection options | ~10 lines | Flexibility | Proposed |
| P2 | Add clarifying docstring about no CoreML | ~5 lines | Clarity | Proposed |
| P3 | Remove dead CoreML checking code | ~20 lines | Cleanup | Proposed |

---

## Implementation Order Recommendation

### Phase 1: Performance (Future)
- [ ] Investigate building pywhispercpp with CoreML support
- [ ] If CoreML build available, add proper `.mlmodelc` model download
- [ ] Benchmark CoreML vs Metal-only performance

### Phase 2: Documentation & Clarity
- [ ] Add docstring clarifying Metal/BLAS acceleration (not CoreML)
- [ ] Update benchmark display name from "whisper.cpp" to "whisper.cpp (Metal)"

### Phase 3: Code Cleanup
- [ ] Remove unused CoreML checking code
- [ ] Remove `os.environ["WHISPER_COREML"] = "1"` (no effect)
- [ ] Consider renaming class to `WhisperCppImplementation`


---

## Sources

- [whisper.cpp GitHub Repository](https://github.com/ggml-org/whisper.cpp)
- [whisper.cpp CoreML Support Documentation](https://panjas.com/blog/2024-11-26/coreml-models-for-whisper-cpp)
- [PyTorch SDPA Documentation](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
