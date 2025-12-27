# Model Caching and Storage in Whisper Implementations

## Overview

Different Whisper implementations use different model caching strategies based on their framework requirements. Understanding these differences is crucial for troubleshooting model download issues and optimizing storage.

## Cache Locations by Implementation

### HuggingFace-Based Implementations

**Location:** `~/.cache/huggingface/hub/`

**Used by:**
- mlx-whisper
- faster-whisper
- insanely-fast-whisper
- lightning-whisper-mlx
- parakeet-mlx

**Format:** Raw PyTorch/TensorFlow weights, model configs, tokenizers

**Why:**
- Standard for Python-based HuggingFace transformers ecosystem
- Cross-platform compatibility
- Shared cache across all HuggingFace models
- Automatic versioning via git-like snapshots

**Example structure:**
```
~/.cache/huggingface/hub/
├── models--mlx-community--whisper-small-mlx-4bit/
│   ├── blobs/
│   ├── refs/
│   └── snapshots/
│       └── {hash}/
│           ├── config.json
│           ├── model.safetensors
│           └── tokenizer.json
```

### Project Local Cache Implementations

**Location:** `<project_root>/models/`

**Used by:**
- whisper-mps
- whisper.cpp (for GGML models)

**Format:** PyTorch `.pt` files (whisper-mps) or GGML quantized models (whisper.cpp)

**Why:**
- Direct control over model storage location
- Simpler path management for benchmarking
- Isolated from system-wide HuggingFace cache
- Easy to verify and manage models within project

**Example structure:**
```
~/projects/mac-whisper-speedtest/models/
├── tiny.pt              (72 MB)
├── small.pt             (461 MB)
├── medium.pt            (1.5 GB)
├── large-v3.pt          (2.9 GB)
├── ggml-small.bin       (466 MB - whisper.cpp)
└── ggml-small-encoder.mlmodelc/  (CoreML encoder for whisper.cpp)
```

#### WhisperMPS Download Details

**Download Source:** `https://openaipublic.azureedge.net/main/whisper/models/`

**Important:** whisper-mps does **NOT** download from HuggingFace. Instead, it downloads single `.pt` files directly from OpenAI's Azure CDN, matching the original OpenAI Whisper implementation.

**Download Mechanism:**
```python
from whisper_mps.whisper.load_models import load_model

# Downloads from Azure CDN to specified directory
model = load_model(
    name="small",
    download_root="~/projects/mac-whisper-speedtest/models/"
)
```

**File Format:** Each model is a single PyTorch checkpoint file (`.pt`) containing:
- Model weights (encoder + decoder)
- Model configuration
- Tokenizer vocabulary
- Training metadata

**Download Pattern:**
```
URL: https://openaipublic.azureedge.net/main/whisper/models/{sha256}/{model}.pt
Example: .../9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
```

The SHA256 hash in the URL is used for integrity verification after download.

**Why Not HuggingFace:**
- whisper-mps is based on OpenAI's original Whisper implementation
- Uses the original PyTorch checkpoint format, not HuggingFace's transformers format
- Downloads are faster (single file vs. multiple files)
- Direct compatibility with OpenAI's model weights

### Native macOS Implementations

**Location:** varies by implementation

**Used by:**
- **WhisperKit**: `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/`
- **FluidAudio**: `~/Library/Application Support/FluidAudio/Models/`

**Format:** Compiled CoreML models (`.mlmodelc` bundles)

**Why:**
- Apple's convention for application-specific data
- CoreML models optimized for Apple Neural Engine
- Platform-specific compilation required
- Better integration with macOS security and permissions

**Example structure:**
```
~/Library/Application Support/FluidAudio/Models/
└── parakeet-tdt-0.6b-v3-coreml/
    ├── Encoder.mlmodelc/       (425 MB - compiled CoreML)
    ├── Decoder.mlmodelc/       (23 MB)
    ├── Preprocessor.mlmodelc/  (520 KB)
    ├── JointDecision.mlmodelc/ (12 MB)
    ├── parakeet_vocab.json
    └── config.json
```

### Implementation-Specific Notes

**whisper.cpp (via pywhispercpp):**
- **This project's implementation**: Uses project local cache (`<project_root>/models/`)
- **Standalone whisper.cpp**: May use `~/Library/Caches/whisper/` (macOS) or `~/.cache/whisper/` (Linux)
- **Format**: GGML quantized models (`.bin` files) + optional CoreML encoder (`.mlmodelc`)
- **Why**: Optimized binary format for C++ inference with optional CoreML acceleration

## The CoreML Compilation Difference

### Why Native macOS Frameworks Use Different Caches

CoreML models require a **compilation step** that transforms raw neural network weights into Apple-optimized `.mlmodelc` bundles:

```
Raw PyTorch/TensorFlow weights
         ↓
    Conversion to CoreML format (.mlpackage)
         ↓
    Compilation for target hardware (.mlmodelc)
         ↓
    Optimized for Apple Neural Engine
```

This compilation:
- Optimizes for specific Apple Silicon chips (M1/M2/M3/M4)
- Enables ANE (Apple Neural Engine) acceleration
- Reduces runtime memory usage
- Improves inference latency

**This is why** WhisperKit and FluidAudio can't directly use HuggingFace cached models - they need the compiled `.mlmodelc` format.

## Model Alignment Verification

### Verification Details for Small Model

The following table shows the verified alignment between `check-models` command and actual benchmark behavior for the `small` model size. All implementations have been verified to download and use the correct models.

| Implementation | Model Used | Repo ID | Status |
|---|---|---|---|
| **FasterWhisper** | small | `Systran/faster-whisper-small` | ✅ Verified in HF cache |
| **MLXWhisper** | small | `mlx-community/whisper-small-mlx-4bit` | ✅ Verified in HF cache |
| **WhisperCppCoreML** | small | Local: `ggml-small.bin` + CoreML encoder | ✅ Verified (633.3 MB) |
| **InsanelyFastWhisper** | small | `openai/whisper-small` | ✅ Verified in HF cache |
| **LightningWhisperMLX** | small | `mlx-community/whisper-small-mlx-q4` | ✅ Verified in HF cache |
| **ParakeetMLX** | small | `mlx-community/parakeet-tdt-0.6b-v2` | ✅ Verified in HF cache (2.3 GB) |
| **FluidAudioCoreML** | ALL sizes | `FluidInference/parakeet-tdt-0.6b-v3-coreml` | ✅ Uses same model for all sizes |
| **WhisperKit** | small | Native CoreML (via bridge) | ✅ Verified (463.9 MB) |
| **WhisperMPS** | small | Local: `small.pt` | ✅ Verified (461.2 MB) |

### Key Findings

1. **All HuggingFace models matched**: Every repo ID returned by `get_model_info()` exists in the HuggingFace cache and matches what `load_model()` actually downloads
2. **ParakeetMLX naming clarified**: Uses `parakeet-tdt-0.6b-v2` (600M params, 2.3GB) for all small/medium/large sizes - this is intentional as Parakeet is a universal ASR model
3. **FluidAudio is size-agnostic**: Always uses the same `parakeet-tdt-0.6b-v3-coreml` model regardless of requested size, as it's optimized for real-time streaming
4. **Local cache models verified**: WhisperCppCoreML, WhisperKit, and WhisperMPS all have correct local cache paths and expected sizes
5. **Model versioning handled**: For `large` model, implementations correctly map to their preferred versions (`large-v3-turbo` for InsanelyFast, `large-v3` for others)
6. **Download sources verified**:
   - HuggingFace Hub: FasterWhisper, MLX, InsanelyFast, Lightning, Parakeet
   - OpenAI Azure CDN: WhisperMPS
   - Native Swift bridges: WhisperKit, FluidAudio
   - Manual installation: WhisperCppCoreML (GGML files)

### Verification Details for Tiny Model

The following table shows the verified alignment between `check-models` command and actual benchmark behavior for the `tiny` model size. This investigation revealed 3 critical issues that were successfully resolved.

| Implementation | Expected Model | Expected Repo ID | Status Before Fix | Issue Found | Fix Applied |
|---------------|----------------|------------------|-------------------|-------------|-------------|
| **FasterWhisper** | tiny | `Systran/faster-whisper-tiny` | ❌ Missing | Model exists in custom cache (`models/`) but check-models only scanned default HF cache | Updated `check_models.py` to scan custom models directory |
| **MLXWhisper** | mlx-community/whisper-tiny-mlx-q4 | `mlx-community/whisper-tiny-mlx-q4` | ❌ Missing | Code expected non-quantized version but quantized q4 version was downloaded | Updated model mapping to use `mlx-community/whisper-tiny-mlx-q4` |
| **WhisperCppCoreML** | tiny-q5_1 + CoreML | None | ⚠️ Incomplete (46.4MB vs 150MB) | Expected size was wrong (150MB vs actual 50MB) | Updated expected size from 150MB to 50MB |
| **InsanelyFastWhisper** | tiny | `openai/whisper-tiny` | ✅ Complete | No issues | No changes needed |
| **LightningWhisperMLX** | mlx-community/whisper-tiny-mlx-q4 | `mlx-community/whisper-tiny-mlx-q4` | ✅ Complete | No issues | No changes needed |
| **ParakeetMLX** | mlx-community/parakeet-tdt-0.6b-v2 | `mlx-community/parakeet-tdt-0.6b-v2` | ✅ Complete | No issues | No changes needed |
| **FluidAudioCoreML** | parakeet-tdt-0.6b-v3-coreml | `FluidInference/parakeet-tdt-0.6b-v3-coreml` | ✅ Complete | No issues | No changes needed |
| **WhisperKit** | tiny | None | ✅ Complete | No issues | No changes needed |
| **WhisperMPS** | tiny | None | ✅ Complete | No issues | No changes needed |

### Key Findings from Tiny Model Investigation

1. **Custom Cache Directory Support**: `check_models.py` was only scanning the default HuggingFace cache (`~/.cache/huggingface/hub/`) but some implementations like FasterWhisper and MLXWhisper use a custom cache directory (`models/`). Fixed by updating `_verify_hf_model()` to scan both locations.

2. **MLX Quantized Model Update**: MLXWhisper was configured to use the non-quantized `mlx-community/whisper-tiny-mlx` model but a quantized version `mlx-community/whisper-tiny-mlx-q4` exists and was previously downloaded. Updated the model mapping to use the q4 version for better performance.

3. **CoreML Expected Size Correction**: WhisperCppCoreML expected size was set to 150MB but the actual quantized model (tiny-q5_1 + CoreML encoder) is only ~50MB total (31MB GGML + 16MB CoreML). Updated expected size to match reality.

4. **All Models Now Aligned**: After fixes, all 9 implementations show as "ready" with correct model references and sizes matching what's actually used by `load_model()`.

5. **Verification Method Consistency**: 5 implementations use HuggingFace verification, 4 use local file size verification, maintaining consistency with their respective download strategies.

6. **Model Fallback Chains Work Correctly**: FasterWhisper's fallback chain for tiny model (`["tiny"]`) works as expected with the primary model being found.

### Verification Details for Large Model

The following table shows the verified alignment between `check-models` command and actual benchmark behavior for the `large` model size. This investigation revealed 1 critical issue that was successfully resolved.

| Implementation | Expected Model | Expected Repo ID | Status Before Fix | Issue Found | Fix Applied |
|---------------|----------------|------------------|-------------------|-------------|-------------|
| **FasterWhisper** | large-v3-turbo | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | ❌ Missing | Repo ID mismatch: expected `Systran/faster-whisper-large-v3-turbo` but actual is `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | Updated repo ID mapping for large-v3-turbo |
| **MLXWhisper** | mlx-community/whisper-large-v3-turbo | `mlx-community/whisper-large-v3-turbo` | ✅ Complete | No issues | No changes needed |
| **WhisperCppCoreML** | large-v3-turbo-q5_0 + CoreML | None | ✅ Complete (1763.5 MB) | No issues (previously fixed) | No changes needed |
| **InsanelyFastWhisper** | large | `openai/whisper-large-v3-turbo` | ✅ Complete | No issues (previously fixed) | No changes needed |
| **LightningWhisperMLX** | mlx-community/whisper-large-v3-mlx | `mlx-community/whisper-large-v3-mlx` | ✅ Complete | No issues | No changes needed |
| **ParakeetMLX** | mlx-community/parakeet-tdt-0.6b-v2 | `mlx-community/parakeet-tdt-0.6b-v2` | ✅ Complete | No issues | No changes needed |
| **FluidAudioCoreML** | parakeet-tdt-0.6b-v3-coreml | `FluidInference/parakeet-tdt-0.6b-v3-coreml` | ✅ Complete | No issues | No changes needed |
| **WhisperKit** | large-v3 | None | ✅ Complete | No issues | No changes needed |
| **WhisperMPS** | large | None | ✅ Complete (2944.3 MB) | No issues (previously fixed) | No changes needed |

### Key Findings from Large Model Investigation

1. **FasterWhisper Repository Source**: The faster-whisper library downloads the large-v3-turbo model from `mobiuslabsgmbh/faster-whisper-large-v3-turbo` instead of the usual `Systran/faster-whisper-{model}` pattern. This is because the large-v3-turbo model is newer and Systran doesn't host it - the community has created it under the mobiuslabsgmbh account.

2. **Custom Cache Scanning Works**: Our earlier fix to scan both default HF cache and custom models directory correctly found the mobiuslabsgmbh model in the custom cache at `models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`.

3. **All Other Implementations Verified**: The remaining 8 implementations were already correctly aligned - no issues found with their repo IDs, model names, or cache paths.

4. **Previous Fixes Still Valid**: The fixes applied during the earlier large model investigation (InsanelyFastWhisper repo ID and WhisperMPS path handling) are still working correctly.

5. **Model Sizes Consistent**: All model sizes match expected values and actual downloaded sizes (ranging from 460 MB for FluidAudio to 2.9 GB for WhisperKit/WhisperMPS).

## The FluidAudio Model Cache Issue

### What Was Happening

FluidAudio has a **two-stage download process** that was partially failing:

**Stage 1: Download to HuggingFace Cache** ✅ **WORKS**
```bash
Models downloaded to:
~/.cache/huggingface/hub/models--FluidInference--parakeet-tdt-0.6b-v3-coreml/
```

**Stage 2: Copy to Application Support** ❌ **FAILS**
```bash
Should copy to:
~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/

Actually copies:
- Only Encoder.mlmodelc (incomplete - 4 KB instead of 425 MB)
- Only Decoder.mlmodelc (23 MB - appears complete)
- Missing: Preprocessor.mlmodelc, JointDecision.mlmodelc, vocab, config
```

### Root Cause

FluidAudio's internal model management code (in `AsrModels.downloadAndLoad()`) has a bug where:

1. It successfully downloads complete models to HuggingFace cache
2. It attempts to copy/symlink them to Application Support
3. **The copy process fails or is incomplete** (likely due to file permissions, race conditions, or interrupted downloads)
4. Framework expects models in Application Support and can't find them
5. Framework attempts re-download on next run
6. Cycle repeats indefinitely

### Why It Went Undetected

- **First-time users:** Download appears to work (files appear in HF cache)
- **Error is silent:** No clear error message, just hangs indefinitely
- **Timeout mistaken for slow download:** Users assume it's still downloading
- **Works in developer environment:** FluidInference developers may have manually placed files

## Solution: Manual Model Copy

### Quick Fix Script

We've created a helper script that automates the manual copy process:

```bash
# Run this script after FluidAudio downloads models
./tools/fluidaudio-bridge/fix_models.sh
```

The script:
1. Checks HuggingFace cache for complete model files
2. Verifies all required files are present
3. Copies models to Application Support
4. Validates file sizes
5. Supports both v2 and v3 model versions

### Manual Process (if script fails)

**For FluidAudio v0.8.0+ (uses v3 models):**

```bash
# Source directory (HuggingFace cache)
SRC=~/.cache/huggingface/hub/models--FluidInference--parakeet-tdt-0.6b-v3-coreml/snapshots/*/

# Destination directory (Application Support)
DEST=~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/

# Create destination and copy all required files
mkdir -p "$DEST"
cp -r "$SRC/Encoder.mlmodelc" "$DEST/"
cp -r "$SRC/Decoder.mlmodelc" "$DEST/"
cp -r "$SRC/Preprocessor.mlmodelc" "$DEST/"
cp -r "$SRC/JointDecision.mlmodelc" "$DEST/"
cp "$SRC/parakeet_vocab.json" "$DEST/"
cp "$SRC/config.json" "$DEST/"

# Verify
du -sh "$DEST"/*
# Should show:
# 425M  Encoder.mlmodelc
# 23M   Decoder.mlmodelc
# 12M   JointDecision.mlmodelc
# 520K  Preprocessor.mlmodelc
```

**For FluidAudio v0.1.0 (uses v2 models):**

Replace `v3-coreml` with `v2-coreml` in paths above.

## Verification

After copying models, verify FluidAudio works:

```bash
# Test the bridge
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge \
  tools/whisperkit-bridge/.build/checkouts/WhisperKit/Tests/WhisperKitTests/Resources/jfk.wav \
  --format json

# Should output transcription in ~0.08 seconds:
{
  "text": "And so, my fellow Americans, ask not what your country can do for you...",
  "transcription_time": 0.079220056533813477
}
```

## Performance After Fix

With complete models, FluidAudio achieves **excellent performance**:

- **Transcription time:** ~0.08 seconds for 11-second audio
- **Real-time factor:** ~138x (processes 1 second of audio in 0.007 seconds)
- **Faster than WhisperKit:** 0.08s vs 0.43s (81% faster!)

## Best Practices

### For Users

1. **After installing FluidAudio bridge:**
   ```bash
   cd tools/fluidaudio-bridge
   swift build -c release
   ./fix_models.sh  # Run the fix script
   ```

2. **Check model integrity:**
   ```bash
   # Encoder should be ~425 MB, NOT 4 KB
   du -sh ~/Library/Application\ Support/FluidAudio/Models/*/Encoder.mlmodelc
   ```

3. **If models are incomplete:**
   - Delete Application Support cache
   - Run `fix_models.sh` to copy from HuggingFace cache
   - Or manually download and place models

### For Developers

1. **When adding new model-based implementations:**
   - Document expected cache location
   - Provide verification commands
   - Test on clean system (no pre-downloaded models)
   - Handle partial/corrupted downloads gracefully

2. **When troubleshooting model issues:**
   - Check BOTH HuggingFace cache AND Application Support
   - Verify file sizes (corrupted files often have suspiciously small sizes)
   - Check file permissions
   - Look for symlinks that might be broken

## WhisperKit Model Caching Details

### Cache Location

**Primary Storage:** `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/`

**Secondary Cache (if using MacWhisper app):**
`~/Library/Application Support/MacWhisper/models/whisperkit/models/argmaxinc/whisperkit-coreml/`

**Temporary Downloads:** `/private/var/folders/.../T/CFNetworkDownload_*.tmp`

**Compiled Model Cache (Apple Neural Engine):**
`~/Library/Caches/whisperkit-bridge/com.apple.e5rt.e5bundlecache/`

### Model Structure

Each WhisperKit model contains three CoreML model bundles:

```
~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-{model}/
├── config.json
├── generation_config.json
├── AudioEncoder.mlmodelc/
│   ├── weights/
│   │   └── weight.bin          (largest file, e.g., ~168M for small, ~600M+ for large-v3)
│   ├── model.mil
│   ├── model.mlmodel
│   ├── metadata.json
│   └── coremldata.bin
├── TextDecoder.mlmodelc/
│   ├── weights/
│   │   └── weight.bin          (large file, e.g., ~293M for small, ~800M+ for large-v3)
│   ├── model.mil
│   ├── model.mlmodel
│   └── metadata.json
└── MelSpectrogram.mlmodelc/
    ├── weights/
    │   └── weight.bin          (~365K, consistent across models)
    └── model.mil
```

### Known Issue: Incomplete Downloads Leave Partial Models

**Symptom:** WhisperKit hangs/times out when downloading large models, even after multiple attempts

**Root Cause:** Large model downloads (large-v3: ~1.4GB) timeout before completion, leaving partial model structures. On retry, WhisperKit fails to resume because the `weights/` subdirectories don't exist.

**Error Message (in logs):**
```
"CFNetworkDownload_XXX.tmp" couldn't be moved to "weights" because either
the former doesn't exist, or the folder containing the latter doesn't exist.
```

**Why This Happens:**
1. **Initial download attempt:** Large models (1.4GB+) timeout during download (common with slow networks)
2. **Partial state created:** WhisperKit creates model structure (`AudioEncoder.mlmodelc/`, `TextDecoder.mlmodelc/`, etc.) and downloads some files (`model.mil`, `metadata.json`)
3. **Timeout occurs:** Before weight files complete downloading, the process times out
4. **Retry fails:** On next attempt, WhisperKit tries to download weight files to temp directory
5. **Move fails:** Attempts to move `/tmp/CFNetworkDownload_*.tmp` → `.../weights/weight.bin` but `weights/` directory was never created during the incomplete first attempt
6. **Temp file orphaned:** Download succeeds but move fails, leaving large temp files orphaned
7. **Cycle repeats:** Each retry creates more orphaned temp files without fixing the underlying issue

**Evidence:** Check `/private/var/folders/.../T/` for multiple orphaned `CFNetworkDownload_*.tmp` files (100-600MB each) from repeated failed attempts

**Why Small/Tiny Models Don't Have This Issue:**
- Smaller downloads (<500MB) complete before timeout
- Complete downloads create proper model structure including `weights/` directories
- Once downloaded, subsequent runs use cached models

### Workaround for WhisperKit Download Issue

**Before running WhisperKit with large models:**

```bash
# Create the missing weights directories for large-v3
mkdir -p ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/AudioEncoder.mlmodelc/weights
mkdir -p ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/TextDecoder.mlmodelc/weights
mkdir -p ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/MelSpectrogram.mlmodelc/weights

# Clean up stuck downloads
rm /private/var/folders/*/T/CFNetworkDownload_*.tmp 2>/dev/null

# Now run WhisperKit
```

**After successful download, verify:**
```bash
du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3
# Should show ~1.4GB for large-v3

# Check that weights exist
ls -lh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*/weights/
```

**Expected sizes:**
- tiny: ~76M
- small: ~467M
- large-v3: ~1.4GB
- distil-large-v3: ~229M

## Comparison with WhisperKit

**WhisperKit has issues with large model downloads:**

1. **Directory creation bug:** Fails to create `weights/` subdirectories before moving downloaded files
2. **Silent failures:** Downloads complete but move fails, leaving orphaned temp files
3. **No retry logic:** Doesn't detect partial downloads and retry
4. **Timeout confusion:** User sees timeout but real issue is file move failure

This demonstrates the importance of **robust model management** in production frameworks.

## Future Improvements

### For FluidAudio Framework

Report this issue to [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio/issues):

1. **Fix copy logic:** Ensure all files copied from HF cache to Application Support
2. **Add validation:** Verify model completeness after download
3. **Better errors:** Clear error messages when models are missing/incomplete
4. **Resume capability:** Continue partial downloads instead of restarting
5. **Direct Application Support download:** Skip HF cache intermediate step

### For This Project

- ✅ Created `fix_models.sh` helper script
- ✅ Documented the issue comprehensively
- ✅ Provided manual workaround
- 🔄 Consider adding automatic fix to installation process
- 🔄 Add model verification to benchmark startup
- 🔄 Provide clear error messages when models are incomplete

## Related Documentation

- [FluidAudio Final Status](FLUIDAUDIO_FINAL_STATUS.md) - Investigation results
- [FluidAudio Issue](FLUIDAUDIO_ISSUE.md) - Technical details
- [Benchmark Results](BENCHMARK_RESULTS.md) - Performance comparisons

## Summary

**Key Takeaways:**

1. **Different frameworks use different caches** based on model format requirements
2. **CoreML models require compilation**, can't use raw HuggingFace weights directly
3. **FluidAudio downloads work** but the copy-to-Application-Support step fails
4. **Manual copy from HF cache** fixes the issue completely
5. **Use `fix_models.sh`** to automate the workaround
6. **FluidAudio is extremely fast** once models are properly installed (0.08s transcription time!)
