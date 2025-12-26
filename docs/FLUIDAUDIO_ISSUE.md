# FluidAudio Swift Bridge Issue

## Problem Summary

The FluidAudio CoreML implementation consistently times out (300+ seconds) and never completes transcription, making it unusable in the benchmark.

**Status:** ❌ Currently broken
**Severity:** Critical - completely prevents FluidAudio from functioning

## Root Cause

The Swift bridge executable re-downloads the FluidAudio models on **every single transcription call** instead of caching them for reuse.

### Problematic Code

**File:** `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift`
**Line:** 46

```swift
// Current implementation (WRONG):
let models = try await AsrModels.downloadAndLoad()
try await asrManager.initialize(models: models)
```

This code is executed inside the main transcription function, which means:
1. User calls Python benchmark
2. Python calls Swift bridge via subprocess
3. Swift bridge starts model download from scratch
4. Download takes 5+ minutes (multi-GB models)
5. Python times out waiting for subprocess
6. Transcription never completes

## Timeline of Investigation

### Initial Symptom
- FluidAudio always times out regardless of timeout value (tested with 120s, 300s)
- No transcription ever produced
- Benchmark shows `inf` time and error message

### Discovery Process
1. Noticed timeout comment said "5 minute timeout" but code had 120s
2. Fixed timeout to 300s - still timed out
3. User reported: "I have never gotten FluidAudio to work, it always seems to timeout"
4. Investigated Swift bridge code
5. Found `AsrModels.downloadAndLoad()` being called on every transcription
6. Confirmed this is the root cause

### Python-Side Fixes Applied
✅ **Fixed**: Timeout increased from 120s to 300s (line 108)
✅ **Fixed**: Removed debug message "RYAN - gettin' them parakeet params..." (line 199)
✅ **Fixed**: Corrected `get_params()` to return FluidAudio model instead of Parakeet model (lines 197-203)
✅ **Fixed**: Changed to use `self.model_name` instead of hardcoded model name

### Swift-Side Fix Required
❌ **Not Fixed**: Model caching in Swift bridge

## Required Solution

### What Needs to Change

The Swift bridge needs to implement model caching so models are downloaded once and reused across multiple invocations.

### Proposed Implementation

**Option 1: Cache models on disk and check before download**

```swift
// Suggested fix (pseudo-code):
func loadOrDownloadModels() async throws -> Models {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    let modelCacheDir = cacheDir.appendingPathComponent("FluidAudio/models")

    // Check if models are already cached
    if modelsExistInCache(modelCacheDir) {
        print("Loading cached models...")
        return try await AsrModels.loadFromCache(modelCacheDir)
    } else {
        print("Downloading models (first run)...")
        let models = try await AsrModels.downloadAndLoad()
        try saveModelsToCache(models, at: modelCacheDir)
        return models
    }
}

// In main:
let models = try await loadOrDownloadModels()
try await asrManager.initialize(models: models)
```

**Option 2: Use FluidAudio's built-in caching (if available)**

Check FluidAudio framework documentation for built-in model caching mechanisms:
```swift
// If FluidAudio supports it:
let models = try await AsrModels.loadCached() ?? AsrModels.downloadAndLoad()
```

**Option 3: Persistent subprocess (advanced)**

Instead of calling the Swift bridge as a one-shot subprocess, keep it running as a daemon:
- Python starts bridge once at initialization
- Bridge loads models once
- Python sends transcription requests via IPC (stdin/stdout or socket)
- Bridge remains alive across multiple transcriptions

## Testing the Fix

### Before Fix
```bash
.venv/bin/mac-whisper-speedtest --model small
# Result: FluidAudio times out after 300 seconds
```

### After Fix (Expected)
```bash
# First run (downloads models)
.venv/bin/mac-whisper-speedtest --model small
# FluidAudio should complete in <2 seconds

# Subsequent runs (uses cached models)
.venv/bin/mac-whisper-speedtest --model small
# FluidAudio should complete in <1 second
```

## Expected Performance

According to FluidAudio documentation, it's designed for real-time streaming with approximately **110x RTF** (Real-Time Factor) on M4 Pro.

For an 11-second audio clip:
- **Expected time:** ~0.1 seconds (110x faster than real-time)
- **Current time:** timeout (infinite)

Once fixed, FluidAudio should be the **fastest implementation**, potentially beating WhisperKit's current 0.43s.

## Impact

### Current State
- FluidAudio completely unusable
- Benchmark always shows timeout error
- Cannot compare FluidAudio against other implementations
- Real-time streaming use case cannot be tested

### After Fix
- FluidAudio should be fastest or near-fastest implementation
- Real-time streaming benchmarks possible
- Complete implementation comparison available
- Users can make informed decisions about which implementation to use

## Related Files

### Python Implementation
- `src/mac_whisper_speedtest/implementations/fluidaudio_coreml.py`
  - ✅ Python-side fixes already applied
  - Calls Swift bridge via subprocess
  - Handles JSON response from bridge

### Swift Bridge
- `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift`
  - ❌ **Line 46: Model caching fix needed here**
  - Parses command-line arguments
  - Loads audio file
  - Calls FluidAudio framework
  - Returns JSON results

### Build System
- `tools/fluidaudio-bridge/Package.swift`
  - Swift Package Manager configuration
  - FluidAudio framework dependency

## Additional Notes

### Why This Wasn't Caught Earlier

1. **First-time setup delay is expected**: Users might assume initial download is normal
2. **Timeout hides the real issue**: Error message says "timeout" not "downloading models"
3. **No progress indicator**: Swift bridge doesn't log download progress to Python
4. **Subprocess isolation**: Each call is independent, no state persists

### Model Download Details

FluidAudio models are likely:
- Multi-GB in size (typical for ASR models)
- Downloaded from HuggingFace or similar
- CoreML format (.mlpackage or .mlmodel)
- Stored in user's cache directory

### Development Workflow

To fix this issue:
1. Open Swift bridge project: `cd tools/fluidaudio-bridge`
2. Edit `Sources/fluidaudio-bridge/main.swift`
3. Implement model caching logic
4. Build: `swift build -c release`
5. Test: `cd ../.. && .venv/bin/mac-whisper-speedtest --model small`
6. Verify FluidAudio completes without timeout

## References

- FluidAudio Framework: (check package dependencies for documentation URL)
- Swift Package Manager: https://swift.org/package-manager/
- CoreML Model Caching: https://developer.apple.com/documentation/coreml

## Investigation Results (2025-12-25)

### Actual Root Cause: Incomplete Model Downloads

After detailed investigation, the real problem was discovered:

**Cached models are incomplete:**
- Located in: `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/`
- **Only 2 of 4 required models exist:**
  - ✅ `Decoder.mlmodelc` - 15 MB (appears complete)
  - ❌ `Encoder.mlmodelc` - **4 KB** (severely incomplete, missing weights)
  - ❌ `Preprocessor.mlmodelc` - **Missing entirely**
  - ❌ `Joint.mlmodelc` - **Missing entirely**

**What's in the incomplete Encoder:**
- `analytics/coremldata.bin` (243 bytes) - metadata only
- **Missing:** `weights/weight.bin` (should be ~50-200 MB)
- **Missing:** `model.mil` (model definition)
- **Missing:** `metadata.json`

**Current Status:**
- FluidAudio version in use: 0.1.0 (resolved from "from: 0.0.3")
- Models re-download on every cache clear but remain incomplete
- No network activity during hang (no active download)
- Process stuck at `AsrModels.downloadAndLoad()` call
- Models appear to be pulled from a corrupted source

### Next Steps

1. **Monitor current download attempt** - waiting to see if models eventually complete
2. **Check FluidAudio 0.1.0 known issues** - may be a bug in this version
3. **Try upgrading to latest FluidAudio** - newer version may have fixes
4. **Manual model download** - check if models can be manually obtained
5. **Document as currently broken** - if no solution found

## Status Tracking

- [x] Issue identified and documented
- [x] Python-side fixes applied (timeout, get_params)
- [x] Root cause identified (incomplete model files)
- [x] Cache cleared for fresh download attempt
- [ ] Models successfully downloaded (in progress)
- [ ] Swift-side model caching implemented
- [ ] Tested with benchmark
- [ ] Performance validated (should be ~0.1-0.5s for 11s audio)
- [ ] Documentation updated with working results
