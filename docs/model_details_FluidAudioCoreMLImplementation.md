# Model Details: FluidAudioCoreMLImplementation

This document provides comprehensive documentation of the `FluidAudioCoreMLImplementation` benchmark execution flow, model handling, caching behavior, and CoreML configuration.

**Test Date:** 2026-01-12

---

## File Reference Legend

| Tag | Meaning | Modifiable? |
|-----|---------|-------------|
| **[PROJECT]** | File in this project's codebase (`src/mac_whisper_speedtest/`) | ✅ Yes |
| **[BRIDGE]** | File in the Swift bridge (`tools/fluidaudio-bridge/Sources/`) | ✅ Yes (owned by this repo) |
| **[LIBRARY]** | FluidAudio Swift Package (`.build/checkouts/FluidAudio/`) | ❌ No (external dependency, pulled from [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio)) |

**Important:** Files tagged [LIBRARY] are external dependencies. To modify them, you would need to:
1. Fork the FluidAudio repository
2. Make changes in your fork
3. Update the Package.swift dependency URL to point to your fork

---

## Key Questions Answered

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| 1 | **Does the Swift bridge re-download models on every transcription call?** | **NO** - `downloadAndLoad()` is called every time (subprocess architecture), but `modelsExist()` and file size checks prevent re-download if complete files exist | `AsrModels.swift:139`, `DownloadUtils.swift:266-272` |
| 2 | **Does it download to HuggingFace Hub cache (`~/.cache/huggingface/hub/`)?** | **NO** - Downloads directly to `~/Library/Application Support/FluidAudio/Models/` | `AsrModels.swift:204-213`, empirical verification |
| 3 | **Does it check HF Hub cache for existing models?** | **NO** - Only checks its own Application Support folder, never looks in HF Hub cache | Code analysis: no HF cache path references |
| 4 | **Is download folder same as runtime folder?** | **YES** - Identical location for both download and model loading | `DownloadUtils.swift:76-82`, `AsrModels.swift:176-177` |
| 5 | **Are model files converted/renamed after download?** | **NO** - Files are used as-is once downloaded (no post-processing) | `DownloadUtils.swift:308-317` moves temp→final directly |
| 6 | **Partial downloads: resume or wipe & restart?** | **NO RESUME** - Despite code structure suggesting resume, `performChunkedDownload` uses `URLSession.download()` which starts fresh. Complete files are skipped via size check. | `DownloadUtils.swift:277-283` (unused), line 342 |
| 7 | **Complete files: skip download?** | **YES** - File size verification: if `fileSize == expectedSize` → skip | `DownloadUtils.swift:266-272` |
| 8 | **Completeness checks on existing files?** | **TWO-LEVEL**: (1) File size matches expected, (2) `coremldata.bin` exists inside each `.mlmodelc` | `DownloadUtils.swift:266-268`, lines 110-118 |
| 9 | **Uses HuggingFace Hub API?** | **NO** - Uses direct HTTP calls to HF REST API (`huggingface.co/api/models/`) and direct URL downloads | `DownloadUtils.swift:182`, line 286 |
| 10 | **Is timeout sufficient for first-run download?** | **NO** - Python timeout 300s (5 min) << ~29 min download time; Swift timeout 1800s (30 min) is adequate | `fluidaudio_coreml.py:116`, `DownloadUtils.swift:17` |
| 11 | **Uses fallback chains for model selection?** | **NO** - Single hardcoded model, no alternatives | `AsrModels.swift:43`, `DownloadUtils.swift:27` |
| 12 | **Cache folder in `~/Library/Application Support/`?** | **YES** - `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/` | `AsrModels.swift:204-213` |
| 13 | **Model name mapping/hardcoding?** | **HARDCODED everywhere** - model_name param ignored entirely | See "Model Mapping Chain" section |
| 14 | **CPU or GPU based?** | **BOTH + Neural Engine** - Uses `.all` compute units (CPU + GPU + ANE) | `AsrModels.swift:121` |
| 15 | **Apple Silicon optimizations?** | **YES** - CoreML with Neural Engine, `allowLowPrecisionAccumulationOnGPU = true` | `AsrModels.swift:119-121` |
| 16 | **What is CoreML?** | Apple's ML framework for optimized inference on Apple Silicon | README.md, used throughout |
| 17 | **SDPA relevant?** | **NO** - PyTorch concept, not applicable to CoreML | Code analysis: no SDPA references |
| 18 | **Uses quantization?** | **PARTIALLY** - Model repo includes `ParakeetEncoder_4bit_par.mlmodelc` (4-bit quantized), but default uses full precision | HuggingFace repo file listing |

---

## Benchmark Execution Flow

### Command (Small Model):
```bash
.venv/bin/python3 test_benchmark2.py small 1 FluidAudioCoreMLImplementation
```

### Command (Large Model):
```bash
.venv/bin/python3 test_benchmark2.py large 1 FluidAudioCoreMLImplementation
```

**Important:** Both commands produce **identical results** because FluidAudio ignores the model_name parameter.

### Step-by-Step Flow

1. **Entry Point** — `test_benchmark2.py:88-94` [PROJECT]
   - Parses CLI args: `model="small"`, `runs=1`, `implementations="FluidAudioCoreMLImplementation"`
   - Loads audio from `tests/jfk.wav` (16kHz mono)

2. **BenchmarkConfig Creation** — `test_benchmark2.py:77-82` [PROJECT]
   ```python
   config = BenchmarkConfig(
       model_name="small",  # IGNORED by FluidAudio
       implementations=[FluidAudioCoreMLImplementation],
       num_runs=1,
       audio_data=whisper_audio,
   )
   ```

3. **Implementation Instantiation** — `fluidaudio_coreml.py:21-44` [PROJECT]
   - Checks platform is macOS (`Darwin`)
   - Locates Swift bridge at `tools/fluidaudio-bridge/.build/release/fluidaudio-bridge`

4. **Model Loading** — `fluidaudio_coreml.py:46-81` [PROJECT]
   - **IGNORES the `model_name` parameter completely**
   - Logs: "fluidaudio-bridge uses hard-coded model: parakeet-tdt-0.6b-v2-coreml"
   - Runs `fluidaudio-bridge --help` to verify bridge is working
   - Does NOT download or load model at this stage

5. **Transcription Call** — `fluidaudio_coreml.py:85-159` [PROJECT]
   - Preprocesses audio (float32, mono, normalized, min 0.1s length)
   - Writes audio to temp WAV file
   - Calls subprocess: `fluidaudio-bridge <temp.wav> --format json`
   - **Timeout: 300 seconds (5 minutes)** — INSUFFICIENT for first run!

6. **Swift Bridge Execution** — `main.swift:20-75` [BRIDGE]
   - Loads audio file via AVFoundation
   - Creates `AsrManager` with config
   - **Calls `AsrModels.downloadAndLoad()`** — downloads if needed, loads models
   - Runs transcription via TDT decoder
   - Returns JSON with text, transcription_time, processing_time, language

7. **Model Download (if needed)** — `AsrModels.swift:172-178` [LIBRARY]
   ```swift
   public static func downloadAndLoad(...) async throws -> AsrModels {
       let targetDir = try await download(to: directory)  // Downloads if needed
       return try await load(from: targetDir, ...)        // Loads CoreML models
   }
   ```

8. **Download Logic** — `AsrModels.swift:130-170` [LIBRARY]
   - Checks `modelsExist(at: targetDir)` — looks for 5 required files
   - If all exist, returns immediately (no download)
   - If any missing, initiates download from HuggingFace via `DownloadUtils`
   - **Per-file granularity**: Only missing files are downloaded (see step 9)

9. **File-Level Download** — `DownloadUtils.swift:252-324` [LIBRARY]
   - For each file: check if exists AND size matches expected
   - Complete file: **SKIP** (size verification)
   - Missing/incomplete file: **DOWNLOAD FRESH** (no resume despite code structure)

10. **Model Loading with Auto-Recovery** — `DownloadUtils.swift:35-56` [LIBRARY]
    ```swift
    public static func loadModels(...) async throws -> [String: MLModel] {
        do {
            return try await loadModelsOnce(...)  // 1st attempt
        } catch {
            // If load fails, WIPE cache and re-download
            try? FileManager.default.removeItem(at: repoPath)
            return try await loadModelsOnce(...)  // 2nd attempt
        }
    }
    ```

11. **Result Parsing** — `fluidaudio_coreml.py:123-142` [PROJECT]
    - Parses JSON output from bridge
    - Attaches `_transcription_time` attribute for accurate benchmarking
    - Returns `TranscriptionResult`

---

## Summary Table

| Item | Value |
|------|-------|
| **Requested Model** | `small` (or any — **IGNORED**) |
| **Actual Model** | `parakeet-tdt-0.6b-v2-coreml` (hardcoded) |
| **HuggingFace Repo** | `FluidInference/parakeet-tdt-0.6b-v2-coreml` |
| **Download URL Pattern** | `https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml/resolve/main/<file>` |
| **Cache/Runtime Location** | `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/` |
| **Uses HuggingFace Hub Cache** | **NO** — downloads directly to Application Support |
| **Uses HuggingFace Hub API (Python)** | **NO** — uses direct HTTP REST API calls |
| **Total Model Size** | ~2.4 GB |
| **Download Time (First Run)** | ~22 minutes (empirically verified) |
| **Python Subprocess Timeout** | 300 seconds (5 minutes) — **INSUFFICIENT** |
| **Swift-side Timeout** | 1800 seconds (30 minutes) |

---

## Model Files Downloaded

| File | Size | Description |
|------|------|-------------|
| `ParakeetEncoder.mlmodelc` | 1.1 GB | Main encoder model (full precision) |
| `ParakeetEncoder_v2.mlmodelc` | 577 MB | Encoder v2 variant |
| `Encoder.mlmodelc` | 433 MB | Legacy encoder |
| `ParakeetEncoder_4bit_par.mlmodelc` | 305 MB | **4-bit quantized encoder** |
| `ParakeetDecoder.mlmodelc` | 27 MB | Decoder model |
| `Decoder.mlmodelc` | 14 MB | Legacy decoder |
| `RNNTJoint.mlmodelc` | 4.0 MB | RNN-T joint model |
| `JointDecision.mlmodelc` | 4.0 MB | Joint decision model |
| `Melspectrogram_v2.mlmodelc` | 620 KB | Mel spectrogram v2 |
| `Melspectogram.mlmodelc` | 612 KB | Mel spectrogram (legacy) |
| `Preprocessor.mlmodelc` | 332 KB | Audio preprocessor |
| `parakeet_vocab.json` | 20 KB | Vocabulary file |
| `config.json` | 4.0 KB | Model config |

**Required for operation** (checked by `modelsExist()`):
- `Melspectogram.mlmodelc`
- `ParakeetEncoder.mlmodelc`
- `ParakeetDecoder.mlmodelc`
- `RNNTJoint.mlmodelc`
- `parakeet_vocab.json`

---

## Model Mapping Chain (Hardcoded Throughout)

The model_name parameter is **IGNORED at every level**:

### Level 1: Python Wrapper — `fluidaudio_coreml.py:46-64` [PROJECT]
```python
def load_model(self, model_name: str) -> None:
    """Load the FluidAudio model (via Swift bridge).

    Args:
        model_name: FluidAudio ignores this! Hardcoded to "parakeet-tdt-0.6b-v2-coreml"
    """
    self.log.info(f"Just kidding! fluidaudio-bridge ignores model_name parameter!")
    self.log.info(f"fluidaudio-bridge uses hard-coded model: parakeet-tdt-0.6b-v2-coreml")
```

### Level 2: Swift Bridge — `main.swift:46` [BRIDGE]
```swift
let models = try await AsrModels.downloadAndLoad()  // No model parameter!
```

### Level 3: AsrModels — `AsrModels.swift:43` [LIBRARY]
```swift
private static func repoPath(from modelsDirectory: URL) -> URL {
    return modelsDirectory.deletingLastPathComponent()
        .appendingPathComponent(DownloadUtils.Repo.parakeet.folderName)  // Hardcoded
}
```

### Level 4: DownloadUtils — `DownloadUtils.swift:27` [LIBRARY]
```swift
public enum Repo: String, CaseIterable {
    case vad = "FluidInference/silero-vad-coreml"
    case parakeet = "FluidInference/parakeet-tdt-0.6b-v2-coreml"  // THE ONLY ASR MODEL
    case diarizer = "FluidInference/speaker-diarization-coreml"
}
```

### Result: All Model Sizes Produce Identical Output

| Requested | Actual Model Used | Time |
|-----------|-------------------|------|
| `tiny` | `parakeet-tdt-0.6b-v2-coreml` | Same |
| `base` | `parakeet-tdt-0.6b-v2-coreml` | Same |
| `small` | `parakeet-tdt-0.6b-v2-coreml` | Same |
| `medium` | `parakeet-tdt-0.6b-v2-coreml` | Same |
| `large` | `parakeet-tdt-0.6b-v2-coreml` | Same |

---

## CoreML & Apple Silicon Configuration

### What is CoreML?

CoreML is Apple's machine learning framework for optimized inference on Apple devices. It enables:
- **Neural Engine execution** — Dedicated ML accelerator on Apple Silicon
- **GPU acceleration** — Metal Performance Shaders
- **CPU fallback** — For operations not supported on ANE/GPU

### Compute Units Configuration — `AsrModels.swift:117-124` [LIBRARY]

```swift
public static func defaultConfiguration() -> MLModelConfiguration {
    let config = MLModelConfiguration()
    config.allowLowPrecisionAccumulationOnGPU = true  // FP16 GPU accumulation
    let isCI = ProcessInfo.processInfo.environment["CI"] != nil
    config.computeUnits = isCI ? .cpuAndNeuralEngine : .all
    return config
}
```

**In normal mode (not CI):**
- `.all` = CPU + GPU + Neural Engine (full Apple Silicon utilization)
- `allowLowPrecisionAccumulationOnGPU = true` enables FP16 for faster GPU execution

**In CI mode:**
- `.cpuAndNeuralEngine` = Excludes GPU (avoids virtualization issues)

### Apple Silicon Optimizations

1. **Neural Engine Optimized**: Models are converted to CoreML format specifically for ANE execution
2. **Low Precision GPU**: FP16 accumulation enabled for faster GPU operations
3. **Unified Memory**: Apple Silicon's unified memory architecture eliminates data transfer overhead

### PyTorch SDPA — NOT RELEVANT

Scaled Dot-Product Attention (SDPA) is a **PyTorch-specific optimization**. CoreML handles attention mechanisms differently through its own optimized operations. FluidAudio models are pre-converted to CoreML format, bypassing PyTorch entirely at runtime.

### Quantization

The model repository includes a **4-bit quantized variant** (`ParakeetEncoder_4bit_par.mlmodelc` at 305MB vs 1.1GB full), but the **default configuration uses full precision models**. The quantized variant is available but not currently selected.

---

## Download Behavior Deep Dive

### Not Using HuggingFace Hub API

Unlike most Python implementations that use `huggingface_hub` library, FluidAudio:

1. **Lists files via REST API**: `https://huggingface.co/api/models/{repo}/tree/main`
2. **Downloads directly**: `https://huggingface.co/{repo}/resolve/main/{path}`

This approach:
- **Pro**: No Python dependency, works in pure Swift
- **Con**: Cannot share models with other tools using standard HF cache

### Resume Support — NOT Actually Implemented

The code **structure suggests resume support** but **doesn't actually use it**:

```swift
// Line 277-283: Calculates startByte (but never used!)
var startByte: Int64 = 0
if let attrs = try? FileManager.default.attributesOfItem(atPath: tempURL.path),
   let fileSize = attrs[.size] as? Int64 {
    startByte = fileSize
    logger.info("⏸️ Resuming download from \(formatBytes(Int(startByte)))")
}

// Line 342: Uses URLSession.download() which IGNORES startByte!
let (tempFile, response) = try await session.download(for: request)
```

The comment at line 341 explains: "Always use URLSession.download for reliability (proven to work in PR #32)"

### File-Level Completeness Check — `DownloadUtils.swift:266-272`

```swift
// Check if file already exists and is complete
if let attrs = try? FileManager.default.attributesOfItem(atPath: destination.path),
   let fileSize = attrs[.size] as? Int64,
   fileSize == expectedSize {
    logger.info("✅ File already downloaded: \(path)")
    progressHandler?(1.0)
    return
}
```

**Per-file behavior:**
- File exists AND size matches expected → **SKIP** (efficient)
- File missing OR size mismatch → **DOWNLOAD FRESH** (no resume)

### Model Load Failure Recovery — `DownloadUtils.swift:35-56`

```swift
public static func loadModels(...) async throws -> [String: MLModel] {
    do {
        return try await loadModelsOnce(...)  // 1st attempt
    } catch {
        // If load fails, WIPE ENTIRE CACHE and re-download
        logger.warning("⚠️ First load failed: \(error.localizedDescription)")
        logger.info("🔄 Deleting cache and re-downloading…")
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try? FileManager.default.removeItem(at: repoPath)  // WIPES ALL ~2.4GB!
        return try await loadModelsOnce(...)  // 2nd attempt
    }
}
```

**Critical behavior**: Any CoreML loading failure triggers COMPLETE cache wipe of ~2.4GB!

---

## Notes

### Why Not HuggingFace Hub Cache?

FluidAudio is a **Swift-native library** designed for Apple platforms. Using the standard `~/.cache/huggingface/hub/` would require:
- Implementing HF Hub's complex directory structure with snapshots/blobs
- Maintaining Python interop for cache management
- Additional complexity for a Swift-only solution

The trade-off is **no model sharing** with Python-based implementations.

### Download-on-Every-Call Architecture

Unlike implementations that load models once during `load_model()`, FluidAudio's Swift bridge:
1. Is a subprocess that reinitializes on every transcription call
2. Calls `downloadAndLoad()` on every invocation
3. Checks for existing models and skips download if present

This means the "download check" runs on EVERY transcription, but actual download only happens if models are missing.

### Timeout Discrepancy

| Level | Timeout | Adequate? |
|-------|---------|-----------|
| Python subprocess | 300s (5 min) | **NO** — download takes ~22 min |
| Swift DownloadConfig | 1800s (30 min) | **YES** — sufficient for most connections |

The Python timeout kills the process before Swift's timeout can complete the download.

---

## Key Source Files

### Project Files [PROJECT]

| File | Purpose | Key Lines |
|------|---------|-----------|
| `fluidaudio_coreml.py` | Python implementation wrapper | 116 (timeout), 46-64 (model ignored) |
| `test_benchmark2.py` | Test entry point | 88-94 |

### Bridge Files [BRIDGE]

| File | Purpose |
|------|---------|
| `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift` | Swift CLI entry point, ASR orchestration |

### Library Files [LIBRARY]

| File | Purpose | Key Lines |
|------|---------|-----------|
| `AsrModels.swift` | Model download and loading | 121 (compute units), 139 (modelsExist), 172-178 (downloadAndLoad) |
| `DownloadUtils.swift` | HuggingFace download utilities | 27 (repo enum), 266-272 (size check), 342 (no resume) |
| `AsrManager.swift` | ASR pipeline orchestration | N/A |
| `TdtDecoder.swift` | Token Duration Transducer decoder | N/A |

---

## Empirical Test Results

**Test Date:** 2026-01-12
**Test Environment:** macOS, Apple Silicon (M-series), ~2.5 MB/s download speed

### Test 1: Fresh Download (No Local Models)

**Pre-condition:**
```bash
$ ls -la ~/Library/Application\ Support/FluidAudio/Models/
total 16
drwxr-xr-x  3 rymalia  staff    96 Jan 12 12:16 .
drwxr-xr-x  4 rymalia  staff   128 Dec 24 23:59 ..
-rw-r--r--@ 1 rymalia  staff  6148 Jan 12 12:16 .DS_Store
# Model folder does not exist
```

**Command run:**
```bash
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json
```

**Terminal output:**
```
📥 Downloading parakeet-tdt-0.6b-v2-coreml...
   ⏳ 100% downloaded of weight.bin
   ⏳ 100% downloaded of weight.bin
   ⏳ 100% downloaded of weight.bin
   ⏳ 100% downloaded of weight.bin
{
  "text" : "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "processing_time" : 0.40139305591583252,
  "language" : "en",
  "transcription_time" : 0.40140604972839355
}
```

**Download statistics:**
- **Total time:** ~22 minutes (started 6:29PM, completed ~6:51PM)
- **Total size:** 2.4GB
- **Files downloaded:** 13 model files

**Final cache state:**
```bash
$ du -sh ~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/
2.4G	/Users/rymalia/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/

$ ls -la ~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/
total 48
drwxr-xr-x@ 15 rymalia  staff    480 Jan 12 18:51 .
drwxr-xr-x   4 rymalia  staff    128 Jan 12 18:29 ..
-rw-------@  1 rymalia  staff      3 Jan 12 18:51 config.json
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:30 Decoder.mlmodelc
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:35 Encoder.mlmodelc
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:35 JointDecision.mlmodelc
drwxr-xr-x@  6 rymalia  staff    192 Jan 12 18:35 Melspectogram.mlmodelc
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:35 Melspectrogram_v2.mlmodelc
-rw-------@  1 rymalia  staff  18762 Jan 12 18:51 parakeet_vocab.json
drwxr-xr-x@  6 rymalia  staff    192 Jan 12 18:35 ParakeetDecoder.mlmodelc
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:46 ParakeetEncoder_4bit_par.mlmodelc
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:51 ParakeetEncoder_v2.mlmodelc
drwxr-xr-x@  6 rymalia  staff    192 Jan 12 18:44 ParakeetEncoder.mlmodelc
drwxr-xr-x@  7 rymalia  staff    224 Jan 12 18:51 Preprocessor.mlmodelc
drwxr-xr-x@  6 rymalia  staff    192 Jan 12 18:51 RNNTJoint.mlmodelc
```

---

### Test 2: Cached Model Performance

**Command run:**
```bash
time .venv/bin/python3 test_benchmark2.py small 2 "FluidAudioCoreMLImplementation"
```

**Terminal output:**
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - FluidAudioCoreMLImplementation

Starting benchmark with model 'small' (2 run(s))...
[info] Just kidding! fluidaudio-bridge ignores model_name parameter!
[info] fluidaudio-bridge uses hard-coded model: parakeet-tdt-0.6b-v2-coreml
[info] Run 1/2 for FluidAudioCoreMLImplementation
[info] FluidAudio transcription time: 0.3714s (internal processing: 0.3714s)
[info] Using internal transcription time: 0.3714s (total with overhead: 1.4029s)
[info] Run 1 completed in 0.3714 seconds
[info] Run 2/2 for FluidAudioCoreMLImplementation
[info] FluidAudio transcription time: 0.3717s (internal processing: 0.3717s)
[info] Using internal transcription time: 0.3717s (total with overhead: 0.6775s)
[info] Run 2 completed in 0.3717 seconds
[info] Average time for FluidAudioCoreMLImplementation: 0.3716 seconds

=== Benchmark Summary for 'small' model ===
Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
fluidaudio-coreml      0.3716          model=parakeet-tdt-0.6b-v2-coreml, backend=FluidAudio Swift Bridge
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your..."

.venv/bin/python3 test_benchmark2.py small 2 "FluidAudioCoreMLImplementation"  1.43s user 0.19s system 69% cpu 2.342 total
```

**Performance metrics:**

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Internal transcription time | 0.3714s | 0.3717s |
| Total with subprocess overhead | 1.4029s | 0.6775s |

| Summary | Value |
|---------|-------|
| Average internal time | **0.3716s** |
| Wall clock (2 runs) | **2.34s** |
| First-run overhead | ~1.0s (model loading) |
| Subsequent run overhead | ~0.3s |

**Observations:**
1. **No re-download:** Models were loaded from cache immediately
2. **Fast transcription:** ~0.37s for 11-second JFK audio (30x real-time)
3. **First-run overhead:** ~1s for CoreML model initialization
4. **Subsequent runs:** Minimal overhead (~0.3s)

---

### Test 3: Incomplete File Handling (Missing Required File)

**Test procedure:**
1. Renamed `RNNTJoint.mlmodelc` → `RNNT__OFF__Joint.mlmodelc` to simulate missing required file
2. Ran bridge to trigger model loading with missing dependency

**Command run:**
```bash
# Rename required file
mv ~/Library/.../RNNTJoint.mlmodelc ~/Library/.../RNNT__OFF__Joint.mlmodelc

# Run bridge
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json
```

**Before test:**
```bash
$ du -sh ~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/
2.4G
```

**After test (within seconds):**
```bash
$ du -sh ~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/
14M

$ ls ~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/
Decoder.mlmodelc
Encoder.mlmodelc
# Only 2 files remain! Entire 2.4GB folder was WIPED!
```

**Critical findings:**

| Event | Result |
|-------|--------|
| Missing required file detected | `modelsExist()` returned false |
| Download initiated | `download()` called |
| First load attempt | **FAILED** (missing RNNTJoint) |
| Recovery triggered | **ENTIRE 2.4GB FOLDER WIPED** |
| Renamed file | **ALSO DELETED** (was in same folder) |
| Download restarted | From scratch (14MB, downloading Decoder/Encoder)

**Confirmed behavior:** The aggressive wipe-and-restart at `DownloadUtils.swift:46-50` [LIBRARY] deletes the **entire model folder** on any loading failure, including:
- All complete model files
- Any renamed/hidden files in the folder
- Forces complete re-download (~22 minutes)

> **⚠️ WARNING:** Missing a single required file triggers deletion of **ALL** model data (~2.4GB) and requires full re-download.

---

### Test Summary

| Test | Result | Key Finding |
|------|--------|-------------|
| Fresh download | ✅ Success | 22 min, 2.4GB, 13 files |
| Cached performance | ✅ Success | 0.37s transcription, 30x real-time |
| Missing file handling | ✅ Verified | **COMPLETE WIPE** of 2.4GB cache |

**Post-test note:** Models were wiped during Test 3 and need to be re-downloaded for future use via:
```bash
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json
```

---

## Known Issues / Conflicts Discovered

### Issue 1: Python Timeout Too Short (P0 - Critical)

**Problem:** Python subprocess timeout of 300s (5 minutes) is insufficient for first-run model download (~22 minutes).

**Impact:** First-run benchmarks will ALWAYS timeout unless models are pre-downloaded.

**Location:** `fluidaudio_coreml.py:116` [PROJECT]
```python
timeout=300  # 5 minute timeout for model download on first run
```

**Recommended Fix:** Increase to at least 1800s (30 minutes).

### Issue 2: Resume Support Not Actually Implemented (P1 - High)

**Problem:** Code structure suggests resume support (`startByte` calculation at line 277-283), but `performChunkedDownload` uses `URLSession.download()` which starts fresh.

**Impact:** Interrupted downloads waste bandwidth (no resume from partial).

**Location:** `DownloadUtils.swift:277-283, 342` [LIBRARY]

### Issue 3: Model Size Parameter Ignored (P2 - Medium)

**Problem:** The `model_name` parameter is completely ignored at all levels.

**Impact:** Benchmark results for "tiny", "small", "medium", "large" are all identical.

**Location:** Multiple — see "Model Mapping Chain" section

### Issue 4: Non-Standard Cache Location (P2 - Medium)

**Problem:** Uses `~/Library/Application Support/FluidAudio/Models/` instead of standard HF cache.

**Impact:** Models cannot be shared with other implementations. Requires separate ~2.4GB storage.

### Issue 5: Aggressive Cache Wipe on Load Failure (P2 - Medium)

**Problem:** Any CoreML loading failure triggers complete deletion of ~2.4GB cache.

**Impact:** A single corrupt file causes re-download of everything.

**Location:** `DownloadUtils.swift:46-50` [LIBRARY]

---

## Recommended Improvements

### Priority Summary

| Priority | Improvement | Effort | Impact | Status |
|----------|-------------|--------|--------|--------|
| **P0** | Fix Python timeout (300s → 1800s) | 1 line | Critical — fixes first-run failures | 🔴 Not started |
| **P1** | Add pre-download warning check | ~20 lines | High — better UX | 🔴 Not started |
| **P1** | Add `--download-only` CLI command | ~50 lines | High — CI/CD friendly | 🔴 Not started |
| **P2** | Implement actual resume support | ~50 lines Swift | Medium — saves bandwidth | 🔴 Not started |
| **P2** | Stream download progress to Python | ~50 lines | Medium — better UX | 🔴 Not started |
| **P2** | Improve partial failure recovery | ~100 lines Swift | Medium — saves bandwidth | 🔴 Not started |
| **P3** | Support model size selection | Large | Low — feature parity | 🔴 Not started |
| **P3** | Consider HuggingFace cache | Large | Low — ecosystem consistency | 🔴 Not started |

### Implementation Order Recommendation

**Phase 1 (Quick Wins):**
- [ ] P0: Increase timeout to 1800s in `fluidaudio_coreml.py:116`
- [ ] P1: Add models exist check with warning message

**Phase 2 (User Experience):**
- [ ] P1: Add `--download-only` CLI command
- [ ] P2: Stream download progress to Python stderr

**Phase 3 (Robustness):**
- [ ] P2: Implement actual resume support with Range headers
- [ ] P2: Improve partial failure recovery (selective re-download)

**Phase 4 (Feature Parity):**
- [ ] P3: Support model size selection (pending upstream)
- [ ] P3: Consider HuggingFace cache migration

---

## Workarounds

### Manual Model Pre-Download

If Python timeout prevents automatic download, run the bridge directly:

```bash
# Run bridge directly (no Python timeout, will complete download)
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json

# This will download ~2.4GB (may take ~30 minutes)
# Once complete, subsequent Python benchmark runs will use cached model
```

### Clean Corrupt Cache

If cache is in corrupt state (incomplete files causing load failures):

```bash
# Delete the model folder entirely
rm -rf ~/Library/Application\ Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml/

# Re-run bridge to trigger fresh download
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json
```
