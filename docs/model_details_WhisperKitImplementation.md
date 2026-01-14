# Model Details: WhisperKitImplementation

This document traces the complete execution flow for `WhisperKitImplementation`, documenting model download behavior, caching, architecture, and critical bugs discovered through empirical testing on both small and large models.

---

## File Reference Legend

Throughout this document, files are categorized as:
- **[PROJECT]** - Files in this repository that you can modify
- **[LIBRARY]** - Installed package files (Swift/Python dependencies)
- **[BRIDGE]** - The Swift CLI executable that interfaces with WhisperKit

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| Does Swift bridge re-download every call? | **NO** - uses file-exists caching | HubApi.swift:204-206 |
| Where are temp files during download? | `/private/var/folders/*/T/CFNetworkDownload_*.tmp` | Empirical: found 611MB, 477MB orphaned files |
| Download cache vs runtime location? | **SAME** - `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/` | HubApi.swift:19-29, empirical verification |
| Is 300s timeout sufficient for large? | **NO** - large model needs ~17 minutes | Empirical: 611MB in 5 min = ~2 MB/s, ~2GB total |
| Does partial download resume? | **NO** - starts fresh, orphans temp files | Empirical: new 477MB temp file on run 2 |
| Completeness check on existing files? | **NONE** - only `FileManager.fileExists()` | HubApi.swift:204-206, empirical: 4K file skipped |
| What happens with incomplete cache? | **SILENT FAILURE** - incomplete files treated as complete | Empirical: 4K TextDecoder skipped |

---

## Benchmark Execution Flow

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py small 1 WhisperKitImplementation
```

### Step-by-Step Execution

1. **Entry Point** — **[PROJECT]** `test_benchmark2.py:88-94`
   - Parses CLI args: `model="small"`, `runs=1`, `implementations="WhisperKitImplementation"`
   - Calls `asyncio.run(main(model, runs, implementations))`

2. **Audio Loading** — **[PROJECT]** `test_benchmark2.py:36-55`
   - Loads `tests/jfk.wav` via soundfile
   - Converts to mono if stereo
   - Resamples to 16kHz if needed
   - Audio is float32 normalized to [-1, 1]

3. **Implementation Filtering** — **[PROJECT]** `test_benchmark2.py:58-74`
   - `get_all_implementations()` returns all available implementation classes
   - Filters to only `WhisperKitImplementation`

4. **BenchmarkConfig Creation** — **[PROJECT]** `test_benchmark2.py:77-82`
   ```python
   config = BenchmarkConfig(
       model_name="small",
       implementations=[WhisperKitImplementation],
       num_runs=1,
       audio_data=whisper_audio,
   )
   ```

5. **Benchmark Runner** — **[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:110-190`
   - Iterates over implementations
   - Creates instance: `implementation = WhisperKitImplementation()`

6. **Implementation `__init__`** — **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:21-31`
   - Validates macOS platform (WhisperKit only works on Apple platforms)
   - Calls `_find_bridge_executable()` to locate the Swift bridge

7. **Bridge Executable Discovery** — **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:33-44`
   ```python
   project_root = Path(__file__).parent.parent.parent.parent
   bridge_path = project_root / "tools" / "whisperkit-bridge" / ".build" / "release" / "whisperkit-bridge"
   ```
   - Looks for the compiled Swift bridge at: `tools/whisperkit-bridge/.build/release/whisperkit-bridge`
   - Raises `RuntimeError` if not found (user must run `swift build -c release` first)

8. **Model Loading** — **[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:134`
   ```python
   implementation.load_model(config.model_name)  # config.model_name = "small"
   ```

9. **`load_model()` Method** — **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:46-95`
   - Maps model name via internal dictionary (see Model Mapping Reference below)
   - For `"small"` → stores `self.model_name = "small"` (no change)
   - For `"large"` → stores `self.model_name = "large-v3"` (upgraded)
   - Tests bridge by running: `whisperkit-bridge --help`
   - **Note**: Model files are NOT downloaded yet — only when transcription runs

10. **Transcription Call** — **[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:143-144`
    ```python
    start_time = time.time()
    result = await implementation.transcribe(config.audio_data)
    ```

11. **`transcribe()` Method** — **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:97-168`
    - Preprocesses audio (mono, float32, normalized)
    - Writes audio to temporary WAV file at 16kHz
    - Calls subprocess:
      ```bash
      whisperkit-bridge /tmp/audio.wav --format json --model small
      ```
    - Parses JSON output from stdout
    - Extracts `transcription_time` from WhisperKit's internal timing
    - Cleans up temp file

12. **Swift Bridge Execution** — **[BRIDGE]** `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift:23-75`
    - Creates `WhisperKitConfig(model: "small")`
    - Initializes WhisperKit: `try await WhisperKit(config)`
    - **This is where model download happens** (if not already cached)
    - Calls: `try await whisperKit.transcribe(audioPath: inputFile)`
    - Reports internal timing: `result.timings.fullPipeline`

13. **WhisperKit Model Setup** — **[LIBRARY]** `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:71-78`
    ```swift
    try await setupModels(
        model: config.model,        // "small" or "large-v3"
        downloadBase: nil,          // Uses default
        modelRepo: nil,             // Uses "argmaxinc/whisperkit-coreml"
        download: true
    )
    ```

14. **Model Download Triggered** — **[LIBRARY]** `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:298-330`
    - `setupModels()` calls `Self.download(variant: modelName, ...)`
    - Uses HubApi from swift-transformers

15. **HubApi Snapshot Download** — **[LIBRARY]** `HubApi.swift:234-257`
    - Downloads files matching `*{model}/*` from `argmaxinc/whisperkit-coreml`
    - For each file, checks `downloaded` property (line 204-206):
      ```swift
      var downloaded: Bool {
          FileManager.default.fileExists(atPath: destination.path)
      }
      ```
    - **CRITICAL BUG**: Only checks existence, not completeness!

16. **HuggingFace Hub Cache** — **[LIBRARY]** `swift-transformers/Sources/Hub/HubApi.swift:19-29`
    ```swift
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    self.downloadBase = documents.appending(component: "huggingface")
    ```
    - Default cache location: `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/`

17. **Model Loading** — **[LIBRARY]** `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:337-458`
    - Loads CoreML models from disk:
      - `MelSpectrogram.mlmodelc` (feature extractor)
      - `AudioEncoder.mlmodelc`
      - `TextDecoder.mlmodelc`
      - `TextDecoderContextPrefill.mlmodelc` (optional, for faster decoding)
    - Loads tokenizer from HuggingFace (e.g., `openai/whisper-small`)

18. **Results** — **[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:165-171`
    - Stores `BenchmarkResult` with timing and transcription text
    - Uses internal `_transcription_time` from WhisperKit (excludes subprocess overhead)
    - Calls `implementation.get_params()` → `{"model": "small", "backend": "WhisperKit Swift Bridge", "platform": "Apple Silicon"}`

---

## Summary Table

| Attribute | Small Model | Large Model |
|-----------|-------------|-------------|
| **Requested Model** | `small` | `large` |
| **Mapped Model Name** | `small` (no change) | `large-v3` (upgraded) |
| **HuggingFace Repo ID** | `argmaxinc/whisperkit-coreml` | `argmaxinc/whisperkit-coreml` |
| **Model Search Pattern** | `*small/*` or `*openai*small/*` | `*large-v3/*` or `*openai*large-v3/*` |
| **Expected Files** | `AudioEncoder.mlmodelc/`, `TextDecoder.mlmodelc/`, `MelSpectrogram.mlmodelc/`, config files, tokenizer | Same structure |
| **Cache Location** | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/` | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/` |
| **Temp Files During Download** | `/private/var/folders/*/T/CFNetworkDownload_*.tmp` | Same (orphaned on timeout) |
| **Expected Total Size** | ~487MB | ~2.9GB |
| **Current Timeout** | 300 seconds (5 minutes) | 300 seconds (5 minutes) |
| **Required Timeout** | <300s ✅ | ~1000 seconds (~17 minutes) ❌ |
| **Tokenizer Source** | `openai/whisper-small` from HuggingFace | `openai/whisper-large-v3` from HuggingFace |
| **Backend** | Swift/CoreML via subprocess bridge | Same |

---

## Model Mapping Reference

### Project-Level Mapping — **[PROJECT]** `whisperkit.py:58-67`

| Input | Output | Notes |
|-------|--------|-------|
| `"tiny"` | `"tiny"` | No change |
| `"base"` | `"base"` | No change |
| `"small"` | `"small"` | No change |
| `"medium"` | `"medium"` | No change |
| `"large"` | `"large-v3"` | **Upgraded to latest version** |
| `"large-v3"` | `"large-v3"` | No change |
| `"large-v3-turbo"` | `"large-v3-turbo"` | Explicit turbo access (smaller, faster) |
| `"large-turbo"` | `"large-v3-turbo"` | Alternative turbo access |
| *(other)* | *(unchanged)* | Passed through as-is |

### WhisperKit Model Variants Available in argmaxinc/whisperkit-coreml

The `argmaxinc/whisperkit-coreml` repository contains pre-compiled CoreML models for various Whisper sizes. Unlike MLX implementations that download weights + config, WhisperKit downloads **compiled `.mlmodelc` bundles** ready for Apple Silicon inference.

Common model folders in the repo:
- `openai_whisper-tiny/`
- `openai_whisper-base/`
- `openai_whisper-small/`
- `openai_whisper-medium/`
- `openai_whisper-large-v3/`
- `openai_whisper-large-v3-turbo/`

---

## Notes

### 1. Swift Bridge Architecture

Unlike Python-only implementations, WhisperKitImplementation uses a **subprocess bridge** pattern:

```
Python (whisperkit.py)
    ↓ subprocess call
Swift CLI (whisperkit-bridge)
    ↓ native API
WhisperKit Framework (Swift/CoreML)
    ↓
Apple Neural Engine / GPU
```

**Why?** WhisperKit is a Swift framework that uses CoreML models compiled specifically for Apple Silicon. There's no Python binding, so the bridge executable provides inter-process communication.

**Trade-offs:**
- **Pro**: Native Apple Silicon performance (ANE + GPU)
- **Pro**: Pre-compiled CoreML models (no conversion needed)
- **Con**: Subprocess overhead (~0.9s per invocation)
- **Con**: Temp file I/O for audio transfer

### 2. Cache Behavior — Swift Hub vs Python HuggingFace Hub

**Critical Difference**: The Swift `Hub` library (from `swift-transformers`) uses a **different default cache location** than Python's `huggingface_hub`:

| Library | Default Cache Location |
|---------|------------------------|
| Python `huggingface_hub` | `~/.cache/huggingface/hub/` |
| Swift `Hub` | `~/Documents/huggingface/` |

This means **WhisperKit models are NOT shared** with Python MLX implementations. Running both will result in separate downloads.

### 3. Model Download Timing

Models are downloaded **lazily** — not during `load_model()`, but during the **first transcription**. This is because:
1. `load_model()` only tests the bridge with `--help`
2. The Swift bridge initializes WhisperKit during transcription
3. WhisperKit's `init()` triggers `setupModels()` which downloads if needed

**Implication**: First transcription run will be much slower (includes download time). Subsequent runs use cached models.

### 4. Internal Timing Extraction

The bridge returns timing information that excludes subprocess overhead:

```swift
// main.swift:41
let transcriptionTime = firstResult?.timings.fullPipeline ?? 0.0
```

This `fullPipeline` timing measures only:
- Mel spectrogram computation
- Audio encoding
- Text decoding

It **excludes**:
- Audio file loading
- Subprocess startup
- JSON serialization
- Model download time

**[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:149-152` uses this internal timing when available:
```python
if hasattr(result, '_transcription_time'):
    run_time = result._transcription_time
```

This means first-run times (with download) show internal timing only, not the actual wall-clock time experienced.

### 5. CoreML Model Compilation

WhisperKit uses **pre-compiled** CoreML models (`.mlmodelc` bundles). These are different from:
- **MLX weights** (`.npz` files) — require runtime loading
- **PyTorch checkpoints** (`.bin` files) — require conversion

Pre-compilation benefits:
- Faster model loading (no JIT compilation)
- Optimized for specific Apple hardware
- Reduced memory during initialization

### 6. Turbo Model Access

The "turbo" models are distilled versions with fewer decoder layers, offering ~3x speedup. Access them explicitly:

```python
# Will NOT use turbo (maps to large-v3)
implementation.load_model("large")

# WILL use turbo
implementation.load_model("large-v3-turbo")
implementation.load_model("large-turbo")  # Alternative
```

### 7. Build Requirement

The Swift bridge must be compiled before use:

```bash
cd tools/whisperkit-bridge
swift build -c release
```

This creates the executable at:
```
tools/whisperkit-bridge/.build/release/whisperkit-bridge
```

If not built, `WhisperKitImplementation()` raises:
```
RuntimeError: WhisperKit bridge not found at {path}.
Please build it first by running: cd tools/whisperkit-bridge && swift build -c release
```

### 8. Audio Preprocessing

**[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:170-209`

The implementation preprocesses audio before writing to temp file:
1. Convert to float32
2. Convert stereo to mono (average channels)
3. Normalize to [-1, 1] range
4. Pad if too short (<0.1s at 16kHz)

### 9. Timeout Configuration

The subprocess has a **5-minute timeout** to accommodate first-run model downloads:

```python
result = subprocess.run(
    [self._bridge_path, temp_path, "--format", "json", "--model", self.model_name],
    timeout=300  # 5 minute timeout for model download on first run
)
```

**NOTE**: This timeout is **insufficient for large models** (~2.9GB), causing guaranteed failure on first run.

### 10. Model Sizes (Approximate)

| Model | AudioEncoder | TextDecoder | MelSpectrogram | Total |
|-------|--------------|-------------|----------------|-------|
| tiny | ~40MB | ~40MB | ~372KB | ~80MB |
| small | ~180MB | ~305MB | ~372KB | ~487MB |
| large-v3 | ~1.2GB | ~1.7GB | ~392KB | ~2.9GB |
| large-v3-turbo | ~300MB | ~200MB | ~392KB | ~500MB |

---

## Key Source Files

### Project Files (You Can Modify)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `test_benchmark2.py` | Entry point, CLI argument parsing | 88-94 |
| `src/mac_whisper_speedtest/benchmark.py` | Benchmark orchestration, timing | 110-190 |
| `src/mac_whisper_speedtest/implementations/whisperkit.py` | Implementation wrapper, subprocess call | 128 (timeout), 58-67 (model mapping) |

### Bridge Files (Swift — Compile with `swift build`)

| File | Purpose |
|------|---------|
| `tools/whisperkit-bridge/Package.swift` | Swift package manifest, WhisperKit dependency |
| `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift` | CLI entry point, JSON output, model init |

### Library Files (Swift Package Dependencies — Read-Only)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift` | Main WhisperKit class, model setup, download trigger | 298-330 (download), 337-458 (loading) |
| `WhisperKit/Sources/WhisperKit/Core/Configurations.swift` | WhisperKitConfig, DecodingOptions | |
| `WhisperKit/Sources/WhisperKit/Utilities/ModelUtilities.swift` | Model detection, tokenizer loading | |
| `swift-transformers/Sources/Hub/HubApi.swift` | HuggingFace download, caching | 204-206 (exists check), 234-257 (snapshot) |
| `swift-transformers/Sources/Hub/Downloader.swift` | URLSession download | 78, 112-119 |

---

## Empirical Test Results

**Test Date**: January 10-11, 2026
**Test Environment**: macOS, Apple Silicon, ~2 MB/s download speed

---

### Small Model Tests (Success ✅)

#### Test 1: Fresh Download (No Cache)

**Initial state**: Cache folder was completely empty (no `whisperkit-coreml` folder existed)
```
$ ls -la ~/Documents/huggingface/models/argmaxinc/
total 24
drwxr-xr-x@ 3 rymalia  staff     96 Jan 10 21:11 .
drwxr-xr-x@ 4 rymalia  staff    128 Dec 19 16:37 ..
-rw-r--r--@ 1 rymalia  staff  10244 Jan 10 21:11 .DS_Store
```

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py small 1 WhisperKitImplementation
```

**Terminal output**:
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - WhisperKitImplementation

Starting benchmark with model 'small' (1 run(s))...
2026-01-10 21:14:58 [info     ] Benchmarking WhisperKitImplementation with model small
2026-01-10 21:14:58 [info     ] Found WhisperKit bridge at: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/tools/whisperkit-bridge/.build/release/whisperkit-bridge
2026-01-10 21:14:58 [info     ] Loading model for WhisperKitImplementation
2026-01-10 21:14:58 [info     ] WhisperKit bridge ready for model: small
2026-01-10 21:14:58 [info     ] WhisperKit bridge is working correctly
2026-01-10 21:14:58 [info     ] Run 1/1 for WhisperKitImplementation
2026-01-10 21:14:58 [info     ] Transcribing with WhisperKit via Swift bridge (model: small)
2026-01-10 21:14:58 [debug    ] Preprocessed audio: shape=(176000,), dtype=float32, range=[-0.724, 0.783]
2026-01-10 21:18:56 [info     ] WhisperKit transcription time: 0.4576s
2026-01-10 21:18:56 [info     ] Using internal transcription time: 0.4576s (total with overhead: 237.3534s)
2026-01-10 21:18:56 [info     ] Run 1 completed in 0.4576 seconds
2026-01-10 21:18:56 [info     ] Transcription: And so my fellow Americans, ask not what your coun...
2026-01-10 21:18:56 [info     ] Average time for WhisperKitImplementation: 0.4576 seconds

=== Benchmark Summary for 'small' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
whisperkit             0.4576          model=small, backend=WhisperKit Swift Bridge, platform=Apple Silicon
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."
```

**Key observation**: Total time with overhead was **237.35s** (~4 minutes, includes model download), internal transcription was **0.46s**.

**Cache folder after download**:
```
$ ls -la ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/
total 40
drwxr-xr-x@ 5 rymalia  staff   160 Jan 10 21:15 .
drwxr-xr-x@ 4 rymalia  staff   128 Jan 10 21:14 ..
-rw-r--r--@ 1 rymalia  staff  6148 Jan 10 21:18 .DS_Store
-rw-------@ 1 rymalia  staff  8705 Jan 10 21:14 config.json
drwxr-xr-x@ 9 rymalia  staff   288 Jan 10 21:18 openai_whisper-small
```

**Model folder contents**:
```
$ ls -la ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/
total 32
drwxr-xr-x@ 9 rymalia  staff   288 Jan 10 21:18 .
drwxr-xr-x@ 5 rymalia  staff   160 Jan 10 21:15 ..
-rw-r--r--@ 1 rymalia  staff  6148 Jan 10 21:18 .DS_Store
drwxr-xr-x@ 8 rymalia  staff   256 Jan 10 21:16 AudioEncoder.mlmodelc
-rw-------@ 1 rymalia  staff  1456 Jan 10 21:16 config.json
-rw-------@ 1 rymalia  staff  2779 Jan 10 21:16 generation_config.json
drwxr-xr-x@ 7 rymalia  staff   224 Jan 10 21:16 MelSpectrogram.mlmodelc
drwxr-xr-x@ 3 rymalia  staff    96 Jan 10 21:18 models
drwxr-xr-x@ 8 rymalia  staff   256 Jan 10 21:16 TextDecoder.mlmodelc
```

**CoreML bundle sizes**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/*.mlmodelc
178M    AudioEncoder.mlmodelc
372K    MelSpectrogram.mlmodelc
305M    TextDecoder.mlmodelc
```

**Total model size**: **487MB**

#### Test 2: Cached Model (Second Run)

**Command run** (same as above):
```bash
.venv/bin/python3 test_benchmark2.py small 1 WhisperKitImplementation
```

**Terminal output**:
```
Loading audio from: tests/jfk.wav
Loaded audio: 176000 samples at 16000 Hz
Audio ready for Whisper: 176000 samples

Chosen implementations: 1
  - WhisperKitImplementation

Starting benchmark with model 'small' (1 run(s))...
2026-01-10 21:19:37 [info     ] Benchmarking WhisperKitImplementation with model small
2026-01-10 21:19:37 [info     ] Found WhisperKit bridge at: /Users/rymalia/projects/mac-whisper-speedtest_MAIN/tools/whisperkit-bridge/.build/release/whisperkit-bridge
2026-01-10 21:19:37 [info     ] Loading model for WhisperKitImplementation
2026-01-10 21:19:37 [info     ] WhisperKit bridge ready for model: small
2026-01-10 21:19:37 [info     ] WhisperKit bridge is working correctly
2026-01-10 21:19:37 [info     ] Run 1/1 for WhisperKitImplementation
2026-01-10 21:19:37 [info     ] Transcribing with WhisperKit via Swift bridge (model: small)
2026-01-10 21:19:37 [debug    ] Preprocessed audio: shape=(176000,), dtype=float32, range=[-0.724, 0.783]
2026-01-10 21:19:38 [info     ] WhisperKit transcription time: 0.4366s
2026-01-10 21:19:38 [info     ] Using internal transcription time: 0.4366s (total with overhead: 1.3747s)
2026-01-10 21:19:38 [info     ] Run 1 completed in 0.4366 seconds
2026-01-10 21:19:38 [info     ] Transcription: And so my fellow Americans, ask not what your coun...
2026-01-10 21:19:38 [info     ] Average time for WhisperKitImplementation: 0.4366 seconds

=== Benchmark Summary for 'small' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
whisperkit             0.4366          model=small, backend=WhisperKit Swift Bridge, platform=Apple Silicon
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."
```

**Key observation**: Total time with overhead was **1.37s**, internal transcription was **0.44s**. No download needed — model was cached.

#### Small Model Performance Summary

| Run | Total Time (with overhead) | Internal Transcription | Notes |
|-----|---------------------------|------------------------|-------|
| First (cold) | 237.35s | 0.46s | Includes ~236s model download |
| Second (cached) | 1.37s | 0.44s | Model already downloaded |

**Conclusions**:
1. Model download adds ~236 seconds (4 minutes) on first run for `small` model
2. Subprocess overhead is ~0.93s (1.37s - 0.44s) when cached
3. Internal transcription time is consistent (~0.45s) across runs
4. Total downloaded size for `small` model is **487MB**

---

### Large Model Tests (Timeout & Failures ❌)

#### Test 1: Fresh Download Attempt (No Cache)

**Initial state**: Empty cache folder
```
$ ls -la ~/Documents/huggingface/models/argmaxinc/
total 16
drwxr-xr-x@ 3 rymalia  staff    96 Jan 10 23:43 .
drwxr-xr-x@ 5 rymalia  staff   160 Jan 11 11:15 ..
-rw-r--r--@ 1 rymalia  staff  6148 Jan 10 23:44 .DS_Store
```

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

**Terminal output**:
```
2026-01-11 11:20:02 [info] Using WhisperKit large-v3 model (requested: large, using: large-v3)
2026-01-11 11:20:02 [info] Transcribing with WhisperKit via Swift bridge (model: large-v3)
2026-01-11 11:25:02 [error] WhisperKit bridge timed out
```

**Result**: **TIMEOUT after exactly 300 seconds**

**Partial cache state after timeout**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*
372K    MelSpectrogram.mlmodelc    # Complete
4.0K    TextDecoder.mlmodelc       # INCOMPLETE (should be ~1.7GB!)
```

**Orphaned temp file**:
```
$ ls -lh /private/var/folders/*/T/CFNetworkDownload*.tmp
611M    CFNetworkDownload_1iOtaj.tmp   # Partial download, abandoned
```

**Key observations**:
1. Download achieved 611MB in 5 minutes (~2 MB/s)
2. MelSpectrogram completed (small file)
3. TextDecoder folder created but INCOMPLETE (only analytics subfolder, 4K)
4. AudioEncoder not even started
5. 611MB temp file orphaned (wasted bandwidth)

#### Test 2: Retry with Partial Cache (Bug Discovered!)

**Initial state**: Partial cache from Test 1
- MelSpectrogram.mlmodelc: 372K (complete)
- TextDecoder.mlmodelc: 4K (INCOMPLETE)
- AudioEncoder.mlmodelc: Missing

**Command run** (same as Test 1):
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

**Result**: **TIMEOUT after 300 seconds**

**Cache state after second attempt**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*
400K    AudioEncoder.mlmodelc      # NEW but INCOMPLETE (empty weights folder!)
372K    MelSpectrogram.mlmodelc    # Unchanged (skipped - already existed)
4.0K    TextDecoder.mlmodelc       # Unchanged (SKIPPED despite being incomplete!)
```

**New orphaned temp file**:
```
477M    CFNetworkDownload_vx8Yul.tmp   # NEW partial download from run 2
```

**CRITICAL FINDINGS**:

| File | Run 1 | Run 2 | Expected | Bug? |
|------|-------|-------|----------|------|
| MelSpectrogram | 372K | 372K (skipped) | ~372K | No - correctly skipped |
| TextDecoder | 4K | 4K (skipped!) | ~1.7GB | **YES - incomplete file treated as complete!** |
| AudioEncoder | Missing | 400K (incomplete) | ~1.2GB | Partial - weights folder empty |

#### Orphaned Temp Files Summary

| File | Size | Date | Source |
|------|------|------|--------|
| CFNetworkDownload_1iOtaj.tmp | 611MB | Jan 11 11:25 | Test 1 - partial download |
| CFNetworkDownload_vx8Yul.tmp | 477MB | Jan 11 11:32 | Test 2 - partial download |
| CFNetworkDownload_O2M1Za.tmp | 362MB | Jan 11 00:20 | Previous session |
| CFNetworkDownload_YeAiwA.tmp | 318MB | Jan 11 00:57 | Previous session |

**Total wasted disk space**: ~1.8GB of orphaned temp files

#### Test 3: Direct Swift Bridge Download (No Python Timeout)

After cleaning up the corrupt cache, ran the Swift bridge directly without Python's 300s timeout.

**Cleanup performed**:
```bash
rm -rf ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/
rm -f /private/var/folders/*/T/CFNetworkDownload_*.tmp
```

**Command run**:
```bash
tools/whisperkit-bridge/.build/release/whisperkit-bridge tests/jfk.wav --model large-v3 --format json
```

**First attempt** (failed after 23:50):
- Downloaded AudioEncoder completely (1.2GB)
- Downloaded MelSpectrogram completely (392K)
- Started TextDecoder but CDN timed out
- Error: `NSURLErrorDomain Code=-1001 "The request timed out"`

**Second attempt** (succeeded after 27:31):
- Skipped AudioEncoder (already complete)
- Skipped MelSpectrogram (already complete)
- Completed TextDecoder download (1.7GB)
- Downloaded tokenizer files (2.6MB)
- **Success!** Transcription returned correctly

**Total download time**: ~51 minutes across two attempts (network conditions vary)

**Final cache state**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*
1.2G    AudioEncoder.mlmodelc
4.0K    config.json
4.0K    generation_config.json
392K    MelSpectrogram.mlmodelc
2.6M    models (tokenizer)
1.7G    TextDecoder.mlmodelc

Total: 2.9GB
```

#### Test 4: Cached Model Performance

With the model fully cached, ran the Python benchmark to verify cached behavior.

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

**Terminal output**:
```
2026-01-11 13:59:31 [info] Using WhisperKit large-v3 model (requested: large, using: large-v3)
2026-01-11 13:59:31 [info] Transcribing with WhisperKit via Swift bridge (model: large-v3)
2026-01-11 13:59:38 [info] WhisperKit transcription time: 3.0519s
2026-01-11 13:59:38 [info] Using internal transcription time: 3.0519s (total with overhead: 6.9720s)

=== Benchmark Summary for 'large' model ===
Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
whisperkit             3.0519          model=large-v3, backend=WhisperKit Swift Bridge
    "And so my fellow Americans, ask not what your country can do for you..."
```

**Key metrics**:

| Metric | Value |
|--------|-------|
| Total time (with overhead) | **6.97 seconds** |
| Internal transcription time | **3.05 seconds** |
| Subprocess overhead | ~3.9 seconds |
| Download required | **NO** |

#### Large Model Performance Comparison

**Comparison with small model** (from previous tests):

| Model | Total (cached) | Internal | Subprocess Overhead | Model Size |
|-------|----------------|----------|---------------------|------------|
| small | 1.37s | 0.44s | ~0.93s | 487MB |
| large-v3 | 6.97s | 3.05s | ~3.9s | 2.9GB |

**Observations**:
1. Large model subprocess overhead is ~4x higher than small (likely due to CoreML model loading time)
2. Internal transcription is ~7x slower than small (expected for 6x larger model)
3. Cached behavior works correctly - no re-download attempted
4. **First-run download always fails** due to timeout — large models require manual workaround

---

## Known Issues / Conflicts Discovered

### P0 (Critical) - Timeout Insufficient for Large Model

**Problem**: The 300-second (5 minute) subprocess timeout is insufficient for downloading the ~2.9GB large-v3 model.

**Impact**: Large model download **always fails** on first run. Users cannot use large models without manual intervention.

**Evidence**:
- Empirical: 611MB downloaded in 300 seconds (~2 MB/s)
- At 2 MB/s, ~2.9GB requires ~1450 seconds (~24 minutes)
- 300s timeout is **~5x too short**

**Location**: **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:128`
```python
timeout=300  # 5 minute timeout for model download on first run
```

**Recommended Fix**:
```python
timeout=1200  # 20 minute timeout for large model download
```

**Effort**: 1 line change
**Priority**: P0 - Blocks large model usage entirely

---

### P0 (Critical) - No Completeness Check on Cached Files

**Problem**: The Hub library only checks `FileManager.default.fileExists()` - it does NOT verify file size, checksum, or completeness. Partially downloaded files are treated as complete.

**Impact**:
- Incomplete downloads create corrupt cache entries
- Subsequent runs skip re-downloading incomplete files
- **Permanent failure state** - must manually delete cache to recover

**Evidence**:
- 4K TextDecoder.mlmodelc folder (should be ~1.7GB) was SKIPPED on run 2
- The folder exists (contains only `analytics/` subfolder), so `fileExists()` returns true

**Location**: **[LIBRARY]** `swift-transformers/Sources/Hub/HubApi.swift:204-206`
```swift
var downloaded: Bool {
    FileManager.default.fileExists(atPath: destination.path)
}
```

**Recommended Fix** (in HubApi.swift):
```swift
var downloaded: Bool {
    guard FileManager.default.fileExists(atPath: destination.path) else { return false }
    // For directories (like .mlmodelc), check for key files
    if destination.pathExtension == "mlmodelc" {
        let weightPath = destination.appendingPathComponent("weights/weight.bin")
        return FileManager.default.fileExists(atPath: weightPath.path)
    }
    // For regular files, could add size check via HEAD request
    return true
}
```

**Effort**: ~20 lines in library code
**Priority**: P0 - Causes permanent failure state

---

### P1 (High) - No Download Resume Capability

**Problem**: Downloads use URLSession which writes to temp files. If interrupted, temp files are orphaned and download restarts from zero on next attempt.

**Impact**:
- Large model downloads waste bandwidth on retry
- ~1.8GB of orphaned temp files observed
- Users on slow connections may never complete large model download

**Evidence**:
- Run 1: Created 611MB temp file, timed out
- Run 2: Created NEW 477MB temp file (didn't resume 611MB file)

**Location**: **[LIBRARY]** `swift-transformers/Sources/Hub/Downloader.swift:78`
```swift
self.urlSession?.downloadTask(with: request).resume()
```

**Recommended Fix**: Implement Range header support for resumable downloads, or use background URLSession with proper state persistence.

**Effort**: Large (significant library changes)
**Priority**: P1 - Wastes bandwidth, poor UX

---

### P2 (Medium) - Orphaned Temp Files Not Cleaned Up

**Problem**: When downloads are interrupted (timeout, crash, Ctrl+C), temp files in `/private/var/folders/*/T/CFNetworkDownload_*.tmp` are not cleaned up.

**Impact**:
- Disk space waste (observed 1.8GB)
- No automatic cleanup mechanism
- Users unaware of accumulated temp files

**Location**: System-level URLSession behavior

**Recommended Fix**: Add cleanup logic to delete stale CFNetworkDownload temp files on startup, or document manual cleanup procedure.

**Effort**: ~20 lines
**Priority**: P2 - Disk space waste

---

### P2 (Medium) - No Progress Feedback During Download

**Problem**: Model download happens inside subprocess with no progress output. Users see the process hang for minutes with no feedback.

**Impact**: Poor UX - users may think process is frozen and kill it (causing partial download)

**Location**: **[BRIDGE]** `main.swift` - no progress callback implemented

**Recommended Fix**: Add progress output to stderr from Swift bridge:
```swift
let modelFolder = try await Self.download(variant: modelVariant, ...) { progress in
    fputs("Download progress: \(Int(progress.fractionCompleted * 100))%\n", stderr)
}
```

**Effort**: ~10 lines
**Priority**: P2 - Poor UX

---

### P3 (Future Considerations) - Architectural Trade-offs

These are inherent design limitations rather than bugs, documented for completeness:

#### 1. Cache Location Differs from Python Libraries

WhisperKit uses `~/Documents/huggingface/` while Python libraries use `~/.cache/huggingface/hub/`. This means:
- No model sharing between WhisperKit and MLX/Transformers implementations
- Potential for duplicate large model downloads (487MB+ per model)

**Impact**: Disk space inefficiency, but required by Swift's Hub library design
**Priority**: P3 - Ecosystem consistency issue

#### 2. Subprocess Overhead

Each transcription incurs ~0.9s overhead from:
- Process spawn time
- Temp file I/O
- JSON parsing
- CoreML model loading

**Impact**: For short audio clips, this overhead may be significant relative to transcription time
**Priority**: P3 - Acceptable trade-off for native CoreML access

#### 3. No Streaming Support

The bridge pattern doesn't support streaming transcription — the entire audio must be written to a file, processed, and results returned as JSON.

**Impact**: Cannot process live audio streams
**Priority**: P3 - Would require architectural redesign

#### 4. First-Run Download in Subprocess

Model download happens inside the subprocess, not in Python. This means:
- No Python-side progress reporting (addressed in P2 above)
- Error handling relies on stderr parsing
- Timeout applies to entire subprocess (download + transcription)

**Impact**: Poor UX for first-run experience
**Priority**: P3 - Related to P0 timeout issue

---

## Recommended Improvements

### Priority Summary

| Priority | Issue | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| P0 | Timeout insufficient (300s → 1200s) | 1 line | Blocks large model | Not Fixed ❌ |
| P0 | No completeness check | ~20 lines (library) | Permanent failure state | Not Fixed ❌ |
| P1 | No download resume | Large | Bandwidth waste | Not Fixed ❌ |
| P2 | Orphaned temp files | ~20 lines | Disk waste | Not Fixed ❌ |
| P2 | No progress feedback | ~10 lines | Poor UX | Not Fixed ❌ |
| P3 | Cache location differs | N/A | Ecosystem issue | By Design |
| P3 | Subprocess overhead | N/A | Minor perf impact | By Design |
| P3 | No streaming support | Large | Feature gap | By Design |

### Implementation Order Recommendation

**Phase 1: Immediate Fixes (Project-level)**
- [ ] Increase timeout from 300s to 1200s in `whisperkit.py:128`
- [ ] Add documentation warning about first-run download time for large models
- [ ] Add cleanup script for orphaned temp files

**Phase 2: Bridge Improvements**
- [ ] Add progress output to stderr during download
- [ ] Add `--timeout` argument to bridge for configurable timeout

**Phase 3: Library Fixes (Requires PR to swift-transformers)**
- [ ] Add completeness check for .mlmodelc directories
- [ ] Implement resumable downloads with Range headers

---

## Workarounds

### Manual Large Model Download

If the timeout prevents automatic download, manually run the bridge with no timeout:

```bash
# Run bridge directly (will complete download)
cd tools/whisperkit-bridge
.build/release/whisperkit-bridge /path/to/any/audio.wav --model large-v3

# This will download the model (may take 15-30+ minutes)
# Once complete, subsequent benchmark runs will use cached model
```

### Clean Corrupt Cache

If cache is in a corrupt state (incomplete files treated as complete):

```bash
# Delete the specific model folder
rm -rf ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/

# Clean orphaned temp files
rm -f /private/var/folders/*/T/CFNetworkDownload_*.tmp
```

### Use Turbo Model Instead

The `large-v3-turbo` model is significantly smaller (~500MB vs ~2.9GB) and may complete within the timeout:

```bash
.venv/bin/python3 test_benchmark2.py large-v3-turbo 1 WhisperKitImplementation
```

---

## Completion Checklist

Documentation completeness criteria:

- [x] Code analysis flow documented for both `small` and `large` model paths
- [x] Benchmark ACTUALLY run for **BOTH** `small` AND `large` models
- [x] Terminal output from benchmark runs included for **BOTH sizes**
- [x] Model file locations verified with `ls` commands for **BOTH sizes**
- [x] "Empirical Test Results" section contains actual observed data (not inferred)
- [x] "Key Questions Answered" table included near top of document
- [x] "Recommended Improvements" section includes improvement proposals
- [x] Priority Summary table included with effort estimates and status tracking
- [x] Implementation Order Recommendation included with phased checkboxes
- [x] Large model timeout documented as P0 issue with workaround
