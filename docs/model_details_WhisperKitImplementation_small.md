# Model Details: WhisperKitImplementation

This document traces the complete execution flow for the `WhisperKitImplementation`, documenting how models are downloaded, cached, and loaded for transcription.

---

## File Reference Legend

Throughout this document, files are categorized as:
- **[PROJECT]** - Files in this repository that you can modify
- **[LIBRARY]** - Installed package files (Swift/Python dependencies)
- **[BRIDGE]** - The Swift CLI executable that interfaces with WhisperKit

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
        model: config.model,        // "small"
        downloadBase: nil,          // Uses default
        modelRepo: nil,             // Uses "argmaxinc/whisperkit-coreml"
        download: true
    )
    ```

14. **Model Download** — **[LIBRARY]** `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:243-295`
    - Uses `HubApi` from `swift-transformers` package
    - Default repo: `argmaxinc/whisperkit-coreml`
    - Searches for model files matching: `*small/*`
    - Downloads via `hubApi.snapshot()` to local cache

15. **HuggingFace Hub Cache** — **[LIBRARY]** `swift-transformers/Sources/Hub/HubApi.swift:19-29`
    ```swift
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    self.downloadBase = documents.appending(component: "huggingface")
    ```
    - Default cache location: `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/`

16. **Model Loading** — **[LIBRARY]** `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:337-458`
    - Loads CoreML models from disk:
      - `MelSpectrogram.mlmodelc` (feature extractor)
      - `AudioEncoder.mlmodelc`
      - `TextDecoder.mlmodelc`
      - `TextDecoderContextPrefill.mlmodelc` (optional, for faster decoding)
    - Loads tokenizer from HuggingFace (e.g., `openai/whisper-small`)

17. **Transcription Execution** — **[LIBRARY]** WhisperKit internal
    - Runs audio through Mel spectrogram → Audio encoder → Text decoder
    - Returns segments with timestamps

18. **Results** — **[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:165-171`
    - Stores `BenchmarkResult` with timing and transcription text
    - Uses internal `_transcription_time` from WhisperKit (excludes subprocess overhead)
    - Calls `implementation.get_params()` → `{"model": "small", "backend": "WhisperKit Swift Bridge", "platform": "Apple Silicon"}`

---

## Summary Table

| Attribute | Value |
|-----------|-------|
| **Requested Model** | `small` |
| **Mapped Model Name** | `small` (no change) |
| **HuggingFace Repo ID** | `argmaxinc/whisperkit-coreml` |
| **Download URL Base** | `https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main/` |
| **Model Search Pattern** | `*small/*` (or `*openai*small/*` if ambiguous) |
| **Files Downloaded** | `MelSpectrogram.mlmodelc/`, `AudioEncoder.mlmodelc/`, `TextDecoder.mlmodelc/` |
| **Cache Location** | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/` |
| **Tokenizer Source** | `openai/whisper-small` from HuggingFace |
| **Backend** | Swift/CoreML via subprocess bridge |

---

## Model Mapping Reference

### Project-Level Mapping — **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:58-67`

| Input | Output | Notes |
|-------|--------|-------|
| `"tiny"` | `"tiny"` | No change |
| `"base"` | `"base"` | No change |
| `"small"` | `"small"` | No change |
| `"medium"` | `"medium"` | No change |
| `"large"` | `"large-v3"` | **Upgraded to latest version** |
| `"large-v3"` | `"large-v3"` | No change |
| `"large-v3-turbo"` | `"large-v3-turbo"` | Explicit turbo access |
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
|---------|----------------------|
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

**[PROJECT]** `src/mac_whisper_speedtest/benchmark.py:149-152` uses this internal timing when available:
```python
if hasattr(result, '_transcription_time'):
    run_time = result._transcription_time
```

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

---

## Key Source Files

### Project Files (You Can Modify)

| File | Purpose |
|------|---------|
| `test_benchmark2.py` | Entry point, CLI argument parsing |
| `src/mac_whisper_speedtest/benchmark.py` | Benchmark orchestration, timing |
| `src/mac_whisper_speedtest/implementations/whisperkit.py` | Implementation wrapper, subprocess call |

### Bridge Files (Swift — Compile with `swift build`)

| File | Purpose |
|------|---------|
| `tools/whisperkit-bridge/Package.swift` | Swift package manifest, WhisperKit dependency |
| `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift` | CLI entry point, JSON output |

### Library Files (Swift Package Dependencies — Read-Only)

| File | Purpose |
|------|---------|
| `WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift` | Main WhisperKit class, model setup |
| `WhisperKit/Sources/WhisperKit/Core/Configurations.swift` | WhisperKitConfig, DecodingOptions |
| `WhisperKit/Sources/WhisperKit/Utilities/ModelUtilities.swift` | Model detection, tokenizer loading |
| `swift-transformers/Sources/Hub/HubApi.swift` | HuggingFace download, cache management |

---

## Empirical Test Results

**Test Date**: January 10, 2026

### Test 1: Fresh Model Download (No Cache)

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

### Files Downloaded

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

**Tokenizer files** (in `models/openai/whisper-small/`):
```
$ ls -la ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/models/openai/whisper-small/
total 5416
drwxr-xr-x@ 5 rymalia  staff      160 Jan 10 21:18 .
drwxr-xr-x@ 3 rymalia  staff       96 Jan 10 21:18 ..
-rw-------@ 1 rymalia  staff     1967 Jan 10 21:18 config.json
-rw-------@ 1 rymalia  staff   282683 Jan 10 21:18 tokenizer_config.json
-rw-------@ 1 rymalia  staff  2480466 Jan 10 21:18 tokenizer.json
```

**Total model size**: **487MB**

### Test 2: Cached Model (Second Run)

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

### Performance Comparison

| Run | Total Time (with overhead) | Internal Transcription | Notes |
|-----|---------------------------|------------------------|-------|
| First (cold) | 237.35s | 0.46s | Includes ~236s model download |
| Second (cached) | 1.37s | 0.44s | Model already downloaded |

**Conclusions**:
1. Model download adds ~236 seconds (4 minutes) on first run for `small` model
2. Subprocess overhead is ~0.93s (1.37s - 0.44s) when cached
3. Internal transcription time is consistent (~0.45s) across runs
4. Total downloaded size for `small` model is **487MB**

### Cache Location Confirmed

**Primary cache**:
```
~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small/
```

**Note**: This is different from Python's HuggingFace cache (`~/.cache/huggingface/hub/`), so models are NOT shared between WhisperKit and Python MLX implementations.

---

## Known Issues / Conflicts Discovered

### 1. Cache Location Differs from Python Libraries

WhisperKit uses `~/Documents/huggingface/` while Python libraries use `~/.cache/huggingface/hub/`. This means:
- No model sharing between WhisperKit and MLX/Transformers implementations
- Potential for duplicate large model downloads (487MB+ per model)

### 2. Subprocess Overhead

Each transcription incurs ~0.9s overhead from:
- Process spawn time
- Temp file I/O
- JSON parsing
- CoreML model loading

For short audio clips, this overhead may be significant relative to transcription time.

### 3. No Streaming Support

The bridge pattern doesn't support streaming transcription — the entire audio must be written to a file, processed, and results returned as JSON.

### 4. First-Run Download in Subprocess

Model download happens inside the subprocess, not in Python. This means:
- No Python-side progress reporting
- 5-minute timeout may be insufficient for large models on slow connections
- If download fails, error comes through stderr parsing
