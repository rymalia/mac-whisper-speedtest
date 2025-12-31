# FluidAudioCoreMLImplementation - Execution Flow Documentation

This document traces the execution flow for `FluidAudioCoreMLImplementation`, which uses a Swift bridge to the [FluidAudio](https://github.com/FluidInference/FluidAudio) framework. Unlike other implementations, FluidAudio uses a **fixed model** (Parakeet TDT 0.6B v3) for all transcription requests regardless of the requested model size.

## Overview

| Aspect | Value |
|--------|-------|
| **Backend** | FluidAudio Swift framework |
| **Integration** | Swift bridge via subprocess |
| **Model** | Parakeet TDT 0.6B v3 (fixed, ~600M parameters) |
| **Model Format** | CoreML compiled models (.mlmodelc) |
| **Download Source** | HuggingFace via Swift URLSession (not Python huggingface_hub) |
| **Model Repository** | `FluidInference/parakeet-tdt-0.6b-v3-coreml` |
| **Cache Location** | `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/` |
| **Equivalent Whisper Size** | Roughly small/base (~600M params) |

---

## Benchmark Execution Flow

**Command:**
```bash
.venv/bin/python3 test_benchmark.py medium 1 FluidAudioCoreMLImplementation
```

### Step-by-Step Flow

1. **Entry Point** (`test_benchmark.py:76-82`)
   - Parses command line: `model="medium"`, `runs=1`, `implementations="FluidAudioCoreMLImplementation"`
   - Calls `asyncio.run(main("medium", 1, "FluidAudioCoreMLImplementation"))`

2. **Load Audio** (`test_benchmark.py:24-43`)
   - Loads test audio from `tools/whisperkit-bridge/.build/checkouts/WhisperKit/Tests/WhisperKitTests/Resources/jfk.wav`
   - Converts to 16kHz mono float32 format

3. **Get Implementation** (`test_benchmark.py:46-62`)
   - Calls `get_all_implementations()`
   - Filters to only `FluidAudioCoreMLImplementation`

4. **Create Benchmark Config** (`test_benchmark.py:65-74`)
   ```python
   config = BenchmarkConfig(
       model_name="medium",  # Note: FluidAudio ignores this!
       implementations=[FluidAudioCoreMLImplementation],
       num_runs=1,
       audio_data=whisper_audio,
   )
   ```

5. **Run Benchmark** (`benchmark.py:121-202`)
   - For each implementation:
     - Creates instance: `implementation = FluidAudioCoreMLImplementation()`
     - Calls `implementation.load_model("medium")`
     - For each run: times `await implementation.transcribe(audio_data)`

6. **FluidAudioCoreMLImplementation.__init__()** (`fluidaudio_coreml.py:21-36`)
   - Verifies macOS platform
   - Calls `_find_bridge_executable()` to locate Swift bridge at:
     `tools/fluidaudio-bridge/.build/release/fluidaudio-bridge`

7. **load_model("medium")** (`fluidaudio_coreml.py:51-78`)
   - **IMPORTANT**: `model_name` is stored but **NOT USED** for model selection
   - FluidAudio uses a **fixed model** (Parakeet TDT 0.6B v3) for all sizes
   - Only validates bridge is working via `subprocess.run([bridge, "--help"])`
   - Actual model loading happens in Swift during `transcribe()`

8. **transcribe(audio)** (`fluidaudio_coreml.py:82-156`)
   - Preprocesses audio: float32, mono, normalized to [-1, 1]
   - Saves audio to temporary WAV file (16kHz mono)
   - Calls Swift bridge: `subprocess.run([bridge_path, temp_path, "--format", "json"], timeout=300)`
   - Parses JSON response with text and timing
   - Sets `result._transcription_time = transcription_time` for accurate benchmarking

9. **Swift Bridge Execution** (`main.swift:20-83`)
   - Loads audio file using AVFoundation
   - Creates ASR config: `ASRConfig.default`
   - **Downloads/loads model** via `AsrModels.downloadAndLoad()`

10. **AsrModels.downloadAndLoad()** (`AsrModels.swift:340-347`)
    ```swift
    // Default version is .v3
    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3  // <-- Always v3
    ) async throws -> AsrModels
    ```
    - Downloads to: `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/`
    - Downloads from: `https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml`
    - Loads CoreML models with `.cpuAndNeuralEngine` compute units

11. **Swift Transcription** (`main.swift:58-78`)
    - Transcribes audio via `asrManager.transcribe(audioData)`
    - Returns JSON with `text`, `transcription_time`, `processing_time`

12. **Display Results** (`benchmark.py:176-183` → `BenchmarkSummary.print_summary()`)
    - Shows transcription time (using internal Swift timing)
    - Shows transcription text

---

## Summary Table

| Attribute | Value |
|-----------|-------|
| **Requested Model** | `medium` (or any size - ignored) |
| **Actual Model Used** | `parakeet-tdt-0.6b-v3-coreml` |
| **HuggingFace Repo** | `FluidInference/parakeet-tdt-0.6b-v3-coreml` |
| **Download URL Base** | `https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/` |
| **Cache Directory** | `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/` |
| **Files Downloaded** | `Encoder.mlmodelc/`, `Decoder.mlmodelc/`, `Preprocessor.mlmodelc/`, `JointDecision.mlmodelc/`, `parakeet_vocab.json` |
| **Total Size** | ~460 MB |

---

## Model Files

| File | Purpose | Compute Units |
|------|---------|---------------|
| `Preprocessor.mlmodelc/` | Audio preprocessing / Mel spectrogram | CPU only |
| `Encoder.mlmodelc/` | Audio encoder (~425 MB) | CPU + Neural Engine |
| `Decoder.mlmodelc/` | Text decoder | CPU + Neural Engine |
| `JointDecision.mlmodelc/` | Joint decision model | CPU + Neural Engine |
| `parakeet_vocab.json` | Vocabulary file with token mappings | N/A |

---

## Notes

1. **Fixed Model Architecture**: FluidAudio is fundamentally different from Whisper implementations. It uses the Parakeet TDT (Token Duration Transducer) architecture which is a streaming-optimized ASR model from NVIDIA NeMo, converted to CoreML.

2. **Model Size Equivalence**: Parakeet TDT 0.6B has ~600M parameters, roughly equivalent to Whisper small/base models.

3. **Model Selection Transparency**: The implementation logs a warning when non-small/base models are requested:
   ```python
   if model_name not in ["small", "base"]:
       self.log.info(
           f"FluidAudio uses parakeet-tdt-0.6b-v3-coreml for all model sizes "
           f"(requested: {model_name}). Parakeet is roughly equivalent to "
           f"Whisper small/base models."
       )
   ```

4. **Cache Location**: Unlike most HuggingFace-based implementations that use `~/.cache/huggingface/hub/`, FluidAudio uses macOS Application Support directory (`~/Library/Application Support/FluidAudio/Models/`).

5. **Internal Timing**: The Swift bridge reports internal transcription time separately from subprocess overhead, allowing accurate benchmarking.

6. **Outdated Comment**: The Swift bridge has a comment saying "v2" but the actual code uses v3. This is an outdated comment.

---

## check-models Command Flow

**Command:**
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations FluidAudioCoreMLImplementation
```

### Step-by-Step Flow

1. **CLI Entry Point** (`cli.py:101-102`)
   ```python
   @app.command(name="check-models")
   def check_models(model: str = "small", implementations: Optional[str] = None, ...):
   ```
   - Parses `--model medium --implementations FluidAudioCoreMLImplementation`

2. **Filter Implementations** (`cli.py:112-126`)
   - Gets all implementations via `get_all_implementations()`
   - Filters to only `FluidAudioCoreMLImplementation`

3. **Create ModelChecker** (`cli.py:130-131`)
   ```python
   checker = ModelChecker(verify_method=verify_method, verbose=verbose)
   ```

4. **Check All Models** (`check_models.py:171-216`)
   ```python
   statuses = checker.check_all_models("medium", implementations)
   ```

5. **For FluidAudioCoreMLImplementation** (`check_models.py:191-208`):
   - Creates instance: `impl = FluidAudioCoreMLImplementation()`
   - Calls `model_info = impl.get_model_info("medium")`

6. **get_model_info("medium")** (`fluidaudio_coreml.py:210-261`)
   - **Returns FIXED model info regardless of `model_name`**:
   ```python
   return ModelInfo(
       model_name="parakeet-tdt-0.6b-v3-coreml",
       repo_id="FluidInference/parakeet-tdt-0.6b-v3-coreml",
       cache_paths=[
           model_dir / "Encoder.mlmodelc",
           model_dir / "Decoder.mlmodelc",
           model_dir / "Preprocessor.mlmodelc",
           model_dir / "JointDecision.mlmodelc",
       ],
       expected_size_mb=460,
       verification_method="size",
       download_trigger="bridge"
   )
   ```
   - Where `model_dir = ~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml`

7. **Check HF Cache** (`check_models.py:218-238`)
   - Since `repo_id` exists but `verification_method="size"` (not "huggingface"):
   - Returns `"n/a"` for HF cache status

8. **Check Local Cache** (`check_models.py:337-384`)
   - Checks if `cache_paths` exist:
     - `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/Encoder.mlmodelc`
     - `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/Decoder.mlmodelc`
     - `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/Preprocessor.mlmodelc`
     - `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/JointDecision.mlmodelc`
   - Calculates total size and compares to expected 460 MB (±10%)
   - Returns status: `"complete"`, `"missing"`, or `"incomplete"`

9. **Print Status Table** (`check_models.py:407-440`)
   - Displays table with implementation, model, HF Hub Cache, Local Cache, and Disk Usage

10. **Download if Requested** (`check_models.py:524-530`)
    - For FluidAudio, uses `download_trigger="bridge"`
    - Triggers download by running bridge with dummy audio (`check_models.py:540-610`):
      ```python
      bridge_path = project_root / "tools" / "fluidaudio-bridge" / ".build" / "release" / "fluidaudio-bridge"
      # Creates 1 second of silence, runs bridge, which downloads model
      ```

---

## check-models Summary Table

| Attribute | Value |
|-----------|-------|
| **ModelInfo.model_name** | `parakeet-tdt-0.6b-v3-coreml` |
| **ModelInfo.repo_id** | `FluidInference/parakeet-tdt-0.6b-v3-coreml` |
| **ModelInfo.verification_method** | `size` |
| **ModelInfo.download_trigger** | `bridge` |
| **ModelInfo.expected_size_mb** | 460 |
| **Cache Paths Checked** | 4 `.mlmodelc` directories under `~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml/` |

---

## Variant Mismatch Analysis

### Current State: No Active Mismatch

Currently, both the Swift bridge and Python wrapper are hardcoded to v3, so there is no mismatch **today**:

| Component | Model Used | Notes |
|-----------|------------|-------|
| `load_model("tiny")` | `parakeet-tdt-0.6b-v3-coreml` | Ignores parameter |
| `load_model("medium")` | `parakeet-tdt-0.6b-v3-coreml` | Ignores parameter |
| `load_model("large")` | `parakeet-tdt-0.6b-v3-coreml` | Ignores parameter |
| `get_model_info("tiny")` | `parakeet-tdt-0.6b-v3-coreml` | Returns fixed info |
| `get_model_info("medium")` | `parakeet-tdt-0.6b-v3-coreml` | Returns fixed info |
| `get_model_info("large")` | `parakeet-tdt-0.6b-v3-coreml` | Returns fixed info |

---

### Known Issue: Potential Variant Mismatch (Latent)

**The architecture has a latent mismatch vulnerability** due to model names being hardcoded in two separate codebases with no programmatic link.

#### Hardcoded Locations

| Component | File | Line | Controls |
|-----------|------|------|----------|
| **Swift Bridge** | `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift` | 48 | What model is **actually downloaded/used** during benchmark |
| **Python Wrapper** | `src/mac_whisper_speedtest/implementations/fluidaudio_coreml.py` | 245-256 | What model **check-models looks for** |

#### Swift Bridge Code
```swift
// main.swift:48 - Uses default version parameter
let models = try await AsrModels.downloadAndLoad()  // Defaults to .v3
```

#### Python Wrapper Code
```python
# fluidaudio_coreml.py:245-256 - Hardcoded to v3
model_dir = home / "Library" / "Application Support" / "FluidAudio" / "Models" / "parakeet-tdt-0.6b-v3-coreml"
return ModelInfo(
    model_name="parakeet-tdt-0.6b-v3-coreml",
    repo_id="FluidInference/parakeet-tdt-0.6b-v3-coreml",
    ...
)
```

#### Mismatch Scenario

If someone updates the Swift bridge to use v2:
```swift
// main.swift - changed to v2
let models = try await AsrModels.downloadAndLoad(version: .v2)
```

But forgets to update the Python wrapper:
```python
# fluidaudio_coreml.py - still says v3!
model_dir = ... / "parakeet-tdt-0.6b-v3-coreml"
repo_id="FluidInference/parakeet-tdt-0.6b-v3-coreml"
```

Then:
- **Benchmark** → downloads and uses **v2** (correct)
- **check-models** → looks for **v3** (wrong!) → reports "missing" even though v2 is cached

#### Comparison to Best Practice

Other implementations use `_get_model_map()` as a **single source of truth**:
```python
def _get_model_map(self) -> Dict[str, str]:
    return {"small": "mlx-community/whisper-small-mlx-4bit", ...}

def load_model(self, model_name):
    repo_id = self._map_model_name(model_name)  # Uses the map

def get_model_info(self, model_name):
    repo_id = self._map_model_name(model_name)  # Uses same map
```

FluidAudio **cannot follow this pattern** because:
1. The actual model selection happens in **Swift** (not Python)
2. Python has no way to query what model the Swift bridge will use
3. The Python wrapper must guess/assume it matches the Swift code

#### Root Cause

The Swift bridge and Python wrapper are separate codebases that must stay synchronized manually.

#### Impact

- **Current**: None (both hardcoded to v3)
- **Future**: Any change to model version requires updates in **both** codebases, or `check-models` will report incorrect status

#### TODO: Potential Fixes

1. **Add version flag to Swift bridge**: Make the bridge report its configured model version via a `--version-info` flag that Python can query
2. **Shared config file**: Create a shared JSON config file that both Swift and Python read
3. **Bridge introspection**: Have the Swift bridge output model info in its JSON response, which Python could cache
4. **Documentation**: At minimum, add comments in both files pointing to the other location that must be updated

### Architectural Note

FluidAudio's Parakeet TDT model is not a Whisper variant - it's a completely different ASR architecture (Token Duration Transducer from NVIDIA NeMo). The framework supports two versions (v2 and v3), but the model size (0.6B parameters) is fixed.

---

## Key Files Analyzed

| File | Purpose |
|------|---------|
| `src/mac_whisper_speedtest/implementations/fluidaudio_coreml.py` | Python wrapper for FluidAudio Swift bridge |
| `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift` | Swift CLI bridge executable |
| `tools/fluidaudio-bridge/Package.swift` | Swift package manifest (FluidAudio 0.7.12+) |
| `.build/checkouts/FluidAudio/Sources/FluidAudio/ASR/AsrModels.swift` | Model download/load logic |
| `.build/checkouts/FluidAudio/Sources/FluidAudio/ModelNames.swift` | Model repository definitions |
| `.build/checkouts/FluidAudio/Sources/FluidAudio/ModelRegistry.swift` | HuggingFace URL construction |
| `.build/checkouts/FluidAudio/Sources/FluidAudio/DownloadUtils.swift` | Download implementation |
| `src/mac_whisper_speedtest/check_models.py` | Model verification and download system |
| `src/mac_whisper_speedtest/benchmark.py` | Benchmark runner |

---

## Swift Bridge Architecture

```
Python (fluidaudio_coreml.py)
    │
    ├── __init__: Find bridge at tools/fluidaudio-bridge/.build/release/fluidaudio-bridge
    │
    ├── load_model(): Validate bridge works (no model loading here)
    │
    └── transcribe(audio):
          │
          ├── Preprocess audio (float32, mono, normalized)
          │
          ├── Save to temp WAV file
          │
          └── subprocess.run([bridge, audio.wav, --format, json])
                    │
                    └── Swift Bridge (main.swift)
                              │
                              ├── Load audio via AVFoundation
                              │
                              ├── AsrModels.downloadAndLoad()
                              │     │
                              │     ├── Check ~/Library/Application Support/FluidAudio/Models/
                              │     │
                              │     ├── If missing: Download from HuggingFace
                              │     │     └── https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/
                              │     │
                              │     └── Load CoreML models (Encoder, Decoder, Preprocessor, JointDecision)
                              │
                              ├── asrManager.transcribe(audioData)
                              │
                              └── Return JSON: {text, transcription_time, processing_time, language}
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Real-Time Factor** | ~110x RTF on M4 Pro |
| **Compute Strategy** | CPU + Neural Engine (no GPU to allow background execution on iOS) |
| **Streaming Support** | Yes (TDT architecture designed for streaming) |
| **Warmup Required** | Yes (CoreML model compilation on first load) |
| **Internal Timing** | Yes (excludes subprocess/bridge overhead) |

---

## Known Issues

### Download Reliability Issue

**Important**: This document traces the Swift code flow, which shows the *intended* behavior. Empirical testing reveals that the actual download behavior can differ in practice.

#### The Problem

The Swift `DownloadUtils.swift` code is designed to download models directly to Application Support via URLSession. However, in practice, the download mechanism can fail or hang indefinitely, leaving the bridge stuck at "Starting model load..." even when network connectivity is fine.

#### Observed Behavior

When Application Support cache is empty:
1. The bridge starts and reaches "Starting model load..."
2. The Swift download mechanism attempts to fetch from HuggingFace
3. The download may hang indefinitely (observed: >30 seconds with no progress)
4. Meanwhile, models may exist in the Python HuggingFace cache (`~/.cache/huggingface/hub/`)

#### Root Cause

The Swift `DownloadUtils` and Python `huggingface_hub` use different cache locations:

| Component | Cache Location |
|-----------|---------------|
| Swift DownloadUtils | `~/Library/Application Support/FluidAudio/Models/` |
| Python huggingface_hub | `~/.cache/huggingface/hub/models--FluidInference--*` |

If models were previously downloaded via Python (e.g., during development, testing, or via `check-models`), they exist in the HF cache but the Swift bridge cannot find them.

#### Workaround

Use the `fix_models.sh` script to copy models from HF cache to Application Support:

```bash
./tools/fluidaudio-bridge/fix_models.sh
```

This script:
1. Locates models in `~/.cache/huggingface/hub/models--FluidInference--parakeet-tdt-0.6b-v3-coreml/`
2. Copies all required files to `~/Library/Application Support/FluidAudio/Models/`
3. Verifies completeness (Encoder, Decoder, Preprocessor, JointDecision, vocab, config)

#### Verification

After running `fix_models.sh`, verify the bridge works:

```bash
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge \
  tools/whisperkit-bridge/.build/checkouts/WhisperKit/Tests/WhisperKitTests/Resources/jfk.wav \
  --format json
```

Expected: Completes in <1 second with transcription output (after initial ~15s model compilation).

#### Documentation Note

This issue illustrates the difference between **code tracing** (documenting intended behavior from source code) and **empirical testing** (documenting actual observed behavior). The main body of this document traces the Swift code's intended flow. This section documents the practical reality observed through testing.

See also: `docs/MODEL_CACHING.md` for additional troubleshooting details.
