# GEMINI.md

## Project Overview

`mac-whisper-speedtest` is a specialized benchmarking tool designed to compare the performance and accuracy of various Automatic Speech Recognition (ASR) implementations on Apple Silicon Macs. While primarily focused on **Whisper** variants, it also supports other modern ASR models like **Parakeet**.

The project aims to find the "fastest transcription on my Mac" by measuring average transcription time across 9+ different implementations, ranging from native Apple CoreML frameworks to MLX-accelerated models and CPU-optimized variants.

### Technology Stack
- **Primary Language:** Python 3.11+
- **Native Components:** Swift 5.9+ (for WhisperKit and FluidAudio bridges)
- **ML Frameworks:** Apple MLX, CoreML, PyTorch (MPS), ctranslate2
- **CLI Framework:** Typer
- **Audio Processing:** NumPy, PyAudio, SoundFile, Librosa
- **Dependency Management:** uv

---

## Architecture

The project follows a **plugin-based architecture** to allow easy integration of new ASR engines.

### Key Components

- **`src/mac_whisper_speedtest/implementations/`**: Contains the various ASR wrappers.
    - **`base.py`**: Defines the `WhisperImplementation` abstract base class and result dataclasses (`TranscriptionResult`, `BenchmarkResult`).
    - **Implementations**: `mlx-whisper`, `faster-whisper`, `insanely-fast-whisper`, `whisperkit`, `fluidaudio-coreml`, `parakeet-mlx`, `lightning-whisper-mlx`, `whisper-mps`, and `whisper.cpp` (CoreML).
- **`benchmark.py`**: The core runner that handles model loading, multiple runs for statistical averaging, and internal timing detection (especially for Swift bridges).
- **`cli.py`**: The entry point providing two main modes:
    - **Interactive Mode**: Uses the microphone (PyAudio) to record live speech.
    - **Batch Mode (`--batch`)**: Uses pre-recorded audio files (e.g., `tests/jfk.wav`) for reproducible benchmarks.
- **`tools/`**: Contains native Swift bridges that communicate with the Python core via JSON over subprocesses.

---

## Building and Running

### Setup
1. **Python Environment:**
   ```bash
   uv sync
   ```
2. **Build Swift Bridges:** (Required for native implementations)
   ```bash
   # WhisperKit
   cd tools/whisperkit-bridge && swift build -c release && cd ../..
   # FluidAudio (Optional)
   cd tools/fluidaudio-bridge && swift build -c release && cd ../..
   ```

### Running Benchmarks
- **Default (Small model, all implementations):**
  ```bash
  .venv/bin/mac-whisper-speedtest
  ```
- **Batch Mode (Recommended for testing):**
  ```bash
  .venv/bin/mac-whisper-speedtest --batch --audio tests/jfk.wav
  ```
- **Specific Model and Implementations:**
  ```bash
  .venv/bin/mac-whisper-speedtest -m large -i "MLXWhisperImplementation,WhisperKitImplementation"
  ```

### Testing
```bash
pytest tests/ -v
```

---

## Development Conventions

- **Implementation Pattern:** New ASR engines should inherit from `WhisperImplementation` and implement `load_model`, `transcribe`, and `get_params`.
- **Conditional Imports:** Use `importlib.util.find_spec` in `implementations/__init__.py` to handle optional dependencies gracefully.
- **Platform Verification:** macOS-specific implementations should check `platform.system() == "Darwin"` during initialization.
- **Audio Standardization:** All audio is processed at **16kHz mono float32**.
- **Logging:** Uses `structlog` for structured, dev-friendly console output.
- **Model Fallbacks:** Implementations should ideally handle fallback chains (e.g., `large` -> `large-v3-turbo`) to ensure a model is always loaded if the specific requested variant is missing.
- **Timing:** Use internal timing (`_transcription_time` attribute) for bridges to exclude subprocess overhead from the benchmark.

---

## Project Status & Roadmap

- [x] 9 Whisper-based implementations
- [x] CoreML Auto-download for `whisper.cpp`
- [x] Native Swift bridges for WhisperKit/FluidAudio
- [ ] Implement Moonshine ASR (ONNX-based)
- [ ] Implement VibeVoice
- [ ] Add streaming benchmark mode
