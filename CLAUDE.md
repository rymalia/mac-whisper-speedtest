# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking tool that compares 9 different Whisper (speech-to-text) implementations on Apple Silicon Macs. Measures transcription performance while displaying actual transcription text for quality comparison.

## Commands

### Setup
```bash
uv sync                                                    # Install Python dependencies
cd tools/whisperkit-bridge && swift build -c release       # Build WhisperKit bridge (required)
cd tools/fluidaudio-bridge && swift build -c release       # Build FluidAudio bridge (optional)
```

### Run Benchmarks
```bash
.venv/bin/mac-whisper-speedtest                           # Default: small model, all implementations
.venv/bin/mac-whisper-speedtest --model large             # Specific model (tiny/small/base/medium/large)
.venv/bin/mac-whisper-speedtest --implementations "WhisperKitImplementation,MLXWhisperImplementation"
.venv/bin/mac-whisper-speedtest --num-runs 5              # Multiple runs for statistical accuracy
```

### Tests
```bash
pytest tests/ -v                                          # Run all tests
pytest tests/test_model_params.py -v                      # Model parameter validation
pytest tests/test_parakeet_integration.py -v              # Parakeet integration tests
pytest tests/test_model_params.py::TestModelParams::test_all_implementations_include_model_in_params -v  # Single test
```

## Architecture

### Data Flow
```
CLI (typer) → Audio Recording (PyAudio @ 16kHz) → Benchmark Runner (async)
    → Load each Implementation → Run transcription N times → Measure timing → Display results
```

### Key Components

- **`cli.py`**: Entry point, handles `--model`, `--implementations`, `--num-runs` args
- **`benchmark.py`**: Core benchmarking logic with `BenchmarkConfig` and `BenchmarkSummary` dataclasses
- **`audio.py`**: Records audio at 16kHz/16-bit mono, converts to float32 numpy arrays
- **`implementations/__init__.py`**: Dynamic registry with conditional imports based on available packages
- **`implementations/base.py`**: Abstract `WhisperImplementation` base class

### Implementation Pattern
All 9 implementations inherit from `WhisperImplementation` and must implement:
- `load_model(model_name)` - Load model, handle name mapping and fallbacks
- `async transcribe(audio)` - Return `TranscriptionResult` with text, segments, language
- `get_params()` - Return dict with `"model"` key and other config info

### Swift Bridges
`tools/whisperkit-bridge/` and `tools/fluidaudio-bridge/` are Swift CLI tools that communicate via JSON over subprocess. They measure internal transcription time (excluding audio loading overhead).

## Implementations

| Category | Implementations |
|----------|----------------|
| Native CoreML | WhisperKit, FluidAudio CoreML |
| MLX-Accelerated | mlx-whisper, Parakeet MLX, lightning-whisper-mlx |
| GPU/MPS | insanely-fast-whisper, whisper-mps |
| CPU | faster-whisper, whisper.cpp+CoreML |

## Requirements

- macOS 14.0+ with Apple Silicon
- Python 3.11+
- Swift 5.9+ / Xcode 15.0+ (for bridges)


## Non-Interactive Benchmarking (Recommended for Agents/CI)
```bash
python3 test_benchmark.py                    # Uses tests/jfk.wav, runs all implementations
```

This script bypasses microphone recording and uses pre-recorded audio files, enabling:
- **Automated testing** in CI/CD pipelines
- **Agent-based development** (Claude agents cannot access microphones or respond to prompts)
- **Reproducible debugging** (same audio input = deterministic comparison across runs)

Test audio files in `tests/`:
| File | Size | Description |
|------|------|-------------|
| `jfk.wav` | 352 KB | Classic JFK speech sample (Whisper demo standard) |
| `ted_60.wav` | 1.9 MB | 60-second TED talk audio |
| `ted_60_stereo_32.wav` | 15 MB | Stereo 32-bit version (tests audio preprocessing) |

**Why this matters for Claude agents:** The interactive CLI (`mac-whisper-speedtest`) requires pressing Enter to start/stop recording and access to microphone hardware—neither of which is available in agent environments. Use `test_benchmark.py` instead.
