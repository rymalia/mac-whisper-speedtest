# Whisper Benchmark for Apple Silicon

A comprehensive benchmarking tool to compare different Whisper implementations optimized for Apple Silicon, focusing on speed while maintaining accuracy. Now supports **9 different implementations** including native Swift frameworks and MLX-based solutions.

## Example output

Test results from MacBook Pro M4 24GB:

```
=== Benchmark Summary for 'large' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
fluidaudio-coreml      0.1935          model=parakeet-tdt-0.6b-v2-coreml, backend=FluidAudio Swift Bridge, platform=Apple Silicon
    "Which is the fastest transcription on my Mac?"

parakeet-mlx           0.4995          model=parakeet-tdt-0.6b-v2, implementation=parakeet-mlx, platform=Apple Silicon (MLX)
    "Which is the fastest transcription on my Mac?"

mlx-whisper            1.0230          model=whisper-large-v3-turbo, quantization=none
    "Which is the fastest transcription on my Mac?"

insanely-fast-whisper  1.1324          model=whisper-large-v3-turbo, device_id=mps, batch_size=12, compute_type=float16, quantization=4bit
    "Which is the fastest transcription on my Mac?"

whisper.cpp            1.2293          model=large-v3-turbo-q5_0, coreml=True, n_threads=4
    "Which is the fastest transcription on my Mac?"

lightning-whisper-mlx  1.8160          model=large, batch_size=12, quant=none
    "which is the fastest transcription on my Mac"

whisperkit             2.2190          model=large-v3, backend=WhisperKit Swift Bridge, platform=Apple Silicon
    "Which is the fastest transcription on my Mac?"

whisper-mps            5.3722          model=large, backend=whisper-mps, device=mps, language=None
    "Which is the fastest transcription on my Mac?"

faster-whisper         6.9613          model=large-v3-turbo, device=cpu, compute_type=int8, beam_size=1, cpu_threads=12, original_model_requested=large
    "Which is the fastest transcription on my Mac?"
```

Demo: https://x.com/anvanvan/status/1913624854584037443

## Overview

This tool measures transcription performance across different implementations of the same base model (e.g., all variants of "small"). It helps you find the fastest Whisper implementation on your Apple Silicon Mac for a given base model. With **9 different implementations** including native Swift frameworks, MLX-based solutions, and CPU-optimized variants, you can find the perfect balance of speed and accuracy for your use case.

## Features

- **Live speech recording** with automatic audio preprocessing
- **8 different Whisper implementations** with Apple Silicon optimizations
- **Base model selection** (tiny, small, base, medium, large) with automatic fallback chains
- **Transcription quality comparison** - see actual transcription text alongside performance metrics
- **Apple Silicon-specific optimizations** with up to 16% performance improvements
- **Native Swift bridge support** for WhisperKit and FluidAudio frameworks
- **Comprehensive parameter reporting** showing actual models and configurations used

## Implementations Tested

### 🚀 Native Apple Silicon Implementations

1. **WhisperKit** ⚡ _Apple Silicon Native_

   - Source: https://github.com/argmaxinc/WhisperKit
   - **Technology**: Native Swift + CoreML, optimized for Apple Neural Engine
   - **Performance**: Fastest implementation, leverages on-device inference
   - **Bridge**: Custom Swift bridge for seamless Python integration

2. **FluidAudio CoreML** ⚡ _Apple Silicon Native_
   - Source: https://github.com/FluidInference/FluidAudio
   - **Technology**: Native Swift + CoreML with Parakeet TDT models
   - **Performance**: Real-time streaming ASR with ~110x RTF on M4 Pro
   - **Bridge**: Custom Swift bridge with internal timing measurement

### 🔥 MLX-Accelerated Implementations

3. **mlx-whisper**

   - Source: https://github.com/ml-explore/mlx-examples
   - **Technology**: Apple MLX framework for unified memory optimization
   - **Models**: Quantized MLX models (4-bit, 8-bit) for memory efficiency

4. **Parakeet MLX** ⚡ _New Implementation_

   - Source: https://github.com/nvidia/parakeet
   - **Technology**: NVIDIA Parakeet models running on Apple MLX
   - **Performance**: Optimized for Apple Silicon with MLX acceleration

5. **lightning-whisper-mlx** ⚡ _Apple Silicon Optimized_
   - Source: https://github.com/lightning-AI/lightning-whisper
   - **Optimization**: 4-bit quantization enabled, optimized batch sizing
   - **Performance**: Significant memory efficiency improvements

### ⚡ GPU/MPS-Accelerated Implementations

6. **insanely-fast-whisper** ⚡ _Apple Silicon Optimized_

   - Source: https://github.com/Vaibhavs10/insanely-fast-whisper
   - **Optimization**: Adaptive batch sizing, SDPA attention, memory layout optimization
   - **Performance**: 16.0% faster on Apple Silicon with MPS acceleration

7. **whisper-mps** ⚡ _New Implementation_
   - Source: https://github.com/AtomGradient/whisper-mps
   - **Technology**: Direct MPS (Metal Performance Shaders) acceleration
   - **Performance**: Optimized for Apple Silicon GPU acceleration

### 🖥️ CPU-Optimized Implementations

8. **faster-whisper** ⚡ _Apple Silicon Optimized_

   - Source: https://github.com/SYSTRAN/faster-whisper
   - **Optimization**: Dynamic CPU thread allocation, Apple Accelerate framework
   - **Performance**: 2.0% faster with intelligent core detection (performance vs efficiency cores)

9. **whisper.cpp + CoreML**
   - Source: https://github.com/abdeladim-s/pywhispercpp (Python bindings for whisper.cpp)
   - Source: https://github.com/ggml-org/whisper.cpp
   - **Technology**: C++ implementation with optional CoreML acceleration
   - **Performance**: Excellent baseline performance with CoreML support

## Installation

```bash
# Clone the repository
git clone https://github.com/anvanvan/mac-whisper-speedtest.git
cd mac-whisper-speedtest

# Install dependencies (Python 3.11+ required)
uv sync

# Build Swift bridges (required for native implementations)
# WhisperKit bridge
cd tools/whisperkit-bridge && swift build -c release && cd ../..

# FluidAudio bridge (optional - only if you want FluidAudio support)
cd tools/fluidaudio-bridge && swift build -c release && cd ../..
```

## Usage

### Interactive Mode (Microphone Recording)

```bash
# Run benchmark with default settings (small model, all implementations)
.venv/bin/mac-whisper-speedtest

# Run benchmark with a specific model
.venv/bin/mac-whisper-speedtest -m large

# Run benchmark with specific implementations
.venv/bin/mac-whisper-speedtest -m small -i "WhisperKitImplementation,FluidAudioCoreMLImplementation"

# Run benchmark with more runs for statistical accuracy
.venv/bin/mac-whisper-speedtest -m small -n 5
```

### Non-Interactive Mode (Batch)

For CI/CD pipelines, remote development, or reproducible testing:

```bash
# Use default test audio (tests/jfk.wav)
.venv/bin/mac-whisper-speedtest --batch

# Use custom audio file
.venv/bin/mac-whisper-speedtest --batch --audio tests/ted_60.wav

# Batch mode with all options
.venv/bin/mac-whisper-speedtest -b -m large -n 1 -i "MLXWhisperImplementation"
```

This bypasses microphone recording and uses pre-recorded audio files.

### CLI Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Model size: tiny/base/small/medium/large | small |
| `--runs` | `-n` | Number of runs per implementation | 3 |
| `--implementations` | `-i` | Comma-separated implementation names | all |
| `--batch` | `-b` | Non-interactive mode (use audio file) | off |
| `--audio` | `-a` | Audio file path for batch mode | tests/jfk.wav |

## Features

### Universal Transcription Display 🎯

All implementations display their transcription results in the benchmark summary, allowing you to compare both **performance** and **transcription quality** across different implementations.

**Key features:**

- ✅ **Universal**: Works with all 9 supported implementations including native Swift bridges
- ✅ **Smart formatting**: Long text is truncated, empty results show "(no transcription)"
- ✅ **Clean display**: Consistent indentation and formatting across all implementations
- ✅ **Model transparency**: Shows actual models used (including fallback substitutions)
- ✅ **Bridge timing**: Native implementations report internal transcription time (excluding bridge overhead)
- ✅ **Performance focus**: Transcription display doesn't interfere with timing comparisons

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4) - Required for optimal performance
- **macOS 14.0+** (for WhisperKit and FluidAudio native support)
- **Xcode 15.0+** (for building Swift bridges)
- **Python 3.11+** (updated requirement for latest dependencies)
- **PyAudio and its dependencies** (for audio recording)
- **Swift Package Manager** (for building native bridges)
- **Various Whisper implementations** (installed automatically via uv)

## Project Structure

```
mac-whisper-speedtest/
├── pyproject.toml                    # Updated dependencies (Python 3.11+)
├── docs/
│   └── APPLE_SILICON_OPTIMIZATIONS.md  # Detailed optimization guide
├── src/
│   └── mac_whisper_speedtest/
│       ├── __init__.py
│       ├── audio.py                  # Audio recording/processing
│       ├── benchmark.py              # Enhanced benchmarking with transcription display
│       ├── implementations/          # Individual implementation wrappers
│       │   ├── __init__.py           # Implementation registry (9 implementations)
│       │   ├── base.py               # Abstract base class
│       │   ├── coreml.py             # WhisperCpp with CoreML
│       │   ├── faster.py             # Faster Whisper (Apple Silicon optimized)
│       │   ├── insanely.py           # Insanely Fast Whisper (Apple Silicon optimized)
│       │   ├── mlx.py                # MLX Whisper
│       │   ├── lightning.py          # Lightning Whisper MLX (4-bit quantization)
│       │   ├── whisperkit.py         # WhisperKit (Swift bridge)
│       │   ├── fluidaudio_coreml.py  # FluidAudio CoreML (Swift bridge)
│       │   ├── parakeet_mlx.py       # Parakeet MLX (new implementation)
│       │   └── whisper_mps.py        # whisper-mps (new implementation)
│       └── cli.py                    # Command line interface
├── tools/
│   ├── whisperkit-bridge/           # Swift bridge for WhisperKit
│   │   ├── Package.swift
│   │   └── Sources/whisperkit-bridge/main.swift
│   └── fluidaudio-bridge/           # Swift bridge for FluidAudio
│       ├── Package.swift
│       └── Sources/fluidaudio-bridge/main.swift
├── tests/
│   ├── jfk.wav                       # Test audio (JFK speech sample)
│   ├── ted_60.wav                    # Test audio (60s TED talk)
│   ├── ted_60_stereo_32.wav          # Test audio (stereo 32-bit version)
│   ├── test_model_params.py          # Model parameter validation tests
│   └── test_parakeet_integration.py  # Parakeet MLX integration tests
└── README.md
```

### Version 2.0 (Latest)

- ✅ **4 new implementations**: WhisperKit, FluidAudio CoreML, Parakeet MLX, whisper-mps
- ✅ **Native Swift bridges**: Direct integration with macOS frameworks
- ✅ **Enhanced Apple Silicon optimizations**: Up to 16% performance improvements
- ✅ **Transcription quality comparison**: See actual transcription text in results
- ✅ **Model fallback chains**: Automatic fallback for unavailable models (e.g., large → large-v3-turbo → large-v3)
- ✅ **Python 3.11+ support**: Updated dependencies and requirements

## License

MIT
