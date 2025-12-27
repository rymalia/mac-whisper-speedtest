# Whisper Benchmark for Apple Silicon

A comprehensive benchmarking tool to compare different Whisper implementations optimized for Apple Silicon, focusing on speed while maintaining accuracy. Now supports **9 working implementations** including native Swift frameworks and MLX-based solutions, with **FluidAudio-CoreML as the fastest** at 0.08s (138x real-time factor).

## Example output

Test results from MacBook Pro M4 24GB:

```
=== Benchmark Summary for 'small' model ===

Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
fluidaudio-coreml      0.0800          model=parakeet-tdt-0.6b-v3-coreml, backend=FluidAudio Swift Bridge, platform=Apple Silicon
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

whisperkit             0.4341          model=small, backend=WhisperKit Swift Bridge, platform=Apple Silicon
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

mlx-whisper            0.7118          model=whisper-small-mlx-4bit, quantization=4bit
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

whisper.cpp            0.8576          model=small, coreml=True, n_threads=4
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

parakeet-mlx           0.9059          model=parakeet-tdt-0.6b-v2, implementation=parakeet-mlx, platform=Apple Silicon (MLX)
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your..."

insanely-fast-whisper  1.2431          model=whisper-small, device_id=mps, batch_size=16, compute_type=float16, quantization=4bit
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

lightning-whisper-mlx  1.4048          model=small, batch_size=12, quant=none
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

faster-whisper         2.1081          model=small, device=cpu, compute_type=int8, beam_size=1, cpu_threads=6
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."

whisper-mps            6.8990          model=small, backend=whisper-mps, device=mps, language=None
    "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c..."
```

Demo: https://x.com/anvanvan/status/1913624854584037443

## Overview

This tool measures transcription performance across different implementations of the same base model (e.g., all variants of "small"). It helps you find the fastest Whisper implementation on your Apple Silicon Mac for a given base model. With **9 working implementations** including native Swift frameworks, MLX-based solutions, and CPU-optimized variants, you can find the perfect balance of speed and accuracy for your use case.

## Features

- **Live speech recording** with automatic audio preprocessing
- **9 working Whisper implementations** with Apple Silicon optimizations
- **FluidAudio-CoreML is the fastest** at 0.08s (138x real-time factor)
- **Base model selection** (tiny, small, base, medium, large) with automatic fallback chains
- **Transcription quality comparison** - see actual transcription text alongside performance metrics
- **Apple Silicon-specific optimizations** with up to 16% performance improvements
- **Native Swift bridge support** for WhisperKit and FluidAudio frameworks
- **Comprehensive parameter reporting** showing actual models and configurations used

## Implementations Tested

### 🚀 Native Apple Silicon Implementations

1. **FluidAudio-CoreML** ⚡⚡⚡ _FASTEST - Apple Silicon Native_

   - Source: https://github.com/FluidInference/FluidAudio
   - **Technology**: Native Swift + CoreML, optimized for Apple Neural Engine with Parakeet model
   - **Performance**: **FASTEST implementation at 0.08s** (138x real-time factor), 81% faster than WhisperKit
   - **Bridge**: Custom Swift bridge for seamless Python integration
   - **Setup**: Requires one-time model fix: `./tools/fluidaudio-bridge/fix_models.sh`
   - **Documentation**: See `docs/MODEL_CACHING.md` for complete setup guide

2. **WhisperKit** ⚡⚡ _Apple Silicon Native_

   - Source: https://github.com/argmaxinc/WhisperKit
   - **Technology**: Native Swift + CoreML, optimized for Apple Neural Engine
   - **Performance**: 2nd fastest implementation, leverages on-device inference
   - **Bridge**: Custom Swift bridge for seamless Python integration

### 🔥 MLX-Accelerated Implementations

3. **mlx-whisper**

   - Source: https://github.com/ml-explore/mlx-examples
   - **Technology**: Apple MLX framework for unified memory optimization
   - **Models**: Quantized MLX models (4-bit, 8-bit) for memory efficiency

4. **Parakeet MLX** ⚡ _MLX Alternative_

   - Source: https://github.com/nvidia/parakeet
   - **Technology**: NVIDIA Parakeet models running on Apple MLX
   - **Performance**: Optimized for Apple Silicon with MLX acceleration
   - **Note**: Alternative Parakeet implementation (FluidAudio-CoreML uses Parakeet v3 with CoreML)

5. **lightning-whisper-mlx** ⚡ _Apple Silicon Optimized_
   - Source: https://github.com/lightning-AI/lightning-whisper
   - **Optimization**: 4-bit quantization enabled, optimized batch sizing
   - **Performance**: Significant memory efficiency improvements

### ⚡ GPU/MPS-Accelerated Implementations

6. **insanely-fast-whisper** ⚡ _Apple Silicon Optimized_

   - Source: https://github.com/Vaibhavs10/insanely-fast-whisper
   - **Optimization**: Adaptive batch sizing, SDPA attention, memory layout optimization
   - **Performance**: 16.0% faster on Apple Silicon with MPS acceleration

7. **whisper-mps**
   - Source: https://github.com/AtomGradient/whisper-mps
   - **Technology**: Direct MPS (Metal Performance Shaders) acceleration
   - **Performance**: Optimized for Apple Silicon GPU acceleration

### 🖥️ CPU-Optimized Implementations

8. **faster-whisper** ⚡ _Apple Silicon Optimized_

   - Source: https://github.com/SYSTRAN/faster-whisper
   - **Optimization**: Dynamic CPU thread allocation, Apple Accelerate framework
   - **Performance**: 2.0% faster with intelligent core detection (performance vs efficiency cores)

9. **whisper.cpp + CoreML**
   - Source: https://github.com/abdeladim-s/pywhispercpp
   - **Technology**: C++ implementation with optional CoreML acceleration
   - **Performance**: Excellent baseline performance with CoreML support

> **📚 For detailed architectural comparison and implementation differences**, see [docs/IMPLEMENTATION_ARCHITECTURE.md](docs/IMPLEMENTATION_ARCHITECTURE.md). This document explains why different implementations have different performance characteristics, integration patterns, and warning behaviors.

## Installation

```bash
# Clone the repository
git clone https://github.com/anvanvan/mac-whisper-speedtest.git
cd mac-whisper-speedtest

# Install dependencies (Python 3.11+ required)
uv sync

# Build Swift bridges
# WhisperKit bridge (no setup required)
cd tools/whisperkit-bridge && swift build -c release && cd ../..

# FluidAudio bridge (FASTEST - requires one-time model fix)
cd tools/fluidaudio-bridge && swift build -c release && cd ../..
# Fix models (copies from HuggingFace cache to Application Support)
./tools/fluidaudio-bridge/fix_models.sh
```

## Usage

```bash
# Run benchmark with default settings (small model, all implementations)
.venv/bin/mac-whisper-speedtest

# Run benchmark with a specific model
.venv/bin/mac-whisper-speedtest --model small

# Run benchmark with specific implementations
.venv/bin/mac-whisper-speedtest --model small --implementations "FluidAudioCoreMLImplementation,WhisperKitImplementation,MLXWhisperImplementation"

# Test fastest implementations (CoreML + Neural Engine)
.venv/bin/mac-whisper-speedtest --model small --implementations "FluidAudioCoreMLImplementation,WhisperKitImplementation"

# Test MLX-based implementations
.venv/bin/mac-whisper-speedtest --model small --implementations "MLXWhisperImplementation,ParakeetMLXImplementation,LightningWhisperMLXImplementation"

# Run benchmark with more runs for statistical accuracy
.venv/bin/mac-whisper-speedtest --model small --num-runs 5
```

## Features

### Universal Transcription Display 🎯

All implementations display their transcription results in the benchmark summary, allowing you to compare both **performance** and **transcription quality** across different implementations.

**Key features:**

- ✅ **Universal**: Works with all 9 working implementations including native Swift bridges
- ✅ **Smart formatting**: Long text is truncated, empty results show "(no transcription)"
- ✅ **Clean display**: Consistent indentation and formatting across all implementations
- ✅ **Model transparency**: Shows actual models used (including fallback substitutions)
- ✅ **Bridge timing**: WhisperKit and FluidAudio report internal transcription time (excluding bridge overhead)
- ✅ **Performance focus**: Transcription display doesn't interfere with timing comparisons

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4) - Required for optimal performance
- **macOS 14.0+** (for WhisperKit native support)
- **Xcode 15.0+** (for building WhisperKit Swift bridge)
- **Python 3.11+** (updated requirement for latest dependencies)
- **PyAudio and its dependencies** (for audio recording)
- **Swift Package Manager** (for building WhisperKit bridge)
- **Various Whisper implementations** (installed automatically via uv)

## Project Structure

```
mac-whisper-speedtest/
├── pyproject.toml                    # Updated dependencies (Python 3.11+)
├── docs/
│   ├── APPLE_SILICON_OPTIMIZATIONS.md  # Detailed optimization guide
│   ├── MODEL_CACHING.md                # Model caching guide (FluidAudio setup)
│   ├── FLUIDAUDIO_FINAL_STATUS.md      # FluidAudio investigation and solution
│   └── BENCHMARK_RESULTS.md            # Performance comparison results
├── src/
│   └── mac_whisper_speedtest/
│       ├── __init__.py
│       ├── audio.py                  # Audio recording/processing
│       ├── benchmark.py              # Enhanced benchmarking with transcription display
│       ├── implementations/          # Individual implementation wrappers
│       │   ├── __init__.py           # Implementation registry (9 working implementations)
│       │   ├── base.py               # Abstract base class
│       │   ├── coreml.py             # WhisperCpp with CoreML
│       │   ├── faster.py             # Faster Whisper (Apple Silicon optimized)
│       │   ├── insanely.py           # Insanely Fast Whisper (Apple Silicon optimized)
│       │   ├── mlx.py                # MLX Whisper
│       │   ├── lightning.py          # Lightning Whisper MLX (4-bit quantization)
│       │   ├── whisperkit.py         # WhisperKit (Swift bridge)
│       │   ├── fluidaudio_coreml.py  # FluidAudio CoreML (FASTEST - with setup)
│       │   ├── parakeet_mlx.py       # Parakeet MLX (alternative Parakeet implementation)
│       │   └── whisper_mps.py        # whisper-mps
│       └── cli.py                    # Command line interface
├── tools/
│   ├── whisperkit-bridge/           # Swift bridge for WhisperKit
│   │   ├── Package.swift
│   │   └── Sources/whisperkit-bridge/main.swift
│   └── fluidaudio-bridge/           # Swift bridge for FluidAudio (FASTEST)
│       ├── Package.swift
│       ├── Sources/fluidaudio-bridge/main.swift
│       └── fix_models.sh             # One-time model setup script
├── tests/
│   ├── test_model_params.py         # Model parameter validation tests
│   └── test_parakeet_integration.py # Parakeet MLX integration tests
└── README.md
```

### Version 2.0 (Latest)

- ✅ **9 working implementations**: FluidAudio-CoreML (FASTEST in most tests), WhisperKit, Parakeet MLX, and more
- ✅ **Native Swift bridges**: WhisperKit and FluidAudio integration with macOS frameworks
- ✅ **FluidAudio working**: Manual model fix enables the fastest implementation (138x real-time factor)
- ✅ **Enhanced Apple Silicon optimizations**: Up to 16% performance improvements
- ✅ **Transcription quality comparison**: See actual transcription text in results
- ✅ **Model fallback chains**: Automatic fallback for unavailable models (e.g., large → large-v3-turbo → large-v3)
- ✅ **Python 3.11+ support**: Updated dependencies and requirements
- ✅ **Comprehensive documentation**: Model caching guide, setup instructions, performance analysis

## License

MIT
