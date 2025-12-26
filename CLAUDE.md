# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mac-Whisper-Speedtest** is a benchmarking tool that compares 9 different Whisper speech-to-text implementations optimized for Apple Silicon Macs. The goal is to help users identify the fastest implementation for their specific use case while maintaining quality comparisons.

## Core Commands

### Development Setup
```bash
# Install dependencies (requires Python 3.11+)
uv sync

# Build Swift bridges (required for WhisperKit and FluidAudio implementations)
cd tools/whisperkit-bridge && swift build -c release && cd ../..
cd tools/fluidaudio-bridge && swift build -c release && cd ../..
```

### Running Benchmarks
```bash
# Run with default settings (small model, all implementations)
.venv/bin/mac-whisper-speedtest

# Run with specific model
.venv/bin/mac-whisper-speedtest --model large

# Run with specific implementations
.venv/bin/mac-whisper-speedtest --model small --implementations "WhisperKitImplementation,FluidAudioCoreMLImplementation"

# Run with more passes for statistical accuracy
.venv/bin/mac-whisper-speedtest --model small --num-runs 5
```

### Testing
```bash
# Run tests
.venv/bin/pytest

# Run specific test
.venv/bin/pytest tests/test_model_params.py
```

## Architecture

### Plugin-Based Implementation System

The codebase uses an **abstract base class (ABC) pattern** with dynamic implementation discovery:

1. **Base Interface** (`implementations/base.py`):
   - `WhisperImplementation` ABC defines the contract
   - Two required methods: `load_model()` and `async transcribe()`
   - Optional methods: `get_params()` for reporting configuration, `cleanup()` for resource management
   - Data models: `TranscriptionResult`, `BenchmarkResult`

2. **Dynamic Registry** (`implementations/__init__.py`):
   - Uses `importlib.util.find_spec()` to conditionally import implementations
   - Gracefully handles missing optional dependencies
   - Implementations only appear if their packages are installed
   - Platform checks for macOS-only implementations (WhisperKit, whisper-mps)

3. **Implementation Categories**:
   - **Native Swift Bridges** (2): WhisperKit, FluidAudio - use subprocess to call compiled Swift binaries
   - **MLX-Accelerated** (3): mlx-whisper, lightning-whisper-mlx, parakeet-mlx
   - **GPU/CPU-Optimized** (4): insanely-fast-whisper (MPS), whisper-mps, faster-whisper (CPU), whisper.cpp (CoreML)

### Swift Bridge Pattern

For native macOS frameworks (WhisperKit, FluidAudio):

1. Python calls compiled Swift executables via `subprocess.run()`
2. Audio is saved to temporary WAV file
3. Swift binary processes audio and returns JSON via stdout
4. JSON includes transcription text and internal timing
5. Internal timing (`_transcription_time` attribute) excludes bridge overhead for accurate benchmarking

**Key files**: `implementations/whisperkit.py`, `implementations/fluidaudio_coreml.py`

### Benchmark Flow

```
CLI Entry (cli.py)
    ↓
Record Audio (async, PyAudio)
    ↓
For Each Implementation:
    ├─ Load Model (not timed)
    ├─ Run N times (default 3):
    │   ├─ Transcribe audio
    │   ├─ Record time (or use internal time if available)
    │   └─ Capture transcription text
    ├─ Calculate average time
    └─ Collect params via get_params()
    ↓
Sort by Performance
    ↓
Display Results Table (implementation, time, params, transcription preview)
```

### Audio Processing

- **Sample Rate**: 16kHz mono (Whisper requirement)
- **Format**: Float32 normalized to [-1, 1]
- **Recording**: PyAudio with async/await pattern using `asyncio.Event` for stop signal
- **Preprocessing**: `audio.to_whisper_ndarray()` converts to proper format
- Audio validation includes silence detection and minimum length checks

### Model Fallback Chains

Implementations use fallback chains for graceful degradation:
```python
# Example for "large" model:
["large-v3-turbo", "large-v3", "large"]
```

This handles version mismatches across implementations automatically.

## Adding New Implementations

1. Create new file in `src/mac_whisper_speedtest/implementations/`
2. Inherit from `WhisperImplementation` ABC
3. Implement required methods:
   - `load_model(model_name: str)` - Load/initialize the model (not timed)
   - `async transcribe(audio: np.ndarray) -> TranscriptionResult` - Perform transcription
4. Implement optional methods:
   - `get_params() -> Dict[str, Any]` - Return implementation-specific parameters for transparency
   - `cleanup()` - Clean up resources
5. Add conditional import to `implementations/__init__.py`:
   ```python
   try:
       if importlib.util.find_spec("package_name"):
           from mac_whisper_speedtest.implementations.your_impl import YourImplementation
           available_implementations.append(YourImplementation)
   except ImportError:
       logger.warning("Failed to import YourImplementation")
   ```
6. Add display name mapping in `benchmark.py` `print_summary()` method

## Apple Silicon Optimizations

Key optimizations per implementation (documented in `docs/APPLE_SILICON_OPTIMIZATIONS.md`):

- **faster-whisper**: Dynamic CPU thread detection via `system_profiler`, targets performance + efficiency cores
- **insanely-fast-whisper**: Adaptive batch sizing for MPS, SDPA attention (16% improvement)
- **lightning-whisper-mlx**: 4-bit quantization enabled, batch size 12 optimized for unified memory
- **WhisperKit**: Native CoreML + Apple Neural Engine
- **FluidAudio**: Real-time streaming ASR (~110x RTF on M4 Pro)

## Important Implementation Notes

### Transcription Display
- All implementations display actual transcription text in benchmark results
- Text is truncated to 100 characters for readability
- Helps verify quality consistency across implementations
- Empty results show "(no transcription)"

### Internal Timing Support
- Swift bridges (WhisperKit, FluidAudio) report internal timing via `_transcription_time` attribute
- Benchmark runner checks for this attribute and uses it when available
- Excludes subprocess/bridge overhead for fair comparisons
- See `benchmark.py` lines 147-156

### Error Handling
- Implementation failures don't stop the benchmark
- Failed implementations show `transcription_time=inf` and `params={"error": "..."}`
- Allows partial results when some implementations are unavailable

## Project Structure

```
src/mac_whisper_speedtest/
├── __init__.py           # Package metadata
├── __main__.py           # Module entry point
├── cli.py                # Typer-based CLI, async audio recording
├── audio.py              # PyAudio recording, 16kHz preprocessing
├── benchmark.py          # Benchmark runner, result aggregation
├── utils.py              # Helper utilities (models dir, project root)
└── implementations/      # Implementation wrappers
    ├── __init__.py       # Dynamic registry with conditional imports
    ├── base.py           # ABC and data models
    └── *.py              # Individual implementations (9 total)

tools/
├── whisperkit-bridge/    # Swift Package Manager project
└── fluidaudio-bridge/    # Swift Package Manager project
```

## Dependencies

### Core Python Requirements
- Python 3.11+ (required)
- `typer` - CLI framework
- `pyaudio` - Audio recording
- `soundfile` - Audio I/O
- `numpy>=2.2.5` - Array processing
- `structlog` - Structured logging

### Whisper Implementations (all optional except faster-whisper and mlx-whisper)
- `faster-whisper>=1.1.1`
- `mlx>=0.5.0`, `mlx-whisper>=0.4.2`
- `insanely-fast-whisper>=0.0.15`
- `lightning-whisper-mlx>=0.0.10`
- `parakeet-mlx>=0.3.5`
- `whisper-mps>=0.0.7`
- `pywhispercpp` (custom fork from github.com/absadiki/pywhispercpp)
- `coremltools>=8.0.0`

### System Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- macOS 14.0+ (for WhisperKit and FluidAudio)
- Xcode 15.0+ (for building Swift bridges)

## Common Patterns

### Async Implementation
All transcription methods must be async:
```python
async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
    # Even if the underlying implementation is sync, just return the result
    result = self.model.transcribe(audio)
    return TranscriptionResult(text=result["text"])
```

### Parameter Reporting
Use `get_params()` to report what was actually used (not what was requested):
```python
def get_params(self) -> Dict[str, Any]:
    return {
        "model": self.actual_model_loaded,  # Not the requested model
        "device": "mps",
        "batch_size": 12,
        "quantization": "4bit"
    }
```

### Resource Cleanup
Implement `cleanup()` to release resources:
```python
def cleanup(self) -> None:
    if hasattr(self, 'model'):
        del self.model
    if hasattr(self, 'processor'):
        del self.processor
```

## Testing Notes

- Tests use `pytest-asyncio` for async functionality
- `test_model_params.py` validates parameter reporting
- `test_parakeet_integration.py` tests Parakeet MLX integration
- All implementations should be testable with missing optional dependencies
