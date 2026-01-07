# Codebase Exploration: 9 Whisper Implementations

This document provides a comprehensive analysis of the mac-whisper-speedtest codebase, focusing on the 9 different Whisper implementations, their patterns, similarities, differences, and recommendations for maintainability and adding future implementations.

## Executive Summary

This codebase implements a **plugin-based benchmarking architecture** for 9 different Whisper implementations on Apple Silicon. The design uses:
- Abstract base class (`WhisperImplementation`) defining a clean contract
- Dynamic registry pattern with conditional imports for optional dependencies
- Async transcription with standardized result types
- Three distinct implementation patterns (pure Python, file-based, Swift bridge)

---

## Architecture Overview

### Design Patterns

**The Strategy Pattern:** Each implementation is a pluggable "strategy" for transcription. The base class (`implementations/base.py:28-47`) defines just 4 methods: `load_model()`, `transcribe()`, `get_params()`, and `cleanup()`. This minimalist interface makes adding new implementations straightforward.

**Registry Pattern:** The `__init__.py` uses `importlib.util.find_spec()` to check for optional dependencies before importing, allowing graceful degradation when packages are missing.

### Core Data Structures (`base.py:10-26`)

```python
@dataclass
class TranscriptionResult:
    text: str                      # Main transcription
    segments: List = []            # Timestamped segments (optional)
    language: Optional[str] = None # Detected language

@dataclass
class BenchmarkResult:
    implementation: str            # Class name
    model_name: str               # User-requested model
    model_params: Dict[str, Any]  # Actual params used (MUST include "model")
    transcription_time: float     # Average execution time
    text: str = ""                # For display
```

---

## Implementation Categories

### 1. Pure Python (Direct NumPy Array)

| Implementation | File | Audio Input | Key Features |
|----------------|------|-------------|--------------|
| **mlx-whisper** | `mlx.py` | numpy array | Quantization detection, fallback chains |
| **faster-whisper** | `faster.py` | numpy array | CPU P/E core optimization |
| **whisper-mps** | `whisper_mps.py` | numpy array | Native MPS acceleration |
| **whisper.cpp** | `coreml.py` | numpy array | CoreML environment variable |

### 2. File-Based Python (Temp WAV Required)

| Implementation | File | Audio Input | Key Features |
|----------------|------|-------------|--------------|
| **lightning-whisper-mlx** | `lightning.py` | temp WAV file | Simple model mapping |
| **parakeet-mlx** | `parakeet_mlx.py` | temp WAV file | Language detection heuristics |
| **insanely-fast-whisper** | `insanely.py` | temp WAV file | Dynamic batch sizing, 4-bit quantization |

### 3. Swift Bridge (Subprocess + JSON)

| Implementation | File | Audio Input | Key Features |
|----------------|------|-------------|--------------|
| **WhisperKit** | `whisperkit.py` | temp WAV → subprocess | Internal timing, extensive preprocessing |
| **FluidAudio** | `fluidaudio_coreml.py` | temp WAV → subprocess | Fixed model, dual timing metrics |

---

## Pattern Analysis: Similarities

### 1. Model Name Mapping Pattern

**Every implementation** has a model mapping dictionary:

```python
# Pattern found in ALL implementations
model_map = {
    "tiny": "<implementation-specific-tiny>",
    "small": "<implementation-specific-small>",
    "large": "<implementation-specific-large>",
}
self.model_name = model_map.get(model_name, model_name)
```

| Implementation | "large" maps to |
|----------------|-----------------|
| mlx-whisper | `mlx-community/whisper-large-v3-turbo` |
| faster-whisper | `large-v3-turbo` (via fallback chain) |
| lightning-whisper-mlx | `large-v3` |
| insanely-fast-whisper | `openai/whisper-large-v3-turbo` |
| whisper.cpp | `large-v3-turbo-q5_0` |
| WhisperKit | `large-v3` |
| parakeet-mlx | `mlx-community/parakeet-tdt-0.6b-v2` (all sizes) |

### 2. Platform Check Pattern

**All macOS-specific implementations** check platform at `__init__`:

```python
# Found in: mlx.py, lightning.py, parakeet_mlx.py, whisperkit.py,
#           fluidaudio_coreml.py, whisper_mps.py
if platform.system() != "Darwin":
    raise RuntimeError("Only supported on macOS with Apple Silicon")
```

### 3. Audio Preprocessing Pattern

**Swift bridges share identical preprocessing** (`whisperkit.py:170-209`, `fluidaudio_coreml.py:153-192`):

```python
def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
    # 1. Convert to float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 2. Convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 3. Normalize to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    # 4. Pad short audio (< 0.1s)
    if len(audio) < 1600:
        audio = np.pad(audio, (0, 1600 - len(audio)))
```

This is **duplicated code** that could be extracted.

### 4. Temp File Pattern

**File-based implementations** use identical temp file handling:

```python
# Found in 5 implementations
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
    sf.write(temp_file.name, audio, 16000)
    try:
        result = self._model.transcribe(temp_file.name)
    finally:
        os.unlink(temp_file.name)
```

### 5. Result Extraction Pattern

**Most implementations** extract results identically:

```python
text = result.get("text", "")
segments = result.get("segments", [])
language = result.get("language")
return TranscriptionResult(text=text, segments=segments, language=language)
```

---

## Pattern Analysis: Differences

### 1. Model Loading Strategies

| Strategy | Implementations | Characteristics |
|----------|-----------------|-----------------|
| **Single attempt** | lightning, whisper_mps, coreml | Simple, fails fast |
| **Fallback chain** | mlx, faster | Tries multiple models sequentially |
| **Quantization preference** | mlx, insanely | Prefers quantized, falls back to full |
| **Fixed model** | parakeet, fluidaudio | Ignores model_name, uses single model |

**faster-whisper's fallback chain** (`faster.py:40-56`) is elegant:
```python
def _get_model_fallback_chain(self, model_name: str) -> List[str]:
    if model_name == "large":
        return ["large-v3-turbo", "large-v3", "large"]
    return [model_name]
```
This pattern ensures users always get the best available model variant.

### 2. Timing Measurement

| Approach | Implementations | Accuracy |
|----------|-----------------|----------|
| **Python timing** | mlx, faster, lightning, parakeet, insanely, whisper_mps, coreml | Includes Python overhead |
| **Internal timing** | WhisperKit, FluidAudio | Excludes subprocess/bridge overhead |

The Swift bridges attach timing via a private attribute:
```python
result_obj._transcription_time = output["transcription_time"]
```

And `benchmark.py:149-154` uses it:
```python
if hasattr(result, '_transcription_time'):
    run_time = result._transcription_time  # Internal timing
else:
    run_time = end_time - start_time       # Python timing
```

### 3. CPU Thread Optimization

Only **faster-whisper** (`faster.py:58-121`) does sophisticated CPU detection:

```python
def _get_optimal_cpu_threads(self) -> int:
    # Parse system_profiler for P-core/E-core info
    result = subprocess.run(["system_profiler", "SPHardwareDataType"], ...)
    # Extract: "Total Number of Cores: 14 (10 performance and 4 efficiency)"
    # Use: perf_cores + 2 efficiency cores
```

### 4. Memory-Based Batch Sizing

Only **insanely-fast-whisper** (`insanely.py:30-70`) adapts batch size:

```python
available_memory_gb = psutil.virtual_memory().available / (1024**3)
if available_memory_gb >= 32:
    batch_size = 16  # High-end Apple Silicon
elif available_memory_gb >= 16:
    batch_size = 12  # Mid-range
# ...
```

### 5. Language Handling

| Approach | Implementations |
|----------|-----------------|
| **Automatic detection** | mlx, faster, lightning, insanely, whisper_mps |
| **Forced English** | parakeet (with heuristic warning) |
| **Fixed "en"** | fluidaudio |
| **From response** | WhisperKit |

Parakeet's language detection heuristic (`parakeet_mlx.py:183-209`) is unique:
```python
def _contains_non_english_patterns(self, text: str) -> bool:
    german_indicators = ['und', 'der', 'die', 'das', 'ß', 'ä', 'ö', 'ü']
    # ... returns True if likely non-English
```

---

## Registry Pattern Deep Dive

The `__init__.py` registry (`implementations/__init__.py:1-130`) demonstrates progressive enhancement:

```python
# Always available (no platform checks)
available_implementations = [
    FasterWhisperImplementation,
    MLXWhisperImplementation,
]

# Conditional: package availability
if importlib.util.find_spec("pywhispercpp"):
    from .coreml import WhisperCppCoreMLImplementation
    available_implementations.append(...)

# Conditional: platform + package
if platform.system() == "Darwin":
    from .whisperkit import WhisperKitImplementation
    available_implementations.append(...)
```

**Error Resilience:** Each import is wrapped in `try/except`, catching both `ImportError` and `RuntimeError`. This means a broken implementation won't crash the entire benchmark—it's simply skipped with a warning.

---

## Benchmark Runner Integration

The `benchmark.py:110-190` orchestrates everything:

1. **Model loading is NOT timed** (line 134)
2. **Multiple runs for averaging** (lines 139-156)
3. **Internal timing detection** (lines 149-154)
4. **Error isolation per implementation** (lines 179-188)
5. **Results sorted by speed** (line 38-41 in `print_summary()`)

---

## Non-Interactive Benchmarking: `test_benchmark.py`

### The Problem with Interactive Mode

The standard CLI (`cli.py`) requires:
- Pressing Enter to start recording
- Speaking into the microphone
- Pressing Enter to stop recording

This is **incompatible with**:
- CI/CD pipelines
- Claude agents (no microphone access, no stdin interaction)
- Reproducible debugging (different speech = different results)

### The Solution: `test_benchmark.py`

```python
# Instead of interactive recording:
#   audio_data = await record_audio(stop_event, ...)

# It loads pre-recorded audio:
audio_data, sample_rate = sf.read("tests/jfk.wav", dtype='float32')
if sample_rate != 16000:
    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
```

### Comparison: Interactive vs Non-Interactive

| Aspect | `cli.py` (Interactive) | `test_benchmark.py` (Non-Interactive) |
|--------|------------------------|---------------------------------------|
| **Audio Source** | Live microphone via PyAudio | Pre-recorded WAV file |
| **User Interaction** | Press Enter twice (start/stop) | None required |
| **Reproducibility** | Variable (depends on speaker) | Deterministic (same audio every run) |
| **CI/CD Compatible** | No (requires human input) | Yes |
| **Agent Compatible** | No (can't access mic) | Yes |
| **Debugging** | Difficult (can't replay exact audio) | Easy (same audio = same results) |

### Available Test Audio Files

| File | Size | Description |
|------|------|-------------|
| `tests/jfk.wav` | 352 KB | Classic JFK speech sample (Whisper demo standard) |
| `tests/ted_60.wav` | 1.9 MB | 60-second TED talk audio |
| `tests/ted_60_stereo_32.wav` | 15 MB | Stereo 32-bit version (tests audio preprocessing) |

### Usage

```bash
# Run full benchmark with pre-recorded audio (no interaction needed)
python3 test_benchmark.py
```

---

## Recommendations for Maintainability & Future Implementations

### 1. Extract Shared Utilities

**Audio Preprocessing** - Create `utils/audio_preprocessing.py`:
```python
def preprocess_for_whisper(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Standardize audio for Whisper: float32, mono, normalized, minimum length."""
    # Currently duplicated in whisperkit.py:170-209 and fluidaudio_coreml.py:153-192
```

**Temp File Helper** - Create a context manager:
```python
@contextmanager
def audio_temp_file(audio: np.ndarray, sample_rate: int = 16000):
    """Writes audio to temp WAV, yields path, cleans up."""
    # Currently duplicated in 5 implementations
```

### 2. Standardize Model Mapping

Create a central configuration:
```python
# models/config.py
MODEL_VERSIONS = {
    "large": {
        "default": "large-v3-turbo",
        "fallbacks": ["large-v3", "large-v2", "large"],
    },
    # ...
}
```

Each implementation would then reference this, ensuring consistency when new model versions are released.

### 3. Create an Implementation Template

For adding new implementations, document the minimal required structure:

```python
class NewWhisperImplementation(WhisperImplementation):
    """Template for new Whisper implementations."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        # Platform check if needed
        if platform.system() != "Darwin":
            raise RuntimeError("Only supported on macOS")

    def load_model(self, model_name: str) -> None:
        # 1. Map model name
        # 2. Download/load model
        # 3. Store in self._model
        pass

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        # 1. Preprocess audio if needed
        # 2. Run transcription
        # 3. Return TranscriptionResult
        pass

    def get_params(self) -> Dict[str, Any]:
        # MUST include "model" key
        return {"model": self.model_name}
```

### 4. Add Implementation Metadata

Enhance the base class to include discoverable capabilities:

```python
class WhisperImplementation(ABC):
    # Add class-level metadata
    requires_file_input: bool = False      # vs numpy array
    supports_gpu: bool = False             # MPS/CUDA support
    supports_streaming: bool = False       # Real-time streaming
    supports_language_detection: bool = True
    min_audio_length_samples: int = 1600   # 0.1s @ 16kHz
```

### 5. Add Automated Testing Hooks

Currently `tests/test_model_params.py` validates `get_params()` includes `"model"`. Expand with:

- **Contract tests**: Verify all implementations follow the interface
- **Audio format tests**: Verify each handles edge cases (silence, very short, stereo)
- **Error handling tests**: Verify graceful failures for missing models

### 6. Consider a Plugin System

For truly extensible architecture, implement auto-discovery:
```python
# implementations/__init__.py
def discover_implementations():
    """Auto-discover implementations from entry points."""
    for entry_point in pkg_resources.iter_entry_points('whisper_implementations'):
        yield entry_point.load()
```

This would allow third-party implementations without modifying the registry.

---

## Checklist for Adding a New Implementation

1. **Create the implementation file** in `implementations/`
2. **Inherit from `WhisperImplementation`**
3. **Implement required methods**: `load_model()`, `transcribe()`, `get_params()`
4. **Add to registry** in `implementations/__init__.py` with appropriate conditional import
5. **Add name mapping** in `benchmark.py:44-54` for short display name
6. **Add tests** validating the `get_params()` contract

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `implementations/base.py` | Abstract base class, data structures |
| `implementations/__init__.py` | Dynamic registry with conditional imports |
| `benchmark.py` | Benchmark orchestration, timing, results |
| `cli.py` | Interactive command-line interface |
| `test_benchmark.py` | Non-interactive benchmarking for agents/CI |
| `audio.py` | Audio recording and format conversion |
| `utils.py` | Shared utilities (models directory, project root) |

---

## Summary

This codebase has a well-designed plugin architecture that successfully abstracts 9 very different Whisper implementations (from pure Python MLX to Swift subprocess bridges) behind a unified interface.

**Key Strengths:**
- Clean abstract base class with minimal required methods
- Graceful degradation via conditional imports
- Error isolation prevents one failing implementation from breaking others
- Internal timing support for accurate Swift bridge benchmarks
- Non-interactive mode enables agent-based development and CI/CD

**Key Opportunities:**
- Extract duplicated audio preprocessing (~60 lines copied across 2 files)
- Extract temp file handling (~15 lines copied across 5 files)
- Centralize model version mappings for easier updates
- Add implementation capability metadata for discoverability

**For Future Implementations:**
- Follow the existing patterns (model mapping, platform checks, structured logging)
- Register in `__init__.py` with appropriate conditional guards
- Add to the `name_map` in `benchmark.py` for clean output
- Ensure `get_params()` returns at minimum `{"model": actual_model_used}`
- Test using `test_benchmark.py` for reproducible, agent-friendly debugging
