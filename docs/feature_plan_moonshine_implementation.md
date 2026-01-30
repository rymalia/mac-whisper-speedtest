# Feature Plan: Add Moonshine Implementation

**Date:** 2026-01-28
**Status:** Planning
**Related:** [Moonshine Analysis](research_moonshine_analysis.md)

---

## Summary

Add `MoonshineOnnxImplementation` as the 10th ASR implementation, using the `useful-moonshine-onnx` package. Moonshine is a family of speech-to-text models that process audio **5-15x faster** than Whisper on short segments.

**What's different about Moonshine:**
- Uses ONNX Runtime (cross-platform, lightweight)
- Scales compute with input length (short audio = fast)
- Outputs text only (no timestamps/segments)
- 9 model variants (2 English, 7 language-specific)

---

## Model Variants

| Model | Language | Params | Token Rate | Notes |
|-------|----------|--------|------------|-------|
| `moonshine/tiny` | English | 27M | 6/sec | Fast, lightweight |
| `moonshine/base` | English | 62M | 6/sec | Higher accuracy |
| `moonshine/tiny-ar` | Arabic | 27M | 13/sec | |
| `moonshine/tiny-zh` | Chinese | 27M | 13/sec | |
| `moonshine/tiny-ja` | Japanese | 27M | 13/sec | |
| `moonshine/tiny-ko` | Korean | 27M | 13/sec | |
| `moonshine/tiny-uk` | Ukrainian | 27M | 8/sec | |
| `moonshine/tiny-vi` | Vietnamese | 27M | 13/sec | |
| `moonshine/base-es` | Spanish | 62M | 6/sec | |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/mac_whisper_speedtest/implementations/moonshine_onnx.py` | CREATE | New Moonshine ONNX implementation |
| `src/mac_whisper_speedtest/implementations/__init__.py` | MODIFY | Register MoonshineOnnxImplementation |
| `pyproject.toml` | MODIFY | Add `useful-moonshine-onnx` dependency |
| `tests/test_moonshine_integration.py` | CREATE | Integration tests |

---

## Implementation Details

### 1. Create `moonshine_onnx.py`

```python
"""Moonshine ASR implementation using ONNX Runtime."""

from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation


class MoonshineOnnxImplementation(WhisperImplementation):
    """Moonshine ASR implementation using ONNX Runtime.

    Moonshine is a family of speech-to-text models optimized for fast,
    on-device ASR. It processes audio 5-15x faster than Whisper on
    short segments by scaling compute with input length.
    """

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self._model = None
        self._tokenizer = None
        self._moonshine_model_name = None

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name."""
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

        self.model_name = model_name
        self.log.info(f"Loading Moonshine ONNX model: {self.model_name}")

        # Map our model names to Moonshine's naming convention
        model_map = {
            # English variants
            "moonshine-tiny": "moonshine/tiny",
            "moonshine-base": "moonshine/base",
            "tiny": "moonshine/tiny",
            "base": "moonshine/base",
            # Non-English variants
            "moonshine-tiny-ar": "moonshine/tiny-ar",
            "moonshine-tiny-zh": "moonshine/tiny-zh",
            "moonshine-tiny-ja": "moonshine/tiny-ja",
            "moonshine-tiny-ko": "moonshine/tiny-ko",
            "moonshine-tiny-uk": "moonshine/tiny-uk",
            "moonshine-tiny-vi": "moonshine/tiny-vi",
            "moonshine-base-es": "moonshine/base-es",
        }

        # Default to tiny if unknown
        self._moonshine_model_name = model_map.get(model_name, "moonshine/tiny")
        self.log.info(f"Mapped to Moonshine model: {self._moonshine_model_name}")

        self._model = MoonshineOnnxModel(model_name=self._moonshine_model_name)
        self._tokenizer = load_tokenizer()

        # Warmup with 1 second of silence (ONNX first inference is slow)
        self.log.info("Warming up ONNX model...")
        _ = self._model.generate(np.zeros((1, 16000), dtype=np.float32))
        self.log.info("Successfully loaded Moonshine model")

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info("Transcribing with Moonshine ONNX")

        # Ensure correct shape: [1, num_samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Check duration constraints (0.1s to 64s)
        duration_secs = audio.shape[-1] / 16000
        if duration_secs < 0.1:
            self.log.warning(f"Audio too short ({duration_secs:.2f}s), padding to 0.1s")
            min_samples = int(0.1 * 16000)
            audio = np.pad(audio, ((0, 0), (0, min_samples - audio.shape[-1])))
        elif duration_secs > 64:
            self.log.warning(f"Audio too long ({duration_secs:.2f}s), truncating to 64s")
            audio = audio[:, :int(64 * 16000)]

        # Generate tokens
        tokens = self._model.generate(audio)

        # Decode to text
        text = self._tokenizer.decode_batch(tokens)[0]

        if text:
            self.log.info(f"Transcription completed: '{text[:50]}...'")
        else:
            self.log.warning("Moonshine returned empty transcription")

        # Moonshine doesn't provide segments/timestamps
        return TranscriptionResult(
            text=text,
            segments=[],  # Moonshine doesn't output segments
            language=self._get_language_from_model(),
        )

    def _get_language_from_model(self) -> str:
        """Get language from model name."""
        language_map = {
            "moonshine/tiny": "en",
            "moonshine/base": "en",
            "moonshine/tiny-ar": "ar",
            "moonshine/tiny-zh": "zh",
            "moonshine/tiny-ja": "ja",
            "moonshine/tiny-ko": "ko",
            "moonshine/tiny-uk": "uk",
            "moonshine/tiny-vi": "vi",
            "moonshine/base-es": "es",
        }
        return language_map.get(self._moonshine_model_name, "en")

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self._moonshine_model_name,
            "implementation": "moonshine-onnx",
            "runtime": "ONNX Runtime",
            "language": self._get_language_from_model(),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self._model = None
        self._tokenizer = None
        self._moonshine_model_name = None
```

### 2. Model Name Mapping Strategy

For consistency with our existing pattern, we'll accept multiple naming formats:

| User Input | Maps To |
|------------|---------|
| `"tiny"` | `"moonshine/tiny"` |
| `"base"` | `"moonshine/base"` |
| `"moonshine-tiny"` | `"moonshine/tiny"` |
| `"moonshine-base"` | `"moonshine/base"` |
| `"moonshine-tiny-ja"` | `"moonshine/tiny-ja"` |
| Unknown | `"moonshine/tiny"` (default) |

### 3. Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "useful-moonshine-onnx>=0.1.0",  # Moonshine ONNX implementation
]
```

**Transitive dependencies brought in by moonshine-onnx:**
- `onnxruntime>=1.16.0` — ONNX inference engine
- `huggingface_hub` — Model download (we already have this)
- `librosa` — Audio loading (we already have this)
- `tokenizers` — Text decoding (lightweight)
- `numba` — JIT compilation for librosa (may already be present)

### 4. Registration in `__init__.py`

Add after existing implementations:

```python
# Try to import MoonshineOnnxImplementation
try:
    if importlib.util.find_spec("moonshine_onnx"):
        from mac_whisper_speedtest.implementations.moonshine_onnx import MoonshineOnnxImplementation
        available_implementations.append(MoonshineOnnxImplementation)
        logger.info("MoonshineOnnxImplementation loaded successfully")
    else:
        logger.warning("moonshine-onnx not found, MoonshineOnnxImplementation will not be available")
except ImportError as e:
    logger.warning(f"Failed to import MoonshineOnnxImplementation: {e}")
```

---

## Audio Handling

### Sample Rate

Moonshine requires **exactly 16kHz**. Our benchmark already uses 16kHz, so no conversion needed.

### Duration Constraints

| Constraint | Value | Handling |
|------------|-------|----------|
| Minimum | 0.1 seconds | Pad with zeros if shorter |
| Maximum | 64 seconds | Truncate if longer (with warning) |

```python
# Check duration constraints
duration_secs = audio.shape[-1] / 16000
if duration_secs < 0.1:
    audio = np.pad(audio, ((0, 0), (0, min_samples - audio.shape[-1])))
elif duration_secs > 64:
    audio = audio[:, :int(64 * 16000)]
```

---

## Output Handling

### TranscriptionResult Mapping

Moonshine returns only text (no segments/timestamps):

```python
# Moonshine output
tokens = model.generate(audio)  # List of token IDs
text = tokenizer.decode_batch(tokens)[0]  # Single string

# Our TranscriptionResult
return TranscriptionResult(
    text=text,
    segments=[],  # Empty - Moonshine doesn't provide segments
    language="en",  # Inferred from model variant
)
```

---

## Verification Plan

### 1. Unit Test

Create `tests/test_moonshine_integration.py`:

```python
import pytest
import numpy as np
from pathlib import Path

# Skip if moonshine-onnx not installed
pytest.importorskip("moonshine_onnx")


class TestMoonshineOnnxImplementation:
    """Tests for MoonshineOnnxImplementation."""

    def test_load_model_tiny(self):
        """Test loading the tiny model."""
        from mac_whisper_speedtest.implementations.moonshine_onnx import MoonshineOnnxImplementation

        impl = MoonshineOnnxImplementation()
        impl.load_model("moonshine-tiny")
        assert impl._model is not None
        assert impl._moonshine_model_name == "moonshine/tiny"

    def test_load_model_base(self):
        """Test loading the base model."""
        from mac_whisper_speedtest.implementations.moonshine_onnx import MoonshineOnnxImplementation

        impl = MoonshineOnnxImplementation()
        impl.load_model("moonshine-base")
        assert impl._model is not None
        assert impl._moonshine_model_name == "moonshine/base"

    @pytest.mark.asyncio
    async def test_transcribe_jfk(self):
        """Test transcription with JFK audio."""
        from mac_whisper_speedtest.implementations.moonshine_onnx import MoonshineOnnxImplementation
        import librosa

        impl = MoonshineOnnxImplementation()
        impl.load_model("moonshine-tiny")

        # Load test audio
        audio_path = Path(__file__).parent / "jfk.wav"
        audio, _ = librosa.load(audio_path, sr=16000)

        result = await impl.transcribe(audio)

        assert result.text is not None
        assert len(result.text) > 0
        assert "ask not" in result.text.lower() or "country" in result.text.lower()

    def test_get_params(self):
        """Test get_params returns expected structure."""
        from mac_whisper_speedtest.implementations.moonshine_onnx import MoonshineOnnxImplementation

        impl = MoonshineOnnxImplementation()
        impl.load_model("moonshine-tiny")

        params = impl.get_params()
        assert "model" in params
        assert params["model"] == "moonshine/tiny"
        assert params["implementation"] == "moonshine-onnx"
        assert params["runtime"] == "ONNX Runtime"
```

### 2. Batch Benchmark

```bash
# Quick test with tiny model
.venv/bin/mac-whisper-speedtest -b -n 1 -i "MoonshineOnnxImplementation" -m moonshine-tiny

# Full comparison with default implementations
.venv/bin/mac-whisper-speedtest -b -n 3 -i "MoonshineOnnxImplementation,MLXWhisperImplementation"
```

### 3. Expected Performance

Based on Moonshine's benchmarks, we expect:

| Audio Length | Moonshine tiny | MLX Whisper tiny |
|-------------|----------------|------------------|
| ~5 seconds (jfk.wav) | ~100ms | ~500ms |
| 30 seconds | ~500ms | ~500ms |
| 60 seconds (ted_60.wav) | ~1000ms | ~1000ms |

**Moonshine should significantly outperform Whisper on short audio.**

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ONNX Runtime version conflicts | Low | Medium | Pin version in pyproject.toml |
| No segment/timestamp output | Certain | Low | Document limitation; segments remain empty |
| Model download on first use | Certain | Low | Expected behavior; same as other HF-based implementations |
| Different accuracy characteristics | Medium | Low | Document in comparison; Moonshine has similar WER |
| Large model files (~200-400MB) | Certain | Low | Normal for ASR; cached after first download |

---

## Future Enhancements (Not in MVP)

| Feature | Description | Priority |
|---------|-------------|----------|
| **MLX backend** | Add `MoonshineMLXImplementation` using native MLX (if available) | Medium |
| **Streaming mode** | Add real-time transcription using Moonshine's streaming architecture | High |
| **Non-English benchmarks** | Test multilingual variants (ar, zh, ja, ko, uk, vi, es) | Low |
| **Transformers backend** | Add HuggingFace transformers-based implementation | Low |

---

## Implementation Checklist

- [ ] Create `implementations/moonshine_onnx.py`
- [ ] Update `implementations/__init__.py` with registration
- [ ] Update `pyproject.toml` with dependency
- [ ] Run `uv sync` to install dependencies
- [ ] Create `tests/test_moonshine_integration.py`
- [ ] Run tests: `pytest tests/test_moonshine_integration.py -v`
- [ ] Run batch benchmark: `mac-whisper-speedtest -b -n 1 -i MoonshineOnnxImplementation`
- [ ] Verify model downloads successfully
- [ ] Verify transcription accuracy on test audio
- [ ] Document any issues encountered

---

## Notes on Streaming (Future Work)

Based on the [Moonshine Analysis](research_moonshine_analysis.md), streaming would require:

1. **Silero VAD** for voice activity detection
2. **Lookback buffer** to capture word beginnings
3. **Periodic re-transcription** during speech (every 200ms)
4. **New CLI mode** (`--stream`) for interactive streaming

This is documented in detail in the research document and can be implemented as a follow-up feature.
