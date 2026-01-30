# Feature Plan: Add VibeVoice Implementation

**Date:** 2026-01-26
**Status:** Planning
**Related:** [mlx-audio Analysis](research_mlx-audio_analysis.md)

---

## Summary

Add `VibeVoiceImplementation` as the 10th ASR implementation, using mlx-audio's VibeVoice-ASR model. This is a 9B parameter LLM-based ASR with speaker diarization.

**What's different about VibeVoice:**
- LLM-based (Qwen2 7B) rather than encoder-decoder like Whisper
- Outputs structured JSON with speaker IDs and timestamps
- Requires 24kHz audio (our benchmark uses 16kHz)
- Single model architecture (no tiny/small/large variants—only quantization levels)

---

## Model Variants

| Model | Bits | Approx Size | Min RAM | Notes |
|-------|------|-------------|---------|-------|
| `mlx-community/VibeVoice-ASR-4bit` | 4-bit | ~5 GB | 8 GB | Lowest memory |
| `mlx-community/VibeVoice-ASR-5bit` | 5-bit | ~6 GB | 10 GB | Low memory |
| `mlx-community/VibeVoice-ASR-6bit` | 6-bit | ~7 GB | 12 GB | **Default** - good balance |
| `mlx-community/VibeVoice-ASR-8bit` | 8-bit | ~10 GB | 16 GB | Higher quality |
| `mlx-community/VibeVoice-ASR-bf16` | bf16 | ~18 GB | 32 GB | Full precision |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/mac_whisper_speedtest/implementations/vibevoice.py` | CREATE | New VibeVoice implementation |
| `src/mac_whisper_speedtest/implementations/__init__.py` | MODIFY | Register VibeVoiceImplementation |
| `pyproject.toml` | MODIFY | Add `mlx-audio` dependency |
| `tests/test_vibevoice_integration.py` | CREATE | Integration tests |

---

## Implementation Details

### 1. Create `vibevoice.py`

Follow the pattern from `parakeet_mlx.py`:

```python
"""VibeVoice ASR implementation using mlx-audio."""

import platform
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation


class VibeVoiceImplementation(WhisperImplementation):
    """VibeVoice ASR implementation using mlx-audio.

    VibeVoice is a 9B parameter LLM-based ASR model that outputs
    structured JSON with speaker IDs and timestamps.
    """

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self._model = None
        self._hf_repo = None

        if platform.system() != "Darwin":
            raise RuntimeError("VibeVoice requires macOS with Apple Silicon")

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name."""
        from mlx_audio.stt import load

        self.model_name = model_name
        self.log.info(f"Loading VibeVoice model parameter: {self.model_name}")

        # Map explicit quantization variants
        model_map = {
            "vibevoice-4bit": "mlx-community/VibeVoice-ASR-4bit",
            "vibevoice-5bit": "mlx-community/VibeVoice-ASR-5bit",
            "vibevoice-6bit": "mlx-community/VibeVoice-ASR-6bit",
            "vibevoice-8bit": "mlx-community/VibeVoice-ASR-8bit",
            "vibevoice-bf16": "mlx-community/VibeVoice-ASR-bf16",
        }

        # Default to 6-bit (good balance of quality and memory)
        self._hf_repo = model_map.get(model_name, "mlx-community/VibeVoice-ASR-6bit")

        self.log.info(f"Loading VibeVoice from {self._hf_repo}")

        # Memory check
        self._check_memory()

        self._model = load(self._hf_repo)
        self.log.info("Successfully loaded VibeVoice model")

    def _check_memory(self) -> None:
        """Warn if available memory is low for the selected variant."""
        try:
            import psutil
            available_mem = psutil.virtual_memory().available / (1024**3)
            min_mem = {"4bit": 8, "5bit": 10, "6bit": 12, "8bit": 16, "bf16": 32}
            for variant, req in min_mem.items():
                if variant in self._hf_repo and available_mem < req:
                    self.log.warning(
                        f"VibeVoice-{variant} needs ~{req}GB RAM, "
                        f"only {available_mem:.1f}GB available"
                    )
                    break
        except ImportError:
            pass  # psutil not available

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy.signal import resample_poly
        gcd = np.gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // gcd, orig_sr // gcd)

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info("Transcribing with VibeVoice")

        # CRITICAL: Resample 16kHz → 24kHz (VibeVoice requires 24kHz)
        audio_24k = self._resample_audio(audio, 16000, 24000)

        # Ensure float32
        if audio_24k.dtype != np.float32:
            audio_24k = audio_24k.astype(np.float32)

        result = self._model.generate(
            audio=audio_24k,
            max_tokens=8192,
            temperature=0.0
        )

        # Extract text and segments
        text = result.text if hasattr(result, 'text') else ""
        segments = result.segments if hasattr(result, 'segments') else []
        language = result.language if hasattr(result, 'language') else None

        if text:
            self.log.info(f"Transcription completed: '{text[:50]}...' ({len(segments)} segments)")
        else:
            self.log.warning("VibeVoice returned empty transcription")

        return TranscriptionResult(
            text=text,
            segments=segments,  # Includes speaker_id from VibeVoice
            language=language,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self._hf_repo,
            "implementation": "vibevoice (mlx-audio)",
            "platform": "Apple Silicon (MLX)",
            "architecture": "LLM-based (Qwen2 7B)",
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self._model = None
        self._hf_repo = None
```

### 2. Sample Rate Handling

**Critical issue:** Our benchmark uses 16kHz audio, VibeVoice requires 24kHz.

**Solution:** Resample in the implementation using scipy (already a transitive dependency via mlx-audio):
```python
from scipy.signal import resample_poly

def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // gcd, orig_sr // gcd)
```

### 3. Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing deps ...
    "mlx-audio>=0.2.0",  # For VibeVoice ASR
]
```

**Transitive dependencies brought in by mlx-audio:**
- `mlx-lm==0.30.5` — We already have mlx 0.24.1, may need compatibility check
- `transformers>=5.0.0` — We already have 5.0.0 ✓
- `scipy` — Already have ✓

### 4. Registration in `__init__.py`

Add after the WhisperMPSImplementation block:

```python
# Try to import VibeVoiceImplementation
try:
    if importlib.util.find_spec("mlx_audio"):
        import platform
        if platform.system() == "Darwin":
            from mac_whisper_speedtest.implementations.vibevoice import VibeVoiceImplementation
            available_implementations.append(VibeVoiceImplementation)
            logger.info("VibeVoiceImplementation loaded successfully")
        else:
            logger.warning("VibeVoice is only supported on macOS")
    else:
        logger.warning("mlx-audio not found, VibeVoiceImplementation will not be available")
except ImportError as e:
    logger.warning(f"Failed to import VibeVoiceImplementation: {e}")
```

---

## Special Features

VibeVoice supports features we can expose:

| Feature | Description | MVP | Future |
|---------|-------------|-----|--------|
| **Speaker diarization** | `segments[*].speaker_id` | ✅ Capture in segments | - |
| **Context injection** | `model.generate(context="terms")` | ❌ | CLI flag `--context` |
| **Streaming** | `model.stream_transcribe()` | ❌ | Real-time output |
| **Timestamps** | `segments[*].start/end` | ✅ Already in segments | - |

For MVP, capture speaker_id and timestamps in segments. Context/streaming can be added later.

---

## Verification Plan

1. **Unit test**: Create `tests/test_vibevoice_integration.py`
   ```python
   def test_vibevoice_loads_model():
       impl = VibeVoiceImplementation()
       impl.load_model("vibevoice-6bit")
       assert impl._model is not None

   def test_vibevoice_transcribes():
       # Use tests/jfk.wav
       ...
   ```

2. **Batch benchmark**:
   ```bash
   .venv/bin/mac-whisper-speedtest -b -n 1 -i "VibeVoiceImplementation"
   ```

3. **Verify segments include speaker_id**:
   ```python
   result = await impl.transcribe(audio)
   assert any('speaker_id' in seg for seg in result.segments)
   ```

4. **Memory monitoring**: Check peak memory during load and inference

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Large model size (9B params) | High | Medium | Document requirements, add warning, default to 6-bit |
| Sample rate mismatch (16kHz vs 24kHz) | Certain | High | Resample in implementation |
| mlx-audio API changes | Low | Medium | Pin version in pyproject.toml |
| Slower than Whisper models | Certain | Low | Expected - LLM-based models are slower; document this |
| mlx version conflicts | Medium | Medium | Check compatibility with our mlx 0.24.1 |

---

## Future Enhancements

After MVP is working:

1. **Context injection CLI flag**: `--context "domain terms"` to improve domain-specific accuracy
2. **Streaming mode**: Real-time transcription output for long audio
3. **Speaker count hint**: Pre-specify expected number of speakers
4. **Quality comparison**: Document transcription quality vs Whisper variants
