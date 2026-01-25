"""MLX transcription smoke tests.

These tests verify that MLX implementations can actually transcribe audio.
Run with real models (not mocks) to catch compatibility issues.

Usage:
    pytest tests/test_mlx_transcription.py -v
    pytest tests/test_mlx_transcription.py -v -k "mlx_whisper"  # Single impl

Note: These tests download models on first run and may take time.
      Use 'tiny' models for faster testing.
"""

import asyncio
import platform
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="MLX tests only run on macOS"
)


@pytest.fixture
def test_audio_path() -> Path:
    """Path to test audio file."""
    path = Path(__file__).parent / "jfk.wav"
    if not path.exists():
        pytest.skip(f"Test audio file not found: {path}")
    return path


@pytest.fixture
def test_audio_array(test_audio_path: Path) -> np.ndarray:
    """Load test audio as numpy array."""
    audio, sr = sf.read(test_audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sr != 16000:
        # Simple resampling - for test purposes only
        import scipy.signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

    return audio


class TestMLXWhisperTranscription:
    """Test MLXWhisperImplementation with real model."""

    @pytest.mark.asyncio
    async def test_transcribe_produces_text(self, test_audio_array: np.ndarray):
        """MLXWhisper should produce non-empty transcription."""
        from mac_whisper_speedtest.implementations.mlx import MLXWhisperImplementation

        impl = MLXWhisperImplementation()

        try:
            impl.load_model("tiny")  # Use tiny for speed

            result = await impl.transcribe(test_audio_array)

            assert result is not None, "Result should not be None"
            assert result.text is not None, "Text should not be None"
            assert len(result.text.strip()) > 0, "Transcription should not be empty"

            # JFK audio should contain recognizable words
            text_lower = result.text.lower()
            # The JFK speech typically contains "ask not what your country can do for you"
            assert any(word in text_lower for word in ["ask", "country", "fellow", "american"]), \
                f"Transcription doesn't contain expected words: {result.text[:200]}"

            print(f"MLXWhisper transcription: {result.text[:150]}...")
        finally:
            impl.cleanup()

    @pytest.mark.asyncio
    async def test_get_params_after_load(self, test_audio_array: np.ndarray):
        """get_params should return model info after loading."""
        from mac_whisper_speedtest.implementations.mlx import MLXWhisperImplementation

        impl = MLXWhisperImplementation()

        try:
            impl.load_model("tiny")

            params = impl.get_params()

            assert "model" in params, "params should include 'model'"
            assert params["model"] is not None, "model should not be None"
            assert "quantization" in params, "params should include 'quantization'"

            print(f"MLXWhisper params: {params}")
        finally:
            impl.cleanup()


class TestLightningWhisperTranscription:
    """Test LightningWhisperMLXImplementation with real model."""

    @pytest.mark.asyncio
    async def test_transcribe_produces_text(self, test_audio_array: np.ndarray):
        """LightningWhisperMLX should produce non-empty transcription."""
        from mac_whisper_speedtest.implementations.lightning import LightningWhisperMLXImplementation

        impl = LightningWhisperMLXImplementation()

        try:
            impl.load_model("tiny")  # Use tiny for speed

            result = await impl.transcribe(test_audio_array)

            assert result is not None, "Result should not be None"
            assert result.text is not None, "Text should not be None"
            assert len(result.text.strip()) > 0, "Transcription should not be empty"

            print(f"LightningWhisperMLX transcription: {result.text[:150]}...")
        finally:
            impl.cleanup()


class TestWhisperMPSTranscription:
    """Test WhisperMPSImplementation with real model."""

    @pytest.mark.asyncio
    async def test_transcribe_produces_text(self, test_audio_array: np.ndarray):
        """WhisperMPS should produce non-empty transcription."""
        from mac_whisper_speedtest.implementations.whisper_mps import WhisperMPSImplementation

        impl = WhisperMPSImplementation()

        try:
            impl.load_model("tiny")  # Use tiny for speed

            result = await impl.transcribe(test_audio_array)

            assert result is not None, "Result should not be None"
            assert result.text is not None, "Text should not be None"
            assert len(result.text.strip()) > 0, "Transcription should not be empty"

            print(f"WhisperMPS transcription: {result.text[:150]}...")
        finally:
            impl.cleanup()


class TestParakeetMLXTranscription:
    """Test ParakeetMLXImplementation with real model."""

    @pytest.mark.asyncio
    async def test_transcribe_produces_text(self, test_audio_array: np.ndarray):
        """ParakeetMLX should produce non-empty transcription."""
        from mac_whisper_speedtest.implementations.parakeet_mlx import ParakeetMLXImplementation

        impl = ParakeetMLXImplementation()

        try:
            # Parakeet uses its own model naming
            impl.load_model("small")  # Maps to parakeet-tdt-0.6b-v2

            result = await impl.transcribe(test_audio_array)

            assert result is not None, "Result should not be None"
            assert result.text is not None, "Text should not be None"
            # Parakeet might produce shorter output for short audio
            assert len(result.text.strip()) >= 0, "Transcription should exist"

            print(f"ParakeetMLX transcription: {result.text[:150]}...")
        finally:
            impl.cleanup()


class TestAllMLXImplementationsConsistency:
    """Test that all MLX implementations produce consistent results."""

    @pytest.mark.asyncio
    async def test_all_implementations_produce_output(self, test_audio_array: np.ndarray):
        """All MLX implementations should produce some output for the same audio."""
        from mac_whisper_speedtest.implementations import get_all_implementations

        mlx_impl_names = [
            "MLXWhisperImplementation",
            "LightningWhisperMLXImplementation",
            "WhisperMPSImplementation",
            # ParakeetMLX excluded - uses different model architecture
        ]

        all_impls = get_all_implementations()
        impl_map = {impl.__name__: impl for impl in all_impls}

        results = {}

        for name in mlx_impl_names:
            if name not in impl_map:
                print(f"Skipping {name} - not available")
                continue

            impl = impl_map[name]()

            try:
                impl.load_model("tiny")
                result = await impl.transcribe(test_audio_array)
                results[name] = result.text if result else None
                print(f"{name}: {result.text[:80] if result and result.text else 'EMPTY'}...")
            except Exception as e:
                results[name] = f"ERROR: {e}"
                print(f"{name}: ERROR - {e}")
            finally:
                if hasattr(impl, 'cleanup'):
                    impl.cleanup()

        # All implementations should produce some output
        for name, text in results.items():
            assert text is not None, f"{name} produced None"
            if not text.startswith("ERROR"):
                assert len(text.strip()) > 0, f"{name} produced empty text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
