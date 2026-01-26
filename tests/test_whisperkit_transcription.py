"""WhisperKit transcription smoke tests.

These tests verify that WhisperKit can actually transcribe audio via both:
1. Direct bridge CLI invocation
2. Python WhisperKitImplementation wrapper

Run with real models (not mocks) to catch compatibility issues.

Usage:
    pytest tests/test_whisperkit_transcription.py -v
    pytest tests/test_whisperkit_transcription.py -v -k "bridge"  # CLI only
    pytest tests/test_whisperkit_transcription.py -v -k "implementation"  # Python only

Note: These tests download models on first run and may take time.
"""

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="WhisperKit tests only run on macOS"
)

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
BRIDGE_EXECUTABLE = PROJECT_ROOT / "tools" / "whisperkit-bridge" / ".build" / "release" / "whisperkit-bridge"
TEST_AUDIO = PROJECT_ROOT / "tests" / "jfk.wav"

# Expected words in JFK audio transcription
# The famous quote: "Ask not what your country can do for you..."
JFK_EXPECTED_WORDS = ["ask", "country", "fellow", "american", "citizen"]


@pytest.fixture
def test_audio_path() -> Path:
    """Path to test audio file."""
    if not TEST_AUDIO.exists():
        pytest.skip(f"Test audio file not found: {TEST_AUDIO}")
    return TEST_AUDIO


@pytest.fixture
def test_audio_array(test_audio_path: Path) -> np.ndarray:
    """Load test audio as numpy array (16kHz mono float32)."""
    audio, sr = sf.read(test_audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != 16000:
        import scipy.signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

    return audio


class TestWhisperKitBridgeTranscription:
    """Test WhisperKit Swift bridge CLI directly."""

    @pytest.fixture
    def bridge_result(self, test_audio_path: Path) -> dict:
        """Run bridge and return parsed JSON output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--model", "small", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            pytest.fail(f"Bridge failed: {result.stderr}")

        return json.loads(result.stdout)

    def test_bridge_cli_transcribes_jfk_audio(self, bridge_result: dict):
        """Bridge should produce transcription for JFK audio."""
        text = bridge_result.get("text", "")

        assert len(text.strip()) > 0, "Transcription should not be empty"
        print(f"Bridge transcription: {text[:150]}...")

    def test_transcription_contains_expected_words(self, bridge_result: dict):
        """Transcription should contain expected words from JFK speech."""
        text = bridge_result.get("text", "").lower()

        found_words = [word for word in JFK_EXPECTED_WORDS if word in text]

        assert len(found_words) >= 2, (
            f"Transcription should contain at least 2 of {JFK_EXPECTED_WORDS}. "
            f"Found: {found_words}. Text: {text[:200]}"
        )
        print(f"Found expected words: {found_words}")

    def test_transcription_time_is_positive(self, bridge_result: dict):
        """Transcription time should be a positive value."""
        time_val = bridge_result.get("transcription_time", 0)

        assert time_val > 0, "Transcription time should be positive"
        assert time_val < 30, f"Transcription time seems too long: {time_val}s"
        print(f"Transcription time: {time_val:.4f}s")

    def test_segments_are_returned(self, bridge_result: dict):
        """Bridge should return segments array."""
        segments = bridge_result.get("segments", [])

        assert len(segments) > 0, "Should have at least one segment"
        print(f"Got {len(segments)} segments")

    def test_text_format_output(self, test_audio_path: Path):
        """Bridge should support text format output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--model", "small", "--format", "text"],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0, f"Text format failed: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "Text output should not be empty"
        print(f"Text output: {result.stdout[:150]}...")


class TestWhisperKitImplementationTranscription:
    """Test Python WhisperKitImplementation wrapper."""

    @pytest.fixture
    def implementation(self):
        """Create and configure WhisperKitImplementation."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        from mac_whisper_speedtest.implementations.whisperkit import WhisperKitImplementation

        impl = WhisperKitImplementation()
        impl.load_model("small")
        yield impl
        impl.cleanup()

    @pytest.mark.asyncio
    async def test_transcribe_produces_text(self, implementation, test_audio_array: np.ndarray):
        """WhisperKitImplementation should produce non-empty transcription."""
        result = await implementation.transcribe(test_audio_array)

        assert result is not None, "Result should not be None"
        assert result.text is not None, "Text should not be None"
        assert len(result.text.strip()) > 0, "Transcription should not be empty"

        print(f"Implementation transcription: {result.text[:150]}...")

    @pytest.mark.asyncio
    async def test_transcribe_contains_expected_words(self, implementation, test_audio_array: np.ndarray):
        """Transcription should contain expected JFK speech words."""
        result = await implementation.transcribe(test_audio_array)
        text_lower = result.text.lower()

        found_words = [word for word in JFK_EXPECTED_WORDS if word in text_lower]

        assert len(found_words) >= 2, (
            f"Should contain at least 2 of {JFK_EXPECTED_WORDS}. "
            f"Found: {found_words}"
        )

    @pytest.mark.asyncio
    async def test_transcribe_returns_segments(self, implementation, test_audio_array: np.ndarray):
        """Transcription should include segments."""
        result = await implementation.transcribe(test_audio_array)

        assert hasattr(result, 'segments'), "Result should have segments attribute"
        assert isinstance(result.segments, list), "Segments should be a list"
        # Note: segments may be empty depending on WhisperKit configuration

    @pytest.mark.asyncio
    async def test_transcribe_returns_language(self, implementation, test_audio_array: np.ndarray):
        """Transcription should include detected language."""
        result = await implementation.transcribe(test_audio_array)

        assert hasattr(result, 'language'), "Result should have language attribute"
        assert result.language is not None, "Language should not be None"
        # JFK speech is in English
        assert result.language.lower() in ["en", "english"], f"Unexpected language: {result.language}"

    @pytest.mark.asyncio
    async def test_transcription_time_stored(self, implementation, test_audio_array: np.ndarray):
        """Implementation should store transcription time from bridge."""
        result = await implementation.transcribe(test_audio_array)

        # The implementation stores this as a private attribute
        assert hasattr(result, '_transcription_time'), "Result should have _transcription_time"
        time_val = result._transcription_time

        assert time_val > 0, "Transcription time should be positive"
        print(f"Stored transcription time: {time_val:.4f}s")


class TestJSONOutputSchema:
    """Verify JSON output schema consistency for upgrade safety."""

    @pytest.fixture
    def bridge_output(self, test_audio_path: Path) -> dict:
        """Get bridge JSON output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--model", "small", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            pytest.fail(f"Bridge failed: {result.stderr}")

        return json.loads(result.stdout)

    def test_output_has_all_required_fields(self, bridge_output: dict):
        """Output should have all required top-level fields."""
        required_fields = ["text", "transcription_time", "language", "segments"]

        for field in required_fields:
            assert field in bridge_output, f"Missing required field: {field}"

    def test_output_field_types(self, bridge_output: dict):
        """Output fields should have correct types."""
        assert isinstance(bridge_output["text"], str), "text should be string"
        assert isinstance(bridge_output["transcription_time"], (int, float)), "transcription_time should be numeric"
        assert isinstance(bridge_output["language"], str), "language should be string"
        assert isinstance(bridge_output["segments"], list), "segments should be list"

    def test_segment_schema(self, bridge_output: dict):
        """Segments should follow expected schema."""
        segments = bridge_output.get("segments", [])

        if not segments:
            pytest.skip("No segments to validate")

        segment = segments[0]
        required_segment_fields = ["start", "end", "text"]

        for field in required_segment_fields:
            assert field in segment, f"Segment missing required field: {field}"

        assert isinstance(segment["start"], (int, float)), "segment.start should be numeric"
        assert isinstance(segment["end"], (int, float)), "segment.end should be numeric"
        assert isinstance(segment["text"], str), "segment.text should be string"

    def test_segment_text_concatenates_to_full_text(self, bridge_output: dict):
        """Segment texts combined should approximate full text."""
        segments = bridge_output.get("segments", [])
        full_text = bridge_output.get("text", "")

        if not segments:
            pytest.skip("No segments to validate")

        # Concatenate segment texts
        segment_texts = " ".join(seg["text"].strip() for seg in segments)

        # Normalize for comparison (remove extra whitespace)
        full_normalized = " ".join(full_text.split())
        segments_normalized = " ".join(segment_texts.split())

        # They should be similar (may not be exactly equal due to joining)
        # Check that segment text is at least 80% of full text length
        ratio = len(segments_normalized) / len(full_normalized) if full_normalized else 0

        assert ratio > 0.8, (
            f"Segment texts ({len(segments_normalized)} chars) don't match "
            f"full text ({len(full_normalized)} chars). Ratio: {ratio:.2f}"
        )


class TestModelVariants:
    """Test different model sizes work correctly."""

    @pytest.mark.parametrize("model", ["tiny", "base", "small"])
    def test_model_sizes_work(self, test_audio_path: Path, model: str):
        """Different model sizes should all produce output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--model", model, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0, f"Model '{model}' failed: {result.stderr}"

        output = json.loads(result.stdout)
        text = output.get("text", "")

        assert len(text.strip()) > 0, f"Model '{model}' produced empty transcription"
        print(f"Model '{model}' output: {text[:100]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
