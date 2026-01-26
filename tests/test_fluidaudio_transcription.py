"""FluidAudio transcription smoke tests.

These tests verify that FluidAudio can actually transcribe audio via both:
1. Direct bridge CLI invocation
2. Python FluidAudioCoreMLImplementation wrapper

Run with real models (not mocks) to catch compatibility issues.

Usage:
    pytest tests/test_fluidaudio_transcription.py -v
    pytest tests/test_fluidaudio_transcription.py -v -k "bridge"  # CLI only
    pytest tests/test_fluidaudio_transcription.py -v -k "implementation"  # Python only

Note: These tests download models on first run and may take time.
      Model cache location: ~/Library/Application Support/FluidAudio/Models/
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
    reason="FluidAudio tests only run on macOS"
)

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
BRIDGE_EXECUTABLE = PROJECT_ROOT / "tools" / "fluidaudio-bridge" / ".build" / "release" / "fluidaudio-bridge"
TEST_AUDIO = PROJECT_ROOT / "tests" / "jfk.wav"
TEST_AUDIO_LONG = PROJECT_ROOT / "tests" / "ted_60.wav"

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


class TestFluidAudioBridgeTranscription:
    """Test FluidAudio Swift bridge CLI directly."""

    @pytest.fixture
    def bridge_result(self, test_audio_path: Path) -> dict:
        """Run bridge and return parsed JSON output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--format", "json"],
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

    def test_processing_time_is_reported(self, bridge_result: dict):
        """Bridge should report FluidAudio's internal processing time."""
        processing_time = bridge_result.get("processing_time", -1)

        # processing_time should be present (may be 0 if not measured)
        assert processing_time >= 0, "Processing time should be non-negative"
        print(f"Processing time: {processing_time:.4f}s")

    def test_text_format_output(self, test_audio_path: Path):
        """Bridge should support text format output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--format", "text"],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0, f"Text format failed: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "Text output should not be empty"
        print(f"Text output: {result.stdout[:150]}...")


class TestFluidAudioImplementationTranscription:
    """Test Python FluidAudioCoreMLImplementation wrapper."""

    @pytest.fixture
    def implementation(self):
        """Create and configure FluidAudioCoreMLImplementation."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation

        impl = FluidAudioCoreMLImplementation()
        impl.load_model("small")  # Note: FluidAudio ignores model parameter
        yield impl
        impl.cleanup()

    @pytest.mark.asyncio
    async def test_transcribe_produces_text(self, implementation, test_audio_array: np.ndarray):
        """FluidAudioCoreMLImplementation should produce non-empty transcription."""
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
    async def test_transcribe_returns_empty_segments(self, implementation, test_audio_array: np.ndarray):
        """FluidAudio returns empty segments list (documented behavior)."""
        result = await implementation.transcribe(test_audio_array)

        assert hasattr(result, 'segments'), "Result should have segments attribute"
        assert isinstance(result.segments, list), "Segments should be a list"
        # FluidAudio bridge doesn't provide segments - this is expected
        assert len(result.segments) == 0, (
            "FluidAudio should return empty segments list. "
            "If this fails, FluidAudio may have added segment support."
        )

    @pytest.mark.asyncio
    async def test_transcribe_returns_language(self, implementation, test_audio_array: np.ndarray):
        """Transcription should include detected language."""
        result = await implementation.transcribe(test_audio_array)

        assert hasattr(result, 'language'), "Result should have language attribute"
        assert result.language is not None, "Language should not be None"
        # FluidAudio defaults to "en"
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
    """Verify JSON output schema consistency for upgrade safety.

    FluidAudio schema differs from WhisperKit:
    - Has: text, transcription_time, processing_time, language
    - Does NOT have: segments
    """

    @pytest.fixture
    def bridge_output(self, test_audio_path: Path) -> dict:
        """Get bridge JSON output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(test_audio_path), "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            pytest.fail(f"Bridge failed: {result.stderr}")

        return json.loads(result.stdout)

    def test_output_has_all_required_fields(self, bridge_output: dict):
        """Output should have all required top-level fields."""
        required_fields = ["text", "transcription_time", "processing_time", "language"]

        for field in required_fields:
            assert field in bridge_output, f"Missing required field: {field}"

    def test_output_field_types(self, bridge_output: dict):
        """Output fields should have correct types."""
        assert isinstance(bridge_output["text"], str), "text should be string"
        assert isinstance(bridge_output["transcription_time"], (int, float)), "transcription_time should be numeric"
        assert isinstance(bridge_output["processing_time"], (int, float)), "processing_time should be numeric"
        assert isinstance(bridge_output["language"], str), "language should be string"

    def test_no_unexpected_fields(self, bridge_output: dict):
        """Output should not have unexpected fields (schema stability check)."""
        expected_fields = {"text", "transcription_time", "processing_time", "language"}
        actual_fields = set(bridge_output.keys())

        unexpected = actual_fields - expected_fields

        if unexpected:
            # Not a failure, but document it for upgrade awareness
            print(f"Note: Found additional fields: {unexpected}")
            print("These may be new in this FluidAudio version")


class TestLongerAudio:
    """Test with longer audio files if available."""

    def test_longer_audio_transcription(self, test_audio_path: Path):
        """Test transcription of longer audio (ted_60.wav if available)."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        if not TEST_AUDIO_LONG.exists():
            pytest.skip(f"Long test audio not found: {TEST_AUDIO_LONG}")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(TEST_AUDIO_LONG), "--format", "json"],
            capture_output=True,
            text=True,
            timeout=600  # Longer timeout for 60s audio
        )

        assert result.returncode == 0, f"Bridge failed on longer audio: {result.stderr}"

        output = json.loads(result.stdout)
        text = output.get("text", "")

        assert len(text.strip()) > 50, "Longer audio should produce substantial transcription"
        print(f"Long audio transcription ({len(text)} chars): {text[:200]}...")


class TestBatchCLI:
    """Test integration with the benchmark CLI."""

    def test_benchmark_cli_runs_fluidaudio(self, test_audio_path: Path):
        """Benchmark CLI should be able to run FluidAudio implementation."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        import sys
        result = subprocess.run(
            [
                sys.executable, "-m", "mac_whisper_speedtest.cli",
                "--batch",
                "--audio", str(test_audio_path),
                "--runs", "1",
                "--implementations", "FluidAudioCoreMLImplementation",
                "--model", "small"  # Ignored by FluidAudio but required by CLI
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT)
        )

        # Check that it ran (may have warnings but should complete)
        if result.returncode != 0:
            print(f"CLI stderr: {result.stderr[:500]}")

        # Look for implementation name or transcription in output
        assert "FluidAudio" in result.stdout or "ask" in result.stdout.lower(), (
            f"CLI output doesn't show FluidAudio results.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
