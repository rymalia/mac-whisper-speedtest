"""WhisperKit bridge health checks for upgrade verification.

Run before and after WhisperKit upgrades to verify:
1. Swift bridge executable exists and runs
2. Bridge can produce valid JSON output
3. Package.resolved versions are in expected range
4. Python implementation imports and instantiates correctly

Usage:
    pytest tests/test_whisperkit_health.py -v
"""

import json
import platform
import subprocess
from pathlib import Path

import pytest

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="WhisperKit tests only run on macOS"
)

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
BRIDGE_DIR = PROJECT_ROOT / "tools" / "whisperkit-bridge"
BRIDGE_EXECUTABLE = BRIDGE_DIR / ".build" / "release" / "whisperkit-bridge"
PACKAGE_RESOLVED = BRIDGE_DIR / "Package.resolved"
TEST_AUDIO = PROJECT_ROOT / "tests" / "jfk.wav"


class TestBridgeExists:
    """Verify WhisperKit Swift bridge is built and accessible."""

    def test_bridge_directory_exists(self):
        """Bridge directory should exist."""
        assert BRIDGE_DIR.exists(), f"Bridge directory not found: {BRIDGE_DIR}"

    def test_bridge_executable_exists(self):
        """Bridge executable should exist (requires prior build)."""
        assert BRIDGE_EXECUTABLE.exists(), (
            f"Bridge executable not found at {BRIDGE_EXECUTABLE}. "
            "Build it with: cd tools/whisperkit-bridge && swift build -c release"
        )

    def test_bridge_is_executable(self):
        """Bridge should be an executable file."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        import os
        assert os.access(BRIDGE_EXECUTABLE, os.X_OK), "Bridge file is not executable"

    def test_bridge_help_works(self):
        """Bridge --help should return successfully."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "whisperkit-bridge" in result.stdout.lower(), "Help output missing expected content"
        print(f"Bridge help output:\n{result.stdout[:200]}...")


class TestBridgeOutput:
    """Verify bridge produces correct JSON output structure."""

    @pytest.fixture
    def bridge_json_output(self) -> dict:
        """Run bridge and return parsed JSON output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")
        if not TEST_AUDIO.exists():
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(TEST_AUDIO), "--model", "small", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300  # Model download may take time on first run
        )

        if result.returncode != 0:
            pytest.fail(f"Bridge failed: {result.stderr}")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"Bridge output is not valid JSON: {e}\nOutput: {result.stdout[:500]}")

    def test_json_has_text_field(self, bridge_json_output: dict):
        """JSON output should have 'text' field."""
        assert "text" in bridge_json_output, "Missing 'text' field"
        assert isinstance(bridge_json_output["text"], str), "'text' should be a string"

    def test_json_has_transcription_time_field(self, bridge_json_output: dict):
        """JSON output should have 'transcription_time' field."""
        assert "transcription_time" in bridge_json_output, "Missing 'transcription_time' field"
        time_val = bridge_json_output["transcription_time"]
        assert isinstance(time_val, (int, float)), "'transcription_time' should be numeric"
        assert time_val > 0, "'transcription_time' should be positive"
        print(f"Transcription time: {time_val:.4f}s")

    def test_json_has_language_field(self, bridge_json_output: dict):
        """JSON output should have 'language' field."""
        assert "language" in bridge_json_output, "Missing 'language' field"
        assert isinstance(bridge_json_output["language"], str), "'language' should be a string"

    def test_json_has_segments_field(self, bridge_json_output: dict):
        """JSON output should have 'segments' array."""
        assert "segments" in bridge_json_output, "Missing 'segments' field"
        assert isinstance(bridge_json_output["segments"], list), "'segments' should be a list"

    def test_segments_have_required_structure(self, bridge_json_output: dict):
        """Each segment should have start, end, and text fields."""
        segments = bridge_json_output.get("segments", [])

        if len(segments) == 0:
            pytest.skip("No segments in output to validate")

        for i, segment in enumerate(segments):
            assert "start" in segment, f"Segment {i} missing 'start'"
            assert "end" in segment, f"Segment {i} missing 'end'"
            assert "text" in segment, f"Segment {i} missing 'text'"

            assert isinstance(segment["start"], (int, float)), f"Segment {i} 'start' should be numeric"
            assert isinstance(segment["end"], (int, float)), f"Segment {i} 'end' should be numeric"
            assert isinstance(segment["text"], str), f"Segment {i} 'text' should be string"

        print(f"Validated {len(segments)} segments")

    def test_segment_timestamps_are_ordered(self, bridge_json_output: dict):
        """Segment timestamps should be non-decreasing."""
        segments = bridge_json_output.get("segments", [])

        if len(segments) < 2:
            pytest.skip("Need at least 2 segments to check ordering")

        for i in range(1, len(segments)):
            prev_end = segments[i-1]["end"]
            curr_start = segments[i]["start"]
            # Allow small overlap (timing precision issues)
            assert curr_start >= prev_end - 0.1, (
                f"Segment {i} starts ({curr_start}) before segment {i-1} ends ({prev_end})"
            )

    def test_transcription_time_is_reasonable(self, bridge_json_output: dict):
        """Transcription time should be reasonable for short audio."""
        time_val = bridge_json_output.get("transcription_time", 0)
        # JFK audio is ~11 seconds; transcription should be faster than real-time
        assert time_val < 60, f"Transcription took too long: {time_val}s"
        assert time_val > 0.01, f"Transcription time suspiciously short: {time_val}s"


class TestPackageVersions:
    """Verify Package.resolved versions are in expected ranges."""

    @pytest.fixture
    def package_resolved(self) -> dict:
        """Load and parse Package.resolved."""
        if not PACKAGE_RESOLVED.exists():
            pytest.skip(f"Package.resolved not found: {PACKAGE_RESOLVED}")

        with open(PACKAGE_RESOLVED) as f:
            return json.load(f)

    def _get_pin_version(self, package_resolved: dict, identity: str) -> str:
        """Extract version for a specific package identity."""
        pins = package_resolved.get("pins", [])
        for pin in pins:
            if pin.get("identity") == identity:
                state = pin.get("state", {})
                return state.get("version", "unknown")
        return "not found"

    def test_package_resolved_exists(self):
        """Package.resolved should exist."""
        assert PACKAGE_RESOLVED.exists(), f"Package.resolved not found: {PACKAGE_RESOLVED}"

    def test_package_resolved_is_valid_json(self, package_resolved: dict):
        """Package.resolved should be valid JSON with expected structure."""
        assert "pins" in package_resolved, "Package.resolved missing 'pins' key"
        assert isinstance(package_resolved["pins"], list), "'pins' should be a list"

    def test_whisperkit_version_in_expected_range(self, package_resolved: dict):
        """WhisperKit version should be in expected upgrade range.

        Before upgrade: 0.13.x
        After Phase 1: 0.14.x
        After Phase 2: 0.15.x
        """
        version = self._get_pin_version(package_resolved, "whisperkit")
        assert version != "not found", "WhisperKit not found in Package.resolved"

        parts = version.split(".")
        assert len(parts) >= 2, f"Invalid version format: {version}"

        major, minor = int(parts[0]), int(parts[1])
        assert major == 0, f"Unexpected major version: {major}"
        assert 13 <= minor <= 15, f"WhisperKit {version} outside expected range (0.13.x - 0.15.x)"

        print(f"WhisperKit version: {version}")

    def test_swift_transformers_version_in_expected_range(self, package_resolved: dict):
        """swift-transformers version should be in expected upgrade range.

        Before upgrade: 0.1.x
        After Phase 1: 1.0.x or higher
        After Phase 2: 1.1.x or higher
        """
        version = self._get_pin_version(package_resolved, "swift-transformers")
        assert version != "not found", "swift-transformers not found in Package.resolved"

        parts = version.split(".")
        assert len(parts) >= 2, f"Invalid version format: {version}"

        major, minor = int(parts[0]), int(parts[1])
        # Valid ranges: 0.1.x (before) or 1.x.x (after)
        valid = (major == 0 and minor == 1) or (major >= 1)
        assert valid, f"swift-transformers {version} outside expected range"

        print(f"swift-transformers version: {version}")

    def test_all_pins_have_versions(self, package_resolved: dict):
        """All pins should have version information."""
        pins = package_resolved.get("pins", [])

        for pin in pins:
            identity = pin.get("identity", "unknown")
            state = pin.get("state", {})
            version = state.get("version")

            # Some pins use branch/revision instead of version
            if version is None and "branch" not in state and "revision" not in state:
                pytest.fail(f"Pin '{identity}' has no version, branch, or revision")


class TestPythonImplementation:
    """Verify Python WhisperKit implementation works correctly."""

    def test_whisperkit_implementation_imports(self):
        """WhisperKitImplementation should import successfully."""
        try:
            from mac_whisper_speedtest.implementations.whisperkit import WhisperKitImplementation
            assert WhisperKitImplementation is not None
        except ImportError as e:
            pytest.fail(f"WhisperKitImplementation import failed: {e}")

    def test_whisperkit_implementation_instantiates(self):
        """WhisperKitImplementation should instantiate (if bridge exists)."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built - cannot test instantiation")

        from mac_whisper_speedtest.implementations.whisperkit import WhisperKitImplementation

        try:
            impl = WhisperKitImplementation()
            assert impl is not None
            print("WhisperKitImplementation instantiated successfully")
        except Exception as e:
            pytest.fail(f"WhisperKitImplementation failed to instantiate: {e}")

    def test_implementation_in_registry(self):
        """WhisperKitImplementation should be in the implementations registry."""
        from mac_whisper_speedtest.implementations import get_all_implementations

        all_impls = get_all_implementations()
        impl_names = [impl.__name__ for impl in all_impls]

        assert "WhisperKitImplementation" in impl_names, (
            f"WhisperKitImplementation not found in registry. Available: {impl_names}"
        )

    def test_get_params_returns_expected_keys(self):
        """get_params should return dict with expected keys."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        from mac_whisper_speedtest.implementations.whisperkit import WhisperKitImplementation

        impl = WhisperKitImplementation()
        impl.load_model("small")

        params = impl.get_params()

        assert isinstance(params, dict), "get_params should return dict"
        assert "model" in params, "params should include 'model'"
        assert params["model"] is not None, "model should not be None"

        print(f"WhisperKit params: {params}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
