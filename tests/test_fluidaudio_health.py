"""FluidAudio bridge health checks for upgrade verification.

Run before and after FluidAudio upgrades to verify:
1. Swift bridge executable exists and runs
2. Bridge can produce valid JSON output
3. Package.resolved versions are in expected range
4. Python implementation imports and instantiates correctly

Usage:
    pytest tests/test_fluidaudio_health.py -v
"""

import json
import platform
import subprocess
from pathlib import Path

import pytest

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="FluidAudio tests only run on macOS"
)

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
BRIDGE_DIR = PROJECT_ROOT / "tools" / "fluidaudio-bridge"
BRIDGE_EXECUTABLE = BRIDGE_DIR / ".build" / "release" / "fluidaudio-bridge"
PACKAGE_RESOLVED = BRIDGE_DIR / "Package.resolved"
TEST_AUDIO = PROJECT_ROOT / "tests" / "jfk.wav"


class TestBridgeExists:
    """Verify FluidAudio Swift bridge is built and accessible."""

    def test_bridge_directory_exists(self):
        """Bridge directory should exist."""
        assert BRIDGE_DIR.exists(), f"Bridge directory not found: {BRIDGE_DIR}"

    def test_bridge_executable_exists(self):
        """Bridge executable should exist (requires prior build)."""
        assert BRIDGE_EXECUTABLE.exists(), (
            f"Bridge executable not found at {BRIDGE_EXECUTABLE}. "
            "Build it with: cd tools/fluidaudio-bridge && swift build -c release"
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
        assert "fluidaudio-bridge" in result.stdout.lower(), "Help output missing expected content"
        print(f"Bridge help output:\n{result.stdout[:200]}...")


class TestBridgeOutput:
    """Verify bridge produces correct JSON output structure.

    Note: FluidAudio's JSON output differs from WhisperKit:
    - Has: text, transcription_time, processing_time, language
    - Does NOT have: segments (FluidAudio bridge doesn't provide segments)
    """

    @pytest.fixture
    def bridge_json_output(self) -> dict:
        """Run bridge and return parsed JSON output."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")
        if not TEST_AUDIO.exists():
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(TEST_AUDIO), "--format", "json"],
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

    def test_json_has_processing_time_field(self, bridge_json_output: dict):
        """JSON output should have 'processing_time' field (FluidAudio-specific)."""
        assert "processing_time" in bridge_json_output, "Missing 'processing_time' field"
        time_val = bridge_json_output["processing_time"]
        assert isinstance(time_val, (int, float)), "'processing_time' should be numeric"
        # processing_time can be 0 if not measured
        print(f"Processing time: {time_val:.4f}s")

    def test_json_has_language_field(self, bridge_json_output: dict):
        """JSON output should have 'language' field."""
        assert "language" in bridge_json_output, "Missing 'language' field"
        assert isinstance(bridge_json_output["language"], str), "'language' should be a string"

    def test_json_does_not_have_segments(self, bridge_json_output: dict):
        """FluidAudio bridge does NOT provide segments (unlike WhisperKit).

        This test documents the expected schema difference.
        If FluidAudio adds segments support later, update this test.
        """
        # This is a documentation test - FluidAudio doesn't provide segments
        if "segments" in bridge_json_output:
            # If segments appear in future versions, that's fine - just note it
            print("Note: FluidAudio now provides segments (schema changed)")
        else:
            # Expected behavior - no segments
            print("Confirmed: FluidAudio does not provide segments (expected)")

    def test_transcription_time_is_reasonable(self, bridge_json_output: dict):
        """Transcription time should be reasonable for short audio."""
        time_val = bridge_json_output.get("transcription_time", 0)
        # JFK audio is ~11 seconds; transcription should be faster than real-time
        assert time_val < 60, f"Transcription took too long: {time_val}s"
        assert time_val > 0.01, f"Transcription time suspiciously short: {time_val}s"

    def test_all_required_fields_present(self, bridge_json_output: dict):
        """Output should have all FluidAudio-specific required fields."""
        required_fields = ["text", "transcription_time", "processing_time", "language"]

        for field in required_fields:
            assert field in bridge_json_output, f"Missing required field: {field}"

        print(f"All {len(required_fields)} required fields present")


class TestPackageVersions:
    """Verify Package.resolved versions are in expected ranges.

    Version progression during upgrade:
    - Before: 0.1.x
    - Phase 1: 0.4.x
    - Phase 2: 0.8.x
    - Phase 3: 0.10.x
    """

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

    def test_fluidaudio_version_in_expected_range(self, package_resolved: dict):
        """FluidAudio version should be in expected upgrade range.

        Valid ranges during upgrade:
        - Before upgrade: 0.1.x
        - After Phase 1: 0.4.x
        - After Phase 2: 0.8.x
        - After Phase 3: 0.10.x

        This test accepts any version in these ranges.
        """
        version = self._get_pin_version(package_resolved, "fluidaudio")
        assert version != "not found", "FluidAudio not found in Package.resolved"

        parts = version.split(".")
        assert len(parts) >= 2, f"Invalid version format: {version}"

        major, minor = int(parts[0]), int(parts[1])
        assert major == 0, f"Unexpected major version: {major}"

        # Valid minor versions: 1, 4, 5-8, 9, 10 (covering all phases)
        valid_minors = [1, 4, 5, 6, 7, 8, 9, 10]
        assert minor in valid_minors, (
            f"FluidAudio {version} outside expected range. "
            f"Valid minor versions: {valid_minors}"
        )

        print(f"FluidAudio version: {version}")

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

    def test_swift_argument_parser_present(self, package_resolved: dict):
        """swift-argument-parser should be in dependencies."""
        version = self._get_pin_version(package_resolved, "swift-argument-parser")
        assert version != "not found", "swift-argument-parser not found in Package.resolved"
        print(f"swift-argument-parser version: {version}")


class TestPythonImplementation:
    """Verify Python FluidAudio implementation works correctly."""

    def test_fluidaudio_implementation_imports(self):
        """FluidAudioCoreMLImplementation should import successfully."""
        try:
            from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation
            assert FluidAudioCoreMLImplementation is not None
        except ImportError as e:
            pytest.fail(f"FluidAudioCoreMLImplementation import failed: {e}")

    def test_fluidaudio_implementation_instantiates(self):
        """FluidAudioCoreMLImplementation should instantiate (if bridge exists)."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built - cannot test instantiation")

        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation

        try:
            impl = FluidAudioCoreMLImplementation()
            assert impl is not None
            print("FluidAudioCoreMLImplementation instantiated successfully")
        except Exception as e:
            pytest.fail(f"FluidAudioCoreMLImplementation failed to instantiate: {e}")

    def test_implementation_in_registry(self):
        """FluidAudioCoreMLImplementation should be in the implementations registry."""
        from mac_whisper_speedtest.implementations import get_all_implementations

        all_impls = get_all_implementations()
        impl_names = [impl.__name__ for impl in all_impls]

        assert "FluidAudioCoreMLImplementation" in impl_names, (
            f"FluidAudioCoreMLImplementation not found in registry. Available: {impl_names}"
        )

    def test_get_params_returns_expected_keys(self):
        """get_params should return dict with expected keys."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation

        impl = FluidAudioCoreMLImplementation()
        impl.load_model("small")  # Note: FluidAudio ignores this parameter

        params = impl.get_params()

        assert isinstance(params, dict), "get_params should return dict"
        assert "model" in params, "params should include 'model'"
        assert params["model"] is not None, "model should not be None"

        # FluidAudio hardcodes the model name
        expected_model = "parakeet-tdt-0.6b-v2-coreml"
        assert params["model"] == expected_model, (
            f"Expected model '{expected_model}', got '{params['model']}'"
        )

        print(f"FluidAudio params: {params}")

    def test_get_params_shows_correct_backend(self):
        """get_params should indicate FluidAudio Swift Bridge backend."""
        if not BRIDGE_EXECUTABLE.exists():
            pytest.skip("Bridge not built")

        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation

        impl = FluidAudioCoreMLImplementation()
        impl.load_model("small")

        params = impl.get_params()

        assert "backend" in params, "params should include 'backend'"
        assert "FluidAudio" in params["backend"], (
            f"backend should mention FluidAudio, got: {params['backend']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
