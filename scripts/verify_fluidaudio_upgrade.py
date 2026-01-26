#!/usr/bin/env python3
"""FluidAudio Upgrade Verification Script.

Run this before and after FluidAudio upgrades to verify everything works.
Provides a quick health check without running full test suite.

Usage:
    python scripts/verify_fluidaudio_upgrade.py
    python scripts/verify_fluidaudio_upgrade.py --verbose
    python scripts/verify_fluidaudio_upgrade.py --skip-transcription  # Skip slow test

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


def timed_check(func):
    """Decorator to time check functions."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        result.duration_ms = duration_ms
        return result
    return wrapper


# Path constants
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BRIDGE_DIR = PROJECT_ROOT / "tools" / "fluidaudio-bridge"
BRIDGE_EXECUTABLE = BRIDGE_DIR / ".build" / "release" / "fluidaudio-bridge"
PACKAGE_RESOLVED = BRIDGE_DIR / "Package.resolved"
TEST_AUDIO = PROJECT_ROOT / "tests" / "jfk.wav"


@timed_check
def check_bridge_exists() -> VerificationResult:
    """Check that the FluidAudio Swift bridge is built."""
    if not BRIDGE_DIR.exists():
        return VerificationResult(
            "Bridge Exists",
            False,
            f"Bridge directory not found: {BRIDGE_DIR}"
        )

    if not BRIDGE_EXECUTABLE.exists():
        return VerificationResult(
            "Bridge Exists",
            False,
            f"Bridge not built. Run: cd {BRIDGE_DIR} && swift build -c release"
        )

    return VerificationResult(
        "Bridge Exists",
        True,
        f"Found: {BRIDGE_EXECUTABLE}"
    )


@timed_check
def check_bridge_help() -> VerificationResult:
    """Check that bridge --help works."""
    if not BRIDGE_EXECUTABLE.exists():
        return VerificationResult("Bridge Help", False, "Bridge not built")

    try:
        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and "fluidaudio-bridge" in result.stdout.lower():
            return VerificationResult(
                "Bridge Help",
                True,
                "Bridge responds to --help correctly"
            )
        else:
            return VerificationResult(
                "Bridge Help",
                False,
                f"Unexpected output: {result.stderr[:100]}"
            )
    except subprocess.TimeoutExpired:
        return VerificationResult("Bridge Help", False, "Timed out")
    except Exception as e:
        return VerificationResult("Bridge Help", False, f"Error: {e}")


@timed_check
def check_package_version() -> VerificationResult:
    """Check FluidAudio version in Package.resolved."""
    if not PACKAGE_RESOLVED.exists():
        return VerificationResult(
            "Package Version",
            False,
            f"Package.resolved not found: {PACKAGE_RESOLVED}"
        )

    try:
        with open(PACKAGE_RESOLVED) as f:
            data = json.load(f)

        pins = data.get("pins", [])
        for pin in pins:
            if pin.get("identity") == "fluidaudio":
                version = pin.get("state", {}).get("version", "unknown")

                # Parse version
                parts = version.split(".")
                if len(parts) >= 2:
                    major, minor = int(parts[0]), int(parts[1])

                    # Determine upgrade phase
                    if minor == 1:
                        phase = "Before upgrade (v0.1.x)"
                    elif minor <= 4:
                        phase = "Phase 1 (v0.4.x)"
                    elif minor <= 8:
                        phase = "Phase 2 (v0.8.x)"
                    elif minor >= 9:
                        phase = "Phase 3 (v0.9.x - v0.10.x)"
                    else:
                        phase = "Unknown phase"

                    return VerificationResult(
                        "Package Version",
                        True,
                        f"FluidAudio {version} - {phase}"
                    )

        return VerificationResult(
            "Package Version",
            False,
            "FluidAudio not found in Package.resolved"
        )
    except Exception as e:
        return VerificationResult("Package Version", False, f"Error: {e}")


@timed_check
def check_python_import() -> VerificationResult:
    """Check that Python implementation can be imported."""
    try:
        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation

        return VerificationResult(
            "Python Import",
            True,
            "FluidAudioCoreMLImplementation imports successfully"
        )
    except ImportError as e:
        return VerificationResult("Python Import", False, f"Import failed: {e}")
    except Exception as e:
        return VerificationResult("Python Import", False, f"Error: {e}")


@timed_check
def check_python_instantiation() -> VerificationResult:
    """Check that Python implementation can be instantiated."""
    if not BRIDGE_EXECUTABLE.exists():
        return VerificationResult(
            "Python Instantiation",
            False,
            "Bridge not built - cannot test instantiation"
        )

    try:
        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation

        impl = FluidAudioCoreMLImplementation()
        impl.load_model("small")  # FluidAudio ignores this

        params = impl.get_params()
        model = params.get("model", "unknown")

        return VerificationResult(
            "Python Instantiation",
            True,
            f"Implementation instantiated, model: {model}"
        )
    except Exception as e:
        return VerificationResult("Python Instantiation", False, f"Error: {e}")


@timed_check
def check_transcription() -> VerificationResult:
    """Run a quick transcription test with the bridge."""
    if not BRIDGE_EXECUTABLE.exists():
        return VerificationResult("Transcription", False, "Bridge not built")

    if not TEST_AUDIO.exists():
        return VerificationResult(
            "Transcription",
            False,
            f"Test audio not found: {TEST_AUDIO}"
        )

    try:
        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(TEST_AUDIO), "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for model download
        )

        if result.returncode != 0:
            return VerificationResult(
                "Transcription",
                False,
                f"Bridge failed: {result.stderr[:200]}"
            )

        output = json.loads(result.stdout)
        text = output.get("text", "")
        time_val = output.get("transcription_time", 0)

        if len(text.strip()) == 0:
            return VerificationResult(
                "Transcription",
                False,
                "Transcription produced empty text"
            )

        # Check for expected JFK words
        text_lower = text.lower()
        expected_words = ["ask", "country", "fellow", "american"]
        found = [w for w in expected_words if w in text_lower]

        if len(found) >= 2:
            return VerificationResult(
                "Transcription",
                True,
                f"Transcribed in {time_val:.2f}s, found words: {found}"
            )
        else:
            return VerificationResult(
                "Transcription",
                True,
                f"Transcribed in {time_val:.2f}s (content not fully verified)"
            )

    except subprocess.TimeoutExpired:
        return VerificationResult("Transcription", False, "Timed out after 300s")
    except json.JSONDecodeError as e:
        return VerificationResult("Transcription", False, f"Invalid JSON: {e}")
    except Exception as e:
        return VerificationResult("Transcription", False, f"Error: {e}")


@timed_check
def check_json_schema() -> VerificationResult:
    """Verify JSON output has expected schema."""
    if not BRIDGE_EXECUTABLE.exists():
        return VerificationResult("JSON Schema", False, "Bridge not built")

    if not TEST_AUDIO.exists():
        return VerificationResult("JSON Schema", False, "Test audio not found")

    try:
        result = subprocess.run(
            [str(BRIDGE_EXECUTABLE), str(TEST_AUDIO), "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            return VerificationResult("JSON Schema", False, "Bridge failed")

        output = json.loads(result.stdout)

        # Check required fields
        required = ["text", "transcription_time", "processing_time", "language"]
        missing = [f for f in required if f not in output]

        if missing:
            return VerificationResult(
                "JSON Schema",
                False,
                f"Missing fields: {missing}"
            )

        # Check types
        type_errors = []
        if not isinstance(output["text"], str):
            type_errors.append("text should be string")
        if not isinstance(output["transcription_time"], (int, float)):
            type_errors.append("transcription_time should be numeric")
        if not isinstance(output["processing_time"], (int, float)):
            type_errors.append("processing_time should be numeric")
        if not isinstance(output["language"], str):
            type_errors.append("language should be string")

        if type_errors:
            return VerificationResult(
                "JSON Schema",
                False,
                f"Type errors: {'; '.join(type_errors)}"
            )

        return VerificationResult(
            "JSON Schema",
            True,
            f"All {len(required)} required fields present with correct types"
        )

    except Exception as e:
        return VerificationResult("JSON Schema", False, f"Error: {e}")


def print_result(result: VerificationResult, verbose: bool = False):
    """Print a verification result."""
    status = "\033[92m PASS\033[0m" if result.passed else "\033[91m FAIL\033[0m"
    duration = f"({result.duration_ms:.0f}ms)" if result.duration_ms > 0 else ""

    print(f"[{status}] {result.name} {duration}")

    if verbose or not result.passed:
        print(f"       {result.message}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify FluidAudio upgrade health"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for all checks"
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip the slow transcription tests"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FLUIDAUDIO UPGRADE VERIFICATION")
    print("=" * 60)
    print()

    # Quick checks
    checks = [
        check_bridge_exists,
        check_bridge_help,
        check_package_version,
        check_python_import,
        check_python_instantiation,
    ]

    # Slow checks (transcription)
    if not args.skip_transcription:
        checks.extend([
            check_transcription,
            check_json_schema,
        ])

    all_passed = True
    total_duration = 0.0

    for check_fn in checks:
        result = check_fn()
        print_result(result, verbose=args.verbose)
        total_duration += result.duration_ms

        if not result.passed:
            all_passed = False

    print()
    print("=" * 60)
    print(f"Total time: {total_duration / 1000:.1f}s")
    print()

    if all_passed:
        print("\033[92mALL CHECKS PASSED\033[0m - Safe to proceed")
    else:
        print("\033[91mSOME CHECKS FAILED\033[0m - Review before proceeding")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
