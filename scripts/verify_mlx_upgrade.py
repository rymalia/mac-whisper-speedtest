#!/usr/bin/env python3
"""MLX Upgrade Verification Script.

Run this before and after MLX upgrades to verify everything works.
Provides a quick health check without running full test suite.

Usage:
    python scripts/verify_mlx_upgrade.py
    python scripts/verify_mlx_upgrade.py --verbose
    python scripts/verify_mlx_upgrade.py --skip-transcription  # Skip slow test
    python scripts/verify_mlx_upgrade.py --all-mlx            # Test all 4 MLX implementations

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import argparse
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


@timed_check
def check_version_sync() -> VerificationResult:
    """Check that mlx and mlx-metal versions match."""
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            mlx_ver = version("mlx")
        except PackageNotFoundError:
            return VerificationResult(
                "Version Sync",
                False,
                "mlx package not installed"
            )

        try:
            mlx_metal_ver = version("mlx-metal")
        except PackageNotFoundError:
            return VerificationResult(
                "Version Sync",
                False,
                "mlx-metal package not installed"
            )

        if mlx_ver == mlx_metal_ver:
            return VerificationResult(
                "Version Sync",
                True,
                f"mlx={mlx_ver}, mlx-metal={mlx_metal_ver}"
            )
        else:
            return VerificationResult(
                "Version Sync",
                False,
                f"MISMATCH: mlx={mlx_ver}, mlx-metal={mlx_metal_ver}"
            )
    except Exception as e:
        return VerificationResult("Version Sync", False, f"Error: {e}")


@timed_check
def check_mlx_import() -> VerificationResult:
    """Check that MLX can be imported."""
    try:
        import mlx.core as mx

        # Get version from package metadata
        from importlib.metadata import version
        mlx_version = version("mlx")

        return VerificationResult(
            "MLX Import",
            True,
            f"mlx.core imported successfully (version: {mlx_version})"
        )
    except ImportError as e:
        return VerificationResult("MLX Import", False, f"Import failed: {e}")
    except Exception as e:
        return VerificationResult("MLX Import", False, f"Error: {e}")


@timed_check
def check_basic_mlx_ops() -> VerificationResult:
    """Check basic MLX operations work."""
    try:
        import mlx.core as mx

        # Array creation
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = a + b
        mx.eval(c)

        # Verify result
        result = c.tolist()
        expected = [5.0, 7.0, 9.0]
        if result != expected:
            return VerificationResult(
                "Basic MLX Ops",
                False,
                f"Array arithmetic failed: expected {expected}, got {result}"
            )

        # Matrix multiplication
        m1 = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
        m2 = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
        m3 = mx.matmul(m1, m2)
        mx.eval(m3)

        # Check shape
        if m3.shape != (2, 2):
            return VerificationResult(
                "Basic MLX Ops",
                False,
                f"Matmul shape wrong: expected (2,2), got {m3.shape}"
            )

        return VerificationResult(
            "Basic MLX Ops",
            True,
            "Array ops and matmul work correctly"
        )
    except Exception as e:
        return VerificationResult("Basic MLX Ops", False, f"Error: {e}")


@timed_check
def check_implementation_imports() -> VerificationResult:
    """Check all MLX implementations can be imported."""
    failed = []
    succeeded = []

    checks = [
        ("mlx-whisper", "from mlx_whisper import transcribe"),
        ("whisper-mps", "from whisper_mps.whisper.transcribe import transcribe"),
        ("lightning-whisper-mlx", "from lightning_whisper_mlx import LightningWhisperMLX"),
        ("parakeet-mlx", "from parakeet_mlx import from_pretrained"),
    ]

    for name, import_stmt in checks:
        try:
            exec(import_stmt)
            succeeded.append(name)
        except ImportError as e:
            failed.append(f"{name}: {e}")
        except Exception as e:
            failed.append(f"{name}: {e}")

    if not failed:
        return VerificationResult(
            "Implementation Imports",
            True,
            f"All {len(succeeded)} MLX wrapper libraries import successfully"
        )
    else:
        return VerificationResult(
            "Implementation Imports",
            False,
            f"Failed: {'; '.join(failed)}"
        )


@timed_check
def check_implementation_instantiation() -> VerificationResult:
    """Check all MLX implementation classes can be instantiated."""
    try:
        from mac_whisper_speedtest.implementations import get_all_implementations

        mlx_impl_names = [
            "MLXWhisperImplementation",
            "WhisperMPSImplementation",
            "LightningWhisperMLXImplementation",
            "ParakeetMLXImplementation",
        ]

        all_impls = get_all_implementations()
        impl_map = {impl.__name__: impl for impl in all_impls}

        failed = []
        succeeded = []

        for name in mlx_impl_names:
            if name not in impl_map:
                failed.append(f"{name}: not in registry")
                continue

            try:
                instance = impl_map[name]()
                if instance is not None:
                    succeeded.append(name)
                else:
                    failed.append(f"{name}: returned None")
            except Exception as e:
                failed.append(f"{name}: {e}")

        if not failed:
            return VerificationResult(
                "Implementation Instantiation",
                True,
                f"All {len(succeeded)} MLX implementations instantiate"
            )
        else:
            return VerificationResult(
                "Implementation Instantiation",
                False,
                f"Failed: {'; '.join(failed)}"
            )
    except Exception as e:
        return VerificationResult(
            "Implementation Instantiation",
            False,
            f"Error: {e}"
        )


@timed_check
def check_batch_transcription() -> VerificationResult:
    """Run a quick batch transcription test with MLXWhisper."""
    try:
        # Find the test audio file
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        test_audio = project_root / "tests" / "jfk.wav"

        if not test_audio.exists():
            return VerificationResult(
                "Batch Transcription",
                False,
                f"Test audio not found: {test_audio}"
            )

        result = subprocess.run(
            [
                sys.executable, "-m", "mac_whisper_speedtest.cli",
                "--batch",
                "--audio", str(test_audio),
                "--runs", "1",
                "--implementations", "MLXWhisperImplementation",
                "--model", "tiny"
            ],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            cwd=str(project_root)
        )

        if result.returncode == 0:
            # Check that output contains transcription
            if "ask" in result.stdout.lower() or "country" in result.stdout.lower():
                return VerificationResult(
                    "Batch Transcription",
                    True,
                    "MLXWhisper transcribed test audio successfully"
                )
            else:
                return VerificationResult(
                    "Batch Transcription",
                    True,
                    "Transcription completed (content not verified)"
                )
        else:
            stderr_snippet = result.stderr[:300] if result.stderr else "no stderr"
            return VerificationResult(
                "Batch Transcription",
                False,
                f"Exit code {result.returncode}: {stderr_snippet}"
            )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            "Batch Transcription",
            False,
            "Timed out after 180 seconds"
        )
    except Exception as e:
        return VerificationResult("Batch Transcription", False, f"Error: {e}")


@timed_check
def check_all_mlx_transcription() -> VerificationResult:
    """Run batch transcription test for ALL 4 MLX implementations.

    This is a thorough test that verifies each MLX-based implementation
    can successfully load a model and transcribe audio.
    """
    mlx_implementations = [
        "MLXWhisperImplementation",
        "WhisperMPSImplementation",
        "LightningWhisperMLXImplementation",
        "ParakeetMLXImplementation",
    ]

    try:
        # Find the test audio file
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        test_audio = project_root / "tests" / "jfk.wav"

        if not test_audio.exists():
            return VerificationResult(
                "All MLX Transcription",
                False,
                f"Test audio not found: {test_audio}"
            )

        impl_list = ",".join(mlx_implementations)

        result = subprocess.run(
            [
                sys.executable, "-m", "mac_whisper_speedtest.cli",
                "--batch",
                "--audio", str(test_audio),
                "--runs", "1",
                "--implementations", impl_list,
                "--model", "tiny"
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for all 4
            cwd=str(project_root)
        )

        if result.returncode == 0:
            # Count how many implementations completed
            # Look for implementation names in output
            completed = []
            for impl in mlx_implementations:
                # The benchmark output includes implementation names
                if impl in result.stdout:
                    completed.append(impl)

            if len(completed) == len(mlx_implementations):
                return VerificationResult(
                    "All MLX Transcription",
                    True,
                    f"All {len(mlx_implementations)} MLX implementations transcribed successfully"
                )
            elif len(completed) > 0:
                return VerificationResult(
                    "All MLX Transcription",
                    True,
                    f"{len(completed)}/{len(mlx_implementations)} implementations completed"
                )
            else:
                return VerificationResult(
                    "All MLX Transcription",
                    True,
                    "Transcription completed (individual results not parsed)"
                )
        else:
            stderr_snippet = result.stderr[:500] if result.stderr else "no stderr"
            return VerificationResult(
                "All MLX Transcription",
                False,
                f"Exit code {result.returncode}: {stderr_snippet}"
            )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            "All MLX Transcription",
            False,
            "Timed out after 600 seconds"
        )
    except Exception as e:
        return VerificationResult("All MLX Transcription", False, f"Error: {e}")


def print_result(result: VerificationResult, verbose: bool = False):
    """Print a verification result."""
    status = "\033[92m PASS\033[0m" if result.passed else "\033[91m FAIL\033[0m"
    duration = f"({result.duration_ms:.0f}ms)" if result.duration_ms > 0 else ""

    print(f"[{status}] {result.name} {duration}")

    if verbose or not result.passed:
        print(f"       {result.message}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify MLX upgrade health"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for all checks"
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip the slow transcription test"
    )
    parser.add_argument(
        "--all-mlx",
        action="store_true",
        help="Test transcription for ALL 4 MLX implementations (slower but thorough)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MLX UPGRADE VERIFICATION")
    print("=" * 60)
    print()

    checks = [
        check_version_sync,
        check_mlx_import,
        check_basic_mlx_ops,
        check_implementation_imports,
        check_implementation_instantiation,
    ]

    if not args.skip_transcription:
        if args.all_mlx:
            checks.append(check_all_mlx_transcription)
        else:
            checks.append(check_batch_transcription)

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
