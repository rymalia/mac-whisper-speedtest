"""MLX framework health checks for upgrade verification.

Run before and after MLX upgrades to verify:
1. MLX and MLX-Metal versions match
2. Basic MLX operations work
3. All MLX-dependent implementations can be imported

Usage:
    pytest tests/test_mlx_health.py -v
"""

import platform

import pytest

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="MLX tests only run on macOS"
)


class TestMLXVersionSync:
    """Verify MLX and MLX-Metal are synchronized."""

    def test_mlx_imports_successfully(self):
        """MLX core should import without errors."""
        import mlx.core as mx
        assert mx is not None

    def test_mlx_metal_imports_successfully(self):
        """MLX Metal backend should import without errors."""
        import mlx
        # mlx-metal is loaded implicitly; this tests the import chain
        assert hasattr(mlx, 'core')

    def test_mlx_version_available(self):
        """MLX version should be available via importlib.metadata."""
        from importlib.metadata import version
        mlx_version = version("mlx")
        assert mlx_version is not None
        assert len(mlx_version) > 0
        print(f"MLX version: {mlx_version}")

    def test_mlx_and_mlx_metal_versions_match(self):
        """CRITICAL: mlx and mlx-metal must have matching versions.

        This is the most important upgrade verification check.
        A version mismatch will cause runtime failures.
        """
        from importlib.metadata import version

        mlx_version = version("mlx")
        mlx_metal_version = version("mlx-metal")

        print(f"mlx: {mlx_version}, mlx-metal: {mlx_metal_version}")

        assert mlx_version is not None, "Could not determine mlx version"
        assert mlx_metal_version is not None, "Could not determine mlx-metal version"
        assert mlx_version == mlx_metal_version, \
            f"VERSION MISMATCH: mlx={mlx_version}, mlx-metal={mlx_metal_version}"


class TestMLXBasicOperations:
    """Verify basic MLX operations work correctly."""

    def test_array_creation(self):
        """Basic array creation should work."""
        import mlx.core as mx

        arr = mx.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.dtype == mx.int32

    def test_array_arithmetic(self):
        """Basic arithmetic operations should work."""
        import mlx.core as mx

        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = a + b

        # Force evaluation (MLX uses lazy evaluation)
        mx.eval(c)

        expected = [5.0, 7.0, 9.0]
        assert list(c.tolist()) == expected

    def test_matrix_multiplication(self):
        """Matrix multiplication (core ML operation) should work."""
        import mlx.core as mx

        # 2x3 @ 3x2 = 2x2
        a = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float32)
        b = mx.array([[1, 2], [3, 4], [5, 6]], dtype=mx.float32)
        c = mx.matmul(a, b)

        mx.eval(c)

        assert c.shape == (2, 2)
        # Verify computation: [1,2,3]·[1,3,5]^T = 1*1 + 2*3 + 3*5 = 22
        assert c[0, 0].item() == 22.0

    def test_gpu_device_available(self):
        """GPU device should be available on Apple Silicon."""
        import mlx.core as mx

        # MLX automatically uses GPU on Apple Silicon
        # This test verifies the Metal backend is working
        arr = mx.array([1, 2, 3])
        mx.eval(arr)  # Force GPU execution

        # If we get here without error, Metal backend is working
        assert True

    def test_random_number_generation(self):
        """Random number generation should work."""
        import mlx.core as mx

        # Generate random numbers
        key = mx.random.key(42)
        samples = mx.random.normal(shape=(100,), key=key)
        mx.eval(samples)

        assert samples.shape == (100,)
        # Basic sanity check: mean should be close to 0 for normal distribution
        mean = mx.mean(samples).item()
        assert -1.0 < mean < 1.0, f"Mean {mean} is unexpectedly far from 0"


class TestMLXImplementationImports:
    """Verify all MLX-dependent implementations can be imported."""

    def test_mlx_whisper_imports(self):
        """mlx-whisper package should import successfully."""
        try:
            from mlx_whisper import transcribe
            from mlx_whisper.load_models import load_model
            assert transcribe is not None
            assert load_model is not None
        except ImportError as e:
            pytest.fail(f"mlx-whisper import failed: {e}")

    def test_whisper_mps_imports(self):
        """whisper-mps package should import successfully."""
        try:
            from whisper_mps.whisper.transcribe import transcribe
            from whisper_mps.whisper.load_models import load_model
            assert transcribe is not None
            assert load_model is not None
        except ImportError as e:
            pytest.fail(f"whisper-mps import failed: {e}")

    def test_lightning_whisper_mlx_imports(self):
        """lightning-whisper-mlx package should import successfully."""
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            assert LightningWhisperMLX is not None
        except ImportError as e:
            pytest.fail(f"lightning-whisper-mlx import failed: {e}")

    def test_parakeet_mlx_imports(self):
        """parakeet-mlx package should import successfully."""
        try:
            from parakeet_mlx import from_pretrained
            assert from_pretrained is not None
        except ImportError as e:
            pytest.fail(f"parakeet-mlx import failed: {e}")

    def test_implementation_classes_instantiate(self):
        """All MLX implementation classes should instantiate."""
        from mac_whisper_speedtest.implementations import get_all_implementations

        mlx_impl_names = [
            "MLXWhisperImplementation",
            "WhisperMPSImplementation",
            "LightningWhisperMLXImplementation",
            "ParakeetMLXImplementation",
        ]

        all_impls = get_all_implementations()
        impl_map = {impl.__name__: impl for impl in all_impls}

        for name in mlx_impl_names:
            assert name in impl_map, f"{name} not found in implementations"

            # Try to instantiate
            try:
                instance = impl_map[name]()
                assert instance is not None
                print(f"  {name} instantiated successfully")
            except Exception as e:
                pytest.fail(f"{name} failed to instantiate: {e}")


class TestMLXDependencyConstraints:
    """Verify MLX version is compatible with wrapper libraries."""

    def test_mlx_version_in_expected_range(self):
        """MLX version should be in expected upgrade range."""
        from importlib.metadata import version as get_version

        mlx_version = get_version("mlx")
        major, minor, patch = map(int, mlx_version.split('.')[:3])

        # After upgrade, we expect 0.29.x or 0.30.x
        # Before upgrade, we expect 0.27.x
        assert major == 0, f"Unexpected major version: {major}"
        assert 27 <= minor <= 30, f"MLX version {mlx_version} outside expected range (0.27.x - 0.30.x)"

        print(f"MLX version {mlx_version} is within expected range")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
