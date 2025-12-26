"""Integration tests for Parakeet implementations."""

import asyncio
import pytest
import numpy as np
import tempfile
import soundfile as sf
from unittest.mock import patch, MagicMock

from mac_whisper_speedtest.implementations import get_all_implementations, get_implementation_by_name
from mac_whisper_speedtest.implementations.base import TranscriptionResult


class TestParakeetIntegration:
    """Test Parakeet implementations integration."""

    @pytest.fixture
    def test_audio(self):
        """Create test audio data."""
        # Create a 1-second sine wave
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        return audio

    def test_parakeet_implementations_in_registry(self):
        """Test that Parakeet implementations are properly registered."""
        all_impls = get_all_implementations()
        impl_names = [impl.__name__ for impl in all_impls]
        
        # Check that ParakeetMLXImplementation is available
        assert "ParakeetMLXImplementation" in impl_names, "ParakeetMLXImplementation not found in registry"

    def test_get_parakeet_implementation_by_name(self):
        """Test getting Parakeet implementations by name."""
        # Test ParakeetMLXImplementation
        impl_class = get_implementation_by_name("ParakeetMLXImplementation")
        assert impl_class.__name__ == "ParakeetMLXImplementation"
        
        # Test that we can instantiate it
        impl = impl_class()
        assert impl is not None

    @pytest.mark.asyncio
    async def test_parakeet_mlx_basic_functionality(self, test_audio):
        """Test basic functionality of ParakeetMLXImplementation."""
        try:
            impl_class = get_implementation_by_name("ParakeetMLXImplementation")
            impl = impl_class()
            
            # Mock the parakeet-mlx model to avoid actual model downloading
            with patch('parakeet_mlx.from_pretrained') as mock_from_pretrained:
                mock_model = MagicMock()
                mock_result = MagicMock()
                mock_result.text = "test transcription from parakeet"
                mock_result.sentences = []
                mock_model.transcribe.return_value = mock_result
                mock_from_pretrained.return_value = mock_model

                # Test model loading
                model_name = "parakeet-tdt-0.6b-v2"
                impl.load_model(model_name)

                # Test get_params
                params = impl.get_params()
                assert "model" in params
                assert "implementation" in params
                assert params["implementation"] == "parakeet-mlx"

                # Test transcription
                result = await impl.transcribe(test_audio)
                assert isinstance(result, TranscriptionResult)
                assert hasattr(result, 'text')
                assert hasattr(result, 'segments')
                assert hasattr(result, 'language')
                assert result.text == "test transcription from parakeet"

                # Test cleanup
                impl.cleanup()
            
        except ImportError:
            pytest.skip("parakeet-mlx not available")

    def test_parakeet_model_name_mapping(self):
        """Test that Parakeet model name mapping works correctly."""
        try:
            impl_class = get_implementation_by_name("ParakeetMLXImplementation")
            impl = impl_class()
            
            # Mock the parakeet-mlx model to avoid actual model downloading
            with patch('parakeet_mlx.from_pretrained') as mock_from_pretrained:
                mock_model = MagicMock()
                mock_from_pretrained.return_value = mock_model

                # Test that common model names are mapped correctly
                test_cases = [
                    ("small", "mlx-community/parakeet-tdt-0.6b-v2"),
                    ("parakeet-tdt-0.6b-v2", "mlx-community/parakeet-tdt-0.6b-v2"),
                    ("parakeet-tdt-1.1b", "mlx-community/parakeet-tdt-1.1b"),
                ]

                for input_name, expected_repo in test_cases:
                    # Test the model name mapping by loading and checking params
                    impl.load_model(input_name)
                    params = impl.get_params()

                    # Verify the params include the expected repo
                    assert "model" in params
                    assert params["model"] == expected_repo, \
                        f"Expected {expected_repo} for input {input_name}, got {params['model']}"

                    impl.cleanup()
                    
        except ImportError:
            pytest.skip("parakeet-mlx not available")

    def test_parakeet_error_handling(self):
        """Test error handling in Parakeet implementations."""
        try:
            impl_class = get_implementation_by_name("ParakeetMLXImplementation")
            impl = impl_class()
            
            # Test transcription without loading model
            with pytest.raises(RuntimeError, match="Model not loaded"):
                asyncio.run(impl.transcribe(np.zeros(1000, dtype=np.float32)))
                
        except ImportError:
            pytest.skip("parakeet-mlx not available")

    def test_parakeet_audio_preprocessing(self, test_audio):
        """Test that Parakeet implementations handle audio preprocessing correctly."""
        try:
            impl_class = get_implementation_by_name("ParakeetMLXImplementation")
            impl = impl_class()
            
            # Mock the parakeet-mlx model to avoid actual model loading
            with patch('parakeet_mlx.from_pretrained') as mock_from_pretrained:
                mock_model = MagicMock()
                mock_result = MagicMock()
                mock_result.text = "test transcription"
                mock_result.sentences = []
                mock_model.transcribe.return_value = mock_result
                mock_from_pretrained.return_value = mock_model
                
                impl.load_model("test-model")
                
                # Test transcription with different audio formats
                result = asyncio.run(impl.transcribe(test_audio))
                
                # Verify that transcribe was called on the mock model
                assert mock_model.transcribe.called
                
                # Verify the result structure
                assert isinstance(result, TranscriptionResult)
                assert result.text == "test transcription"
                
        except ImportError:
            pytest.skip("parakeet-mlx not available")

    def test_models_directory_usage(self):
        """Test that Parakeet implementations use the correct models directory."""
        try:
            impl_class = get_implementation_by_name("ParakeetMLXImplementation")
            impl = impl_class()

            # Test that get_models_dir is called (patch in the implementation module)
            with patch('parakeet_mlx.from_pretrained') as mock_from_pretrained, \
                 patch('mac_whisper_speedtest.implementations.parakeet_mlx.get_models_dir') as mock_get_models_dir:

                mock_get_models_dir.return_value = "/test/models/dir"
                mock_model = MagicMock()
                mock_from_pretrained.return_value = mock_model

                impl.load_model("test-model")

                # Verify that get_models_dir was called
                mock_get_models_dir.assert_called_once()

        except ImportError:
            pytest.skip("parakeet-mlx not available")

    def test_benchmark_integration(self):
        """Test that Parakeet implementations work with the benchmark framework."""
        try:
            impl_class = get_implementation_by_name("ParakeetMLXImplementation")

            # Test that the implementation class exists and can be instantiated
            impl = impl_class()
            assert impl is not None

            # Test that the implementation name is correct
            assert impl_class.__name__ == "ParakeetMLXImplementation"

        except ImportError:
            pytest.skip("parakeet-mlx not available")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
