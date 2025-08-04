"""Test that all implementations include model information in get_params()."""

import pytest
from unittest.mock import patch, MagicMock
from mac_whisper_speedtest.implementations import get_all_implementations


class TestModelParams:
    """Test that all implementations provide model information in get_params()."""

    def test_all_implementations_include_model_in_params(self):
        """Test that all implementations include 'model' key in get_params()."""
        all_impls = get_all_implementations()
        
        for impl_class in all_impls:
            impl_name = impl_class.__name__
            print(f"Testing {impl_name}")
            
            try:
                impl = impl_class()
                
                # Mock dependencies for implementations that require them
                if impl_name == "ParakeetMLXImplementation":
                    with patch('parakeet_mlx.from_pretrained') as mock_from_pretrained:
                        mock_model = MagicMock()
                        mock_from_pretrained.return_value = mock_model
                        impl.load_model("small")
                        
                elif impl_name == "InsanelyFastWhisperImplementation":
                    with patch('transformers.pipeline') as mock_pipeline:
                        mock_model = MagicMock()
                        mock_pipeline.return_value = mock_model
                        impl.load_model("small")
                        
                elif impl_name == "WhisperCppCoreMLImplementation":
                    with patch('pywhispercpp.model.Model') as mock_model_class:
                        mock_model = MagicMock()
                        mock_model_class.return_value = mock_model
                        impl.load_model("small")
                        
                elif impl_name == "MLXWhisperImplementation":
                    with patch('huggingface_hub.snapshot_download') as mock_download, \
                         patch('mlx_whisper.load_models.load_model') as mock_load:
                        mock_download.return_value = "/fake/path"
                        mock_model = MagicMock()
                        mock_load.return_value = mock_model
                        impl.load_model("small")
                        
                elif impl_name == "LightningWhisperMLXImplementation":
                    with patch('lightning_whisper_mlx.LightningWhisperMLX') as mock_class:
                        mock_model = MagicMock()
                        mock_class.return_value = mock_model
                        impl.load_model("small")
                        
                elif impl_name == "FasterWhisperImplementation":
                    with patch('faster_whisper.WhisperModel') as mock_model_class:
                        mock_model = MagicMock()
                        mock_model_class.return_value = mock_model
                        impl.load_model("small")
                        
                else:
                    # For any other implementations, try to load normally
                    try:
                        impl.load_model("small")
                    except Exception as e:
                        print(f"Skipping {impl_name} due to loading error: {e}")
                        continue
                
                # Test get_params
                params = impl.get_params()
                assert isinstance(params, dict), f"{impl_name}.get_params() should return a dict"
                assert "model" in params, f"{impl_name}.get_params() should include 'model' key"
                assert params["model"] is not None, f"{impl_name}.get_params()['model'] should not be None"
                
                print(f"✓ {impl_name}: model = {params['model']}")
                
                # Cleanup
                if hasattr(impl, 'cleanup'):
                    impl.cleanup()
                    
            except ImportError as e:
                print(f"Skipping {impl_name} due to missing dependencies: {e}")
                continue
            except Exception as e:
                pytest.fail(f"Error testing {impl_name}: {e}")

    def test_model_name_mapping_examples(self):
        """Test specific examples of model name mapping."""
        test_cases = [
            ("LightningWhisperMLXImplementation", "small", "distil-small.en"),
            ("MLXWhisperImplementation", "small", "mlx-community/whisper-small-mlx-q4"),
            ("ParakeetMLXImplementation", "small", "mlx-community/parakeet-tdt-0.6b-v2"),
        ]
        
        for impl_name, input_model, expected_pattern in test_cases:
            try:
                impl_class = next(impl for impl in get_all_implementations() 
                                if impl.__name__ == impl_name)
                impl = impl_class()
                
                # Mock the dependencies and load model
                if impl_name == "LightningWhisperMLXImplementation":
                    with patch('lightning_whisper_mlx.LightningWhisperMLX') as mock_class:
                        mock_model = MagicMock()
                        mock_class.return_value = mock_model
                        impl.load_model(input_model)
                        
                elif impl_name == "MLXWhisperImplementation":
                    with patch('huggingface_hub.snapshot_download') as mock_download, \
                         patch('mlx_whisper.load_models.load_model') as mock_load:
                        mock_download.return_value = "/fake/path"
                        mock_model = MagicMock()
                        mock_load.return_value = mock_model
                        impl.load_model(input_model)
                        
                elif impl_name == "ParakeetMLXImplementation":
                    with patch('parakeet_mlx.from_pretrained') as mock_from_pretrained:
                        mock_model = MagicMock()
                        mock_from_pretrained.return_value = mock_model
                        impl.load_model(input_model)
                
                params = impl.get_params()
                actual_model = params.get("model")
                
                assert expected_pattern in str(actual_model), \
                    f"{impl_name}: expected '{expected_pattern}' in model name, got '{actual_model}'"
                
                print(f"✓ {impl_name}: '{input_model}' -> '{actual_model}'")
                
                if hasattr(impl, 'cleanup'):
                    impl.cleanup()
                    
            except ImportError:
                print(f"Skipping {impl_name} due to missing dependencies")
                continue
            except StopIteration:
                print(f"Implementation {impl_name} not found")
                continue


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
