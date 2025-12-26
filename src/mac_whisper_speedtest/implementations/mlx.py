"""MLX Whisper implementation."""

import platform
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo
from mac_whisper_speedtest.utils import get_models_dir


class MLXWhisperImplementation(WhisperImplementation):
    """Whisper implementation using MLX Whisper."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.language = None

        # Check if we're on macOS (MLX only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("MLX is only supported on macOS with Apple Silicon")

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using MLX Whisper")
        self.log.info("====== ====== ====== ====== ====== ======")
    
    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            # Import mlx-whisper package
            try:
                from huggingface_hub import snapshot_download
                from mlx_whisper.load_models import load_model
            except ImportError:
                self.log.error("Failed to import mlx-whisper. Make sure it's installed.")
                raise
        except ImportError:
            self.log.error("Failed to import MLX. Make sure it's installed.")
            raise

        self.model_name = model_name
        self.log.info(f"Loading MLX Whisper model {self.model_name}")

        # Map model name to the format expected by mlx-whisper
        # Prefer quantized models for better performance on Apple Silicon
        model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",  # No quantized version available
            "base": "mlx-community/whisper-base-mlx",  # Use base for now, fallback available
            "small": "mlx-community/whisper-small-mlx-4bit",
            "medium": "mlx-community/whisper-medium-mlx-8bit",
            "large": "mlx-community/whisper-large-v3-turbo",
            "large-v2": "mlx-community/whisper-large-v2-mlx-4bit",
            "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
        }

        # Fallback to non-quantized models if quantized versions fail
        fallback_model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-turbo",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
        }

        # Get the appropriate model path (prefer quantized)
        self.hf_repo = model_map.get(self.model_name, self.model_name)
        # Check if this is actually a quantized model by looking for quantization indicators
        self._is_quantized = any(indicator in self.hf_repo for indicator in ["4bit", "8bit", "2bit", "-q4", "-q8"])

        # Get the models directory from the utility function
        models_dir = str(get_models_dir())

        # Try to download and load the quantized model first
        try:
            self.log.info(f"Downloading quantized model {self.hf_repo} to {models_dir}")
            model_path = snapshot_download(
                repo_id=self.hf_repo,
                cache_dir=models_dir,
            )

            # Load the model
            self.log.info(f"Loading quantized model from {model_path}")
            self._model = load_model(model_path)
            self._model_path = model_path
            self.log.info(f"Successfully loaded quantized MLX Whisper model: {self.hf_repo}")

        except Exception as e:
            # Fallback to non-quantized model
            self.log.warning(f"Failed to load quantized model {self.hf_repo}: {e}")
            fallback_repo = fallback_model_map.get(self.model_name, self.model_name)
            self.log.info(f"Falling back to non-quantized model: {fallback_repo}")

            try:
                model_path = snapshot_download(
                    repo_id=fallback_repo,
                    cache_dir=models_dir,
                )

                self._model = load_model(model_path)
                self._model_path = model_path
                self.hf_repo = fallback_repo  # Update for parameter reporting
                self._is_quantized = False
                self.log.info(f"Successfully loaded fallback model: {fallback_repo}")

            except Exception as fallback_error:
                self.log.error(f"Failed to load fallback model {fallback_repo}: {fallback_error}")
                raise RuntimeError(f"Failed to load both quantized and fallback models for {self.model_name}")

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not hasattr(self, "_model_path") or self._model_path is None:
            raise RuntimeError("Model path not set. Call load_model() first.")

        self.log.info(f"Transcribing with MLX Whisper using model {self.model_name}")
        if self.language is None:
            self.log.info("Using automatic language detection")
        else:
            self.log.info(f"Using specified language: {self.language}")

        # Import here to avoid circular imports
        from mlx_whisper import transcribe

        # Run transcription directly on the audio array
        # mlx_whisper can handle numpy arrays directly
        result = transcribe(
            audio=audio,
            path_or_hf_repo=self._model_path,
            temperature=0.0,  # Use deterministic decoding
            language=self.language,  # None enables automatic language detection
            task="transcribe"  # Ensure we transcribe in original language, not translate to English
        )

        # Extract the text and ensure it's a string
        text = result.get("text", "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Extract segments and ensure it's a list
        segments = result.get("segments", [])
        if not isinstance(segments, list):
            segments = []

        # Extract language and ensure it's a string or None
        detected_language = result.get("language")
        if detected_language is not None and not isinstance(detected_language, str):
            detected_language = str(detected_language)

        if detected_language:
            self.log.info(f"Detected/used language: {detected_language}")

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=detected_language,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        params = {
            "model": getattr(self, 'hf_repo', self.model_name),
        }

        # Add quantization information
        if hasattr(self, '_is_quantized') and hasattr(self, 'hf_repo'):
            if self._is_quantized:
                # Extract quantization info from model name
                if "4bit" in self.hf_repo or "-4bit" in self.hf_repo:
                    params["quantization"] = "4bit"
                elif "8bit" in self.hf_repo or "-8bit" in self.hf_repo:
                    params["quantization"] = "8bit"
                elif "2bit" in self.hf_repo or "-2bit" in self.hf_repo:
                    params["quantization"] = "2bit"
                elif "-q4" in self.hf_repo:
                    params["quantization"] = "q4"
                else:
                    params["quantization"] = "quantized"
            else:
                params["quantization"] = "none"
        else:
            params["quantization"] = "unknown"

        return params

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download."""
        from pathlib import Path

        # Model mapping (same as in load_model)
        model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx-4bit",
            "medium": "mlx-community/whisper-medium-mlx-8bit",
            "large": "mlx-community/whisper-large-v3-turbo",
            "large-v2": "mlx-community/whisper-large-v2-mlx-4bit",
            "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
        }

        repo_id = model_map.get(model_name, model_name)

        return ModelInfo(
            model_name=repo_id,
            repo_id=repo_id,
            cache_paths=[],  # HuggingFace manages cache automatically
            expected_size_mb=None,  # Will be determined by HF verification
            verification_method="huggingface",
            download_trigger="auto"
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No explicit cleanup needed for MLX
        self._model = None
        self._model_path = None
