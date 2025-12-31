"""Lightning Whisper MLX implementation."""

import platform
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo


class LightningWhisperMLXImplementation(WhisperImplementation):
    """Whisper implementation using Lightning Whisper MLX."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.whisper_model = None
        # Apple Silicon optimizations:
        # - batch_size=12: Recommended default for optimal memory usage on unified memory architecture
        # - quant=None: Disable quantization to avoid compatibility issues
        self.batch_size = 12
        self.quant = None
        self.language = None  # Language code for transcription

        # Check if we're on macOS (MLX only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("LightningWhisperMLX is only supported on macOS with Apple Silicon")

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using Lightning Whisper MLX")
        self.log.info("====== ====== ====== ====== ====== ======")
    
    def _get_model_map(self) -> Dict[str, str]:
        """Model name mappings for Lightning Whisper MLX.

        Maps standard Whisper model names to quantized MLX community models.
        Uses base class standardized pattern for consistency.

        Note: LightningWhisperMLX manages its own HuggingFace downloads internally.
        These repos are used by check-models for verification only.
        """
        return {
            "tiny": "mlx-community/whisper-tiny-mlx-q4",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx-4bit",
            "medium": "mlx-community/whisper-medium-mlx-8bit",
            "large": "mlx-community/whisper-large-v3-turbo",
            "large-v2": "mlx-community/whisper-large-v2-mlx-4bit",
            "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
        }

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load (e.g., "small", "base", "medium", "large")
        """
        self.log.info(f"Loading LightningWhisperMLX model {model_name}")

        # Import Lightning Whisper MLX 
        # (.venv/lib/python3.12/site-packages/lightning_whisper_mlx/lightning.py)
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
        except ImportError:
            self.log.error("Failed to import lightning_whisper_mlx. Make sure it's installed.")
            raise

        # LightningWhisperMLX expects simple model names and handles HuggingFace downloads internally
        # Map "large" to "large-v3" for consistency with other implementations
        model_for_loading = "large-v3" if model_name == "large" else model_name

        # Store original model name
        self.model_name = model_name
        self._model_for_loading = model_for_loading

        # Create the Lightning Whisper MLX instance with default model management
        # Note: LightningWhisperMLX downloads from its own preferred repos
        try:
            self.whisper_model = LightningWhisperMLX(
                model=model_for_loading,
                batch_size=self.batch_size,
                quant=self.quant
            )
            self.log.info(f"LightningWhisperMLX model {model_name} loaded successfully (using {model_for_loading})")
        except Exception as e:
            self.log.error(f"Failed to load model {model_name}: {e}")
            raise





    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "whisper_model") or self.whisper_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with LightningWhisperMLX using model {self.model_name}")
        if self.language is None:
            self.log.info("Using automatic language detection")
        else:
            self.log.info(f"Using specified language: {self.language}")

        try:
            # Lightning Whisper MLX expects audio file path, so we need to save the numpy array temporarily
            import tempfile
            import soundfile as sf

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                # Write the numpy array as a WAV file
                sf.write(temp_path, audio, 16000)  # Whisper expects 16kHz sample rate

            try:
                # Use Lightning Whisper MLX transcribe method
                result = self.whisper_model.transcribe(
                    audio_path=temp_path,
                    language=self.language  # None enables automatic language detection
                )
                self.log.info("Transcription completed using Lightning Whisper MLX")
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors

        except Exception as e:
            self.log.error(f"Lightning Whisper MLX transcription failed: {e}")
            raise RuntimeError(f"Lightning Whisper MLX transcription failed: {e}") from e

        # Extract the text and ensure it's a string
        text = result.get("text", "") if result else ""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Extract segments and ensure it's a list
        segments = result.get("segments", []) if result else []
        if not isinstance(segments, list):
            segments = []

        # Extract language and ensure it's a string or None
        detected_language = result.get("language") if result else None
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
            "model": self.model_name,  # Show the actual model being used
            "batch_size": self.batch_size,
            "quant": self.quant or "none",
        }

        if self.language:
            params["language"] = self.language

        return params

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download.

        Uses the same model mapping as load_model() to ensure consistency
        between verification and actual model loading.
        """
        # Use base class helper (same as load_model) - single source of truth
        repo_id = self._map_model_name(model_name)

        return ModelInfo(
            model_name=repo_id,
            repo_id=repo_id,
            cache_paths=[],
            expected_size_mb=None,
            verification_method="huggingface",
            download_trigger="auto",
            timeout_seconds=30 if "large" in model_name else 15
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # MLX models don't need explicit cleanup
        if hasattr(self, "whisper_model"):
            self.whisper_model = None
