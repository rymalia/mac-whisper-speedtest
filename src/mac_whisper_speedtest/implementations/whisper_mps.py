"""Whisper MPS implementation."""

import platform
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo
from mac_whisper_speedtest.utils import get_models_dir


class WhisperMPSImplementation(WhisperImplementation):
    """Whisper implementation using whisper-mps with Apple MLX acceleration.

    Note: Despite the library name 'whisper-mps', this implementation uses Apple's
    MLX framework exclusively - NOT Metal Performance Shaders (MPS). The library
    downloads PyTorch model files and converts them to MLX format at load time.
    """

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.language = "en"  # Default to English; can be set to None for auto-detection
        self._model = None

        # Check if we're on macOS (whisper-mps/MLX only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("whisper-mps is only supported on macOS with Apple Silicon")

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using whisper-mps with Apple MLX acceleration")
        self.log.info("Whisper MPS implementation (uses MLX backend)")
        self.log.info("====== ====== ====== ====== ====== ======")
    
    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            from whisper_mps.whisper.load_models import load_model, available_models
        except ImportError:
            self.log.error("Failed to import whisper-mps. Make sure it's installed.")
            raise

        self.model_name = model_name
        self.log.info(f"Loading whisper-mps model {self.model_name}")

        # Check if model name is valid
        available = available_models()
        if model_name not in available:
            self.log.warning(f"Model {model_name} not in available models: {available}")
            # Try to proceed anyway, whisper-mps might handle it

        # Get models directory in project root
        models_dir = get_models_dir()
        self.log.info(f"Using models directory: {models_dir}")

        try:
            # Load the model with MLX support
            # whisper-mps converts PyTorch models to MLX format
            self._model = load_model(
                name=model_name,
                download_root=str(models_dir)
            )
            self.log.info(f"Successfully loaded whisper-mps model: {model_name}")
        except Exception as e:
            self.log.error(f"Failed to load whisper-mps model {model_name}: {e}")
            raise

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with whisper-mps using model {self.model_name}")
        if self.language is None:
            self.log.info("Using automatic language detection")
        else:
            self.log.info(f"Using specified language: {self.language}")

        try:
            # Import here to avoid circular imports
            from whisper_mps.whisper.transcribe import transcribe

            # Run transcription directly on the audio array
            # whisper-mps can handle numpy arrays directly and uses MLX acceleration
            result = transcribe(
                audio=audio,
                model=self.model_name,  # Use model name directly
                temperature=0.0,  # Use deterministic decoding
                language=self.language,  # None enables automatic language detection
                verbose=False,  # Disable verbose output
                condition_on_previous_text=True,  # Enable context awareness
            )

            # Extract the text and ensure it's a string
            text = result.get("text", "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            # Extract segments and ensure it's a list
            segments = result.get("segments", [])
            if not isinstance(segments, list):
                segments = []

            # Extract language
            language = result.get("language", None)

            self.log.info(f"Transcription completed. Text length: {len(text)} characters")

            return TranscriptionResult(
                text=text.strip(),
                segments=segments,
                language=language,
            )

        except Exception as e:
            self.log.error(f"Transcription failed with whisper-mps: {e}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self.model_name,
            "backend": "whisper-mps",
            "device": "mlx",  # Note: despite library name, uses MLX not MPS
            "language": self.language,
        }

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download.

        Consistent with load_model() - uses the model name directly as whisper-mps
        handles internal version mapping (e.g., "large" may map to "large-v3.pt").
        """
        from mac_whisper_speedtest.utils import get_models_dir

        # whisper-mps downloads to the project's models directory
        models_dir = get_models_dir()

        # Use model name directly - whisper-mps handles versioning internally
        # This matches what load_model() does (passes model_name directly to library)
        model_file = models_dir / f"{model_name}.pt"

        # Expected sizes (approximate, in MB)
        size_map = {
            "tiny": 75,
            "base": 145,
            "small": 470,
            "medium": 1500,
            "large": 2900,
        }

        return ModelInfo(
            model_name=model_name,
            repo_id=None,  # whisper-mps downloads from openaipublic.azureedge.net, not HuggingFace
            cache_paths=[model_file],
            expected_size_mb=size_map.get(model_name, 100),
            verification_method="size",  # Local file verification
            download_trigger="native",  # Use implementation's native download via load_model()
            timeout_seconds=30 if "large" in model_name else 15
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # Clean up model reference
        self._model = None
        self.log.info("whisper-mps implementation cleaned up")
