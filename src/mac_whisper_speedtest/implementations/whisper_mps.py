"""Whisper MPS implementation."""

import platform
from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo
from mac_whisper_speedtest.utils import get_models_dir


class WhisperMPSImplementation(WhisperImplementation):
    """Whisper implementation using whisper-mps with Apple MPS acceleration."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.language = None
        self._model = None

        # Check if we're on macOS (MPS only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("whisper-mps is only supported on macOS with Apple Silicon")

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using whisper-mps with Apple MPS acceleration")
        self.log.info("Whisper MPS implementation")
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
            # Load the model with MPS support
            # whisper-mps automatically uses MPS when available
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
            # whisper-mps can handle numpy arrays directly and uses MPS acceleration
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
            "device": "mps",
            "language": self.language,
        }

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download."""
        from mac_whisper_speedtest.utils import get_models_dir
        from pathlib import Path

        # whisper-mps downloads to the project's models directory
        models_dir = get_models_dir()

        # whisper-mps has internal model versioning similar to other implementations
        # "large" downloads as "large-v3.pt", others download as "{model}.pt"
        # Check which file actually exists and report that one
        if model_name == "large":
            # For large, check both large-v3.pt and large.pt (in order of preference)
            large_v3_path = models_dir / "large-v3.pt"
            large_path = models_dir / "large.pt"

            # Use whichever exists, prefer large-v3.pt
            if large_v3_path.exists():
                model_file = large_v3_path
            elif large_path.exists():
                model_file = large_path
            else:
                # Neither exists, default to large-v3.pt (what will be downloaded)
                model_file = large_v3_path
        else:
            # For other models, just use the model name
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
            cache_paths=[model_file],  # Single file path (the one that exists or will be downloaded)
            expected_size_mb=size_map.get(model_name, 100),
            verification_method="size",  # Local file verification
            download_trigger="native"  # Use implementation's native download via load_model()
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # Clean up model reference
        self._model = None
        self.log.info("whisper-mps implementation cleaned up")
