"""WhisperCpp with CoreML implementation."""

import os

# Set environment variable to enable CoreML
os.environ["WHISPER_COREML"] = "1"

from typing import Any, Dict

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo
from mac_whisper_speedtest.utils import get_models_dir


class WhisperCppCoreMLImplementation(WhisperImplementation):
    """Whisper implementation using pywhispercpp with CoreML support."""
    
    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.models_dir = get_models_dir()  # use models folder in the project root
        self.n_threads = 4 # 4
        self.coreml_enabled = False

        # Set environment variable to enable CoreML
        # os.environ["WHISPER_COREML"] = "1"

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using pywhispercpp with CoreML support")
        self.log.info("WhisperCpp with CoreML implementation")
        self.log.info("====== ====== ====== ====== ====== ======")

    def _get_model_map(self) -> Dict[str, str]:
        """Model name mappings for whisper.cpp (GGML quantized versions).

        Maps standard Whisper model names to quantized GGML model names.
        Uses base class standardized pattern for consistency.
        """
        return {
            "tiny": "tiny-q5_1",
            "base": "base-q5_1",
            "small": "small",
            "medium": "medium-q5_0",
            "large": "large-v3-turbo-q5_0",
        }

    def _get_coreml_model_map(self) -> Dict[str, str]:
        """Model name mappings for CoreML encoder models.

        Maps standard Whisper model names to CoreML encoder model names.
        This is whisper.cpp-specific logic for CoreML acceleration.
        """
        return {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large-v3-turbo",
        }

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        import pywhispercpp.model
        self._pywhispercpp = pywhispercpp
        pywhispercpp.model.logging = self.log

        # Use base class helper to map to quantized GGML model name
        self.model_name = self._map_model_name(model_name)

        # Check if CoreML is enabled
        self.coreml_enabled = os.environ.get("WHISPER_COREML", "0") == "1"
        if self.coreml_enabled:
            self.log.info("CoreML support is enabled for whisper.cpp")

            # Get CoreML encoder model name using helper method
            coreml_model_map = self._get_coreml_model_map()
            coreml_model_name = coreml_model_map.get(model_name, model_name)

            # Check if CoreML model files exist
            coreml_model_path = self.models_dir / f"ggml-{coreml_model_name}-encoder.mlmodelc"
            if coreml_model_path.exists():
                self.log.info(f"Found CoreML model at {coreml_model_path}")
            else:
                self.log.warning(f"CoreML model not found at {coreml_model_path}. Will use CPU fallback.")
                self.coreml_enabled = False

        # Load the model
        self._model = self._pywhispercpp.model.Model(
            self.model_name,
            models_dir=str(self.models_dir),
            n_threads=self.n_threads,
        )

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with whisper.cpp using model {self.model_name}")
        if self.coreml_enabled:
            self.log.info("Using CoreML acceleration")



        # Transcribe with automatic language detection
        # Explicitly set translate=False to ensure transcription in original language, not translation to English
        segments = self._model.transcribe(audio, language=None, translate=False)
        text = " ".join([segment.text for segment in segments])

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=None,  # Language detection is handled natively by the transcribe method
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self.model_name,
            "coreml": self.coreml_enabled,
            "n_threads": self.n_threads,
        }

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download.

        Uses base class helpers for model mapping to ensure consistency
        between verification and actual model loading.
        """
        from pathlib import Path
        from mac_whisper_speedtest.utils import get_models_dir

        # Use base class helper for GGML model name (single source of truth)
        ggml_model_name = self._map_model_name(model_name)

        # Use CoreML helper for CoreML encoder name
        coreml_model_map = self._get_coreml_model_map()
        coreml_model_name = coreml_model_map.get(model_name, model_name)

        models_dir = Path(get_models_dir())

        # Cache paths include GGML model + optional CoreML encoder
        cache_paths = [
            models_dir / f"ggml-{ggml_model_name}.bin",
            models_dir / f"ggml-{coreml_model_name}-encoder.mlmodelc",
        ]

        # Expected sizes (GGML + CoreML encoder combined, approximate in MB)
        size_map = {
            "tiny": 50,       # ~31MB GGML (q5_1) + ~16MB CoreML
            "base": 250,      # ~142MB GGML + ~108MB CoreML
            "small": 700,     # ~466MB GGML + ~234MB CoreML
            "medium": 2200,   # ~1500MB GGML + ~700MB CoreML
            "large": 1800,    # ~547MB GGML (turbo q5_0) + ~1200MB CoreML (turbo encoder)
        }

        return ModelInfo(
            model_name=f"{ggml_model_name} + CoreML",
            repo_id=None,
            cache_paths=cache_paths,
            expected_size_mb=size_map.get(model_name, 100),
            verification_method="size",
            download_trigger="manual"
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # Free the model if possible
        if hasattr(self, "_model") and hasattr(self._model, "free"):
            self._model.free()
