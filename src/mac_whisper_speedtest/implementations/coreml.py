"""WhisperCpp with CoreML implementation."""

import os
import zipfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import requests
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir

# HuggingFace repository for CoreML encoder models
COREML_HF_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


class WhisperCppCoreMLImplementation(WhisperImplementation):
    """Whisper implementation using pywhispercpp with CoreML support."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.models_dir = get_models_dir()  # use models folder in the project root
        self.n_threads = 4
        self.coreml_enabled = False

        # Set environment variable to enable CoreML
        os.environ["WHISPER_COREML"] = "1"

    def _has_coreml_support(self) -> bool:
        """Check if pywhispercpp was built with CoreML support.

        Returns:
            True if pywhispercpp has CoreML enabled, False otherwise.
        """
        try:
            info = self._pywhispercpp.model.Model.system_info()
            return "COREML = 1" in info
        except Exception:
            return False

    def _download_coreml_model(self, model_name: str) -> Path | None:
        """Download and extract CoreML encoder model from HuggingFace.

        Args:
            model_name: CoreML model name (tiny, base, small, medium, large-v3-turbo)

        Returns:
            Path to extracted .mlmodelc directory, or None if download failed.
        """
        zip_filename = f"ggml-{model_name}-encoder.mlmodelc.zip"
        download_url = f"{COREML_HF_BASE_URL}/{zip_filename}"
        zip_path = self.models_dir / zip_filename
        mlmodelc_path = self.models_dir / f"ggml-{model_name}-encoder.mlmodelc"

        self.log.info(f"Downloading CoreML model from {download_url}")

        try:
            # Download the zip file
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                            self.log.info(f"Download progress: {pct:.1f}%")

            self.log.info(f"Downloaded {zip_filename} ({downloaded / 1024 / 1024:.1f} MB)")

            # Extract the zip file
            self.log.info(f"Extracting to {self.models_dir}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.models_dir)

            # Validate extraction
            if not mlmodelc_path.exists():
                self.log.error(f"Extraction failed: {mlmodelc_path} not found")
                return None

            # Clean up zip file
            zip_path.unlink()
            self.log.info(f"CoreML model ready at {mlmodelc_path}")

            return mlmodelc_path

        except requests.RequestException as e:
            self.log.error(f"Failed to download CoreML model: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None
        except zipfile.BadZipFile as e:
            self.log.error(f"Failed to extract CoreML model: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None
        except Exception as e:
            self.log.error(f"Unexpected error downloading CoreML model: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        import pywhispercpp.model
        self._pywhispercpp = pywhispercpp
        pywhispercpp.model.logging = self.log

        models_map = {
            "tiny": "tiny-q5_1",
            "base": "base-q5_1",
            # "small": "small-q5_1",
            "small": "small",
            "medium": "medium-q5_0",
            "large": "large-v3-turbo-q5_0",
        }
        self.model_name = models_map.get(model_name, model_name)

        # Check if CoreML is enabled
        self.coreml_enabled = os.environ.get("WHISPER_COREML", "0") == "1"
        if self.coreml_enabled:
            self.log.info("CoreML support is enabled for whisper.cpp")

            # Map model names to their corresponding CoreML model file names
            coreml_models_map = {
                "tiny": "tiny",
                "base": "base",
                "small": "small",
                "medium": "medium",
                "large": "large-v3-turbo",
            }

            # Get the CoreML model name (fallback to original model_name if not in map)
            coreml_model_name = coreml_models_map.get(model_name, model_name)

            # Check if CoreML model files exist
            coreml_model_path = self.models_dir / f"ggml-{coreml_model_name}-encoder.mlmodelc"
            if coreml_model_path.exists():
                self.log.info(f"Found CoreML model at {coreml_model_path}")
            else:
                # CoreML model not found - attempt download if pywhispercpp has CoreML support
                if self._has_coreml_support():
                    self.log.info(f"CoreML model not found at {coreml_model_path}. Attempting download...")
                    downloaded_path = self._download_coreml_model(coreml_model_name)
                    if downloaded_path and downloaded_path.exists():
                        self.log.info(f"CoreML model downloaded successfully")
                    else:
                        self.log.warning("CoreML model download failed. Will use Metal GPU fallback.")
                        self.coreml_enabled = False
                else:
                    self.log.warning(
                        "CoreML model not found and pywhispercpp lacks CoreML support. "
                        "Will use Metal GPU fallback. See docs/optimizations_2026-01-13_pywhispercpp_CoreML_Build_Guide.md"
                    )
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

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # Free the model if possible
        if hasattr(self, "_model") and hasattr(self._model, "free"):
            self._model.free()
