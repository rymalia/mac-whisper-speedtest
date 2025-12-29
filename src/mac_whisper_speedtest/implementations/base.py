"""Base class for Whisper implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np


@dataclass
class ModelInfo:
    """Information about a model's requirements and cache locations."""
    model_name: str  # The actual model identifier (e.g., "whisper-medium-mlx-4bit")
    repo_id: Optional[str] = None  # HuggingFace repo ID if applicable
    cache_paths: List[Path] = field(default_factory=list)  # Expected cache locations
    expected_size_mb: Optional[int] = None  # Expected total size in MB
    verification_method: str = "huggingface"  # "huggingface", "size", or "structure"
    download_trigger: str = "auto"  # "auto" (HF download), "bridge" (needs Swift bridge), or "manual"
    hf_cache_dir: Optional[str] = None  # Specific HF cache directory to check (None = default HF cache)


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    segments: List = field(default_factory=list)
    language: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    implementation: str
    model_name: str
    model_params: Dict[str, Any]
    transcription_time: float
    text: str = ""


class WhisperImplementation(ABC):
    """Base class for Whisper implementations.

    Model Mapping Convention
    ------------------------
    To ensure consistency between load_model() and get_model_info(), implementations
    should use the standardized helper method pattern:

    1. Override _get_model_map() to provide model name mappings
    2. Call _map_model_name() in both load_model() and get_model_info()
    3. This ensures a single source of truth and eliminates duplication

    Example:
        class MyImplementation(WhisperImplementation):
            def _get_model_map(self) -> Dict[str, str]:
                return {
                    "tiny": "my-repo/whisper-tiny-quantized",
                    "small": "my-repo/whisper-small-quantized",
                    "large": "my-repo/whisper-large-v3",
                }

            def load_model(self, model_name: str) -> None:
                repo_id = self._map_model_name(model_name)
                # Use repo_id to load model...

            def get_model_info(self, model_name: str) -> ModelInfo:
                repo_id = self._map_model_name(model_name)
                return ModelInfo(repo_id=repo_id, ...)

    State Management Convention
    ---------------------------
    - self.model_name: Store the ORIGINAL requested model name
    - self._model: The loaded model instance
    - get_params() should return the ACTUAL loaded model name, not just requested
    """

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load the model with the given name."""
        pass

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {}

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about model requirements for verification/download.

        Args:
            model_name: The model size (tiny, base, small, medium, large)

        Returns:
            ModelInfo with cache locations and verification details
        """
        # Default implementation - subclasses should override
        return ModelInfo(
            model_name=model_name,
            verification_method="none"
        )

    # Model Mapping Helpers (Optional - Override in Subclasses)

    def _get_model_map(self) -> Dict[str, str]:
        """Override in subclass to provide model name mappings.

        This method should return a dictionary mapping standard Whisper model names
        (tiny, base, small, medium, large) to implementation-specific model identifiers
        or HuggingFace repo IDs.

        Returns:
            Dict mapping standard names to implementation-specific names.
            Empty dict by default (no mapping needed).

        Example:
            return {
                "tiny": "mlx-community/whisper-tiny-mlx-q4",
                "small": "mlx-community/whisper-small-mlx-4bit",
                "large": "mlx-community/whisper-large-v3-mlx",
            }
        """
        return {}

    def _map_model_name(self, model_name: str) -> str:
        """Map a standard model name to implementation-specific name.

        This method uses _get_model_map() to perform the mapping. Subclasses should
        call this method in both load_model() and get_model_info() to ensure
        consistency.

        Args:
            model_name: Standard model name (tiny, base, small, medium, large, etc.)

        Returns:
            Implementation-specific model name or repo ID
        """
        model_map = self._get_model_map()
        return model_map.get(model_name, model_name)

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        pass
