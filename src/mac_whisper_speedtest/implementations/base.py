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
    """Base class for Whisper implementations."""

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

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        pass
