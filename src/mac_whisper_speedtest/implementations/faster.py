"""Faster Whisper implementation."""

from typing import Any, Dict, List

import numpy as np
import structlog
import subprocess
import platform

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo
from mac_whisper_speedtest.utils import get_models_dir


class FasterWhisperImplementation(WhisperImplementation):
    """Whisper implementation using Faster Whisper.

    APPLE SILICON LIMITATION:
    faster-whisper does NOT support GPU acceleration on Apple Silicon.
    It only supports CPU processing via Apple Accelerate framework.
    For maximum Apple Silicon performance, consider alternatives like:
    - whisper.cpp with CoreML support
    - MLX-based implementations (mlx-whisper, lightning-whisper-mlx)
    """

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        # Apple Silicon optimizations:
        # - device="cpu": Only option available (faster-whisper doesn't support MPS/GPU on Apple Silicon)
        # - compute_type: Optimized for Apple Silicon CPU architecture
        # - cpu_threads: Dynamically optimized based on Apple Silicon chip architecture
        self.device = "cpu"
        self.compute_type = "int8"
        # Apple Silicon optimization: beam_size=1 for fastest inference
        # Higher beam sizes improve quality but significantly slow down inference
        self.beam_size = 1
        self.language = None
        self.cpu_threads = self._get_optimal_cpu_threads()

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using Faster Whisper")
        self.log.info("====== ====== ====== ====== ====== ======")

    def _get_model_map(self) -> Dict[str, str]:
        """Model name mappings for Faster Whisper.

        Maps standard Whisper model names to HuggingFace repo IDs.
        Uses base class standardized pattern for consistency.

        Note: For "large" model, this returns the primary model (large-v3-turbo).
        The fallback chain is handled separately in _get_model_fallback_chain().
        """
        return {
            "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # Primary for "large"
            "medium": "Systran/faster-whisper-medium",
            "small": "Systran/faster-whisper-small",
            "base": "Systran/faster-whisper-base",
            "tiny": "Systran/faster-whisper-tiny",
        }

    def _get_model_fallback_chain(self, model_name: str) -> List[str]:
        """Get the fallback chain for a given model name.

        For 'large' model, returns: [large-v3-turbo, large-v3, large]
        For other models, returns: [model_name]

        This is implementation-specific logic for Faster Whisper's fallback mechanism.

        Args:
            model_name: The originally requested model name

        Returns:
            List of model names to try in order
        """
        if model_name == "large":
            return ["large-v3-turbo", "large-v3", "large"]

        # No fallback chain for other models
        return [model_name]

    def _get_optimal_cpu_threads(self) -> int:
        """Calculate optimal CPU thread count for Apple Silicon.

        Returns:
            Optimal number of CPU threads for the current system
        """
        try:
            if platform.system() == "Darwin":
                # Get Apple Silicon chip information
                try:
                    # Get total cores and performance cores
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    total_cores = 0
                    perf_cores = 0

                    for line in result.stdout.split('\n'):
                        if "Total Number of Cores:" in line:
                            # Extract total cores and performance cores
                            # Format: "Total Number of Cores: 14 (10 performance and 4 efficiency)"
                            parts = line.split(':')[1].strip()
                            total_cores = int(parts.split()[0])
                            if '(' in parts and 'performance' in parts:
                                perf_part = parts.split('(')[1].split('performance')[0].strip()
                                perf_cores = int(perf_part)
                            break

                    if perf_cores > 0:
                        # Use performance cores + 2 efficiency cores for optimal performance
                        # This balances performance with system responsiveness
                        optimal_threads = min(perf_cores + 2, total_cores)
                        self.log.info(f"Apple Silicon detected: {total_cores} total cores ({perf_cores} performance), using {optimal_threads} threads")
                        return optimal_threads
                    elif total_cores > 0:
                        # Fallback: use 75% of total cores
                        optimal_threads = max(4, int(total_cores * 0.75))
                        self.log.info(f"Apple Silicon detected: {total_cores} total cores, using {optimal_threads} threads")
                        return optimal_threads

                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
                    self.log.warning(f"Failed to detect Apple Silicon configuration: {e}")

                # Fallback for Apple Silicon
                try:
                    import os
                    cpu_count = os.cpu_count() or 8 # My MacBook Air M3 has 8 cores total (4 performance and 4 efficiency)
                    optimal_threads = max(4, min(12, cpu_count - 2))  # Leave 2 cores for system
                    self.log.info(f"Apple Silicon fallback: {cpu_count} CPUs detected, using {optimal_threads} threads")
                    return optimal_threads
                except Exception:
                    pass

            # Non-Apple Silicon fallback
            self.log.info("Non-Apple Silicon system, using default 8 threads")
            return 8

        except Exception as e:
            self.log.warning(f"Failed to determine optimal CPU threads: {e}, using default=8")
            return 8

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            from faster_whisper import WhisperModel
            self._faster_whisper = WhisperModel
        except ImportError:
            self.log.error("Failed to import faster_whisper. Make sure it's installed.")
            raise

        # Get fallback chain for the requested model
        self.original_model_name = model_name  # Keep track of original request
        model_chain = self._get_model_fallback_chain(model_name)

        # Log the fallback strategy
        if len(model_chain) > 1:
            self.log.info(f"Model fallback chain for '{model_name}': {' → '.join(model_chain)}")

        # Get models directory in project root
        models_dir = get_models_dir()
        self.log.info(f"Using models directory: {models_dir}")

        # Try each model in the fallback chain
        last_error = None
        for i, model_to_try in enumerate(model_chain):
            try:
                # Log the attempt
                if i == 0 and model_to_try != model_name:
                    self.log.info(f"Attempting to load preferred model '{model_to_try}' (requested: '{model_name}')")
                elif i > 0:
                    self.log.info(f"Falling back to model '{model_to_try}' (attempt {i + 1}/{len(model_chain)})")
                else:
                    self.log.info(f"Loading Faster Whisper model '{model_to_try}'")

                # Load the model
                self._model = self._faster_whisper(
                    model_size_or_path=model_to_try,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(models_dir),  # Use models directory in project root
                    cpu_threads=self.cpu_threads,
                )

                # Success! Update model_name and break
                self.model_name = model_to_try
                if model_to_try != model_name:
                    self.log.info(f"Successfully loaded model '{model_to_try}' (substituted from '{model_name}')")
                else:
                    self.log.info(f"Successfully loaded model '{model_to_try}'")
                break

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                self.log.warning(f"Failed to load model '{model_to_try}': {error_type}: {e}")

                if i == len(model_chain) - 1:
                    # This was the last attempt, re-raise the error
                    self.log.error(f"All model loading attempts failed for requested model '{model_name}'")
                    self.log.error(f"Tried models: {', '.join(model_chain)}")
                    self.log.error(f"Final error: {error_type}: {e}")
                    raise RuntimeError(f"Failed to load any model variant for '{model_name}'. Last error: {e}") from e
                else:
                    # Continue to next model in chain
                    continue

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with Faster Whisper using model {self.model_name}")

        # Transcribe
        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Convert segments to list (it's a generator)
        segments_list = list(segments)
        text = " ".join([segment.text for segment in segments_list])

        return TranscriptionResult(
            text=text,
            segments=segments_list,
            language=info.language,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        params = {
            "model": self.model_name,  # Actual model loaded (may be substituted)
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "cpu_threads": self.cpu_threads,
        }

        # Include original requested model if different from actual model
        if hasattr(self, 'original_model_name') and self.original_model_name != self.model_name:
            params["original_model_requested"] = self.original_model_name

        return params

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download.

        Uses base class helper for model mapping to ensure consistency
        between verification and actual model loading.
        """
        from mac_whisper_speedtest.utils import get_models_dir

        # Get primary model from fallback chain (for display purposes)
        fallback_chain = self._get_model_fallback_chain(model_name)
        primary_model = fallback_chain[0]

        # Use base class helper to map to HuggingFace repo ID (single source of truth)
        repo_id = self._map_model_name(primary_model)

        return ModelInfo(
            model_name=primary_model,
            repo_id=repo_id,
            cache_paths=[],  # HuggingFace manages cache automatically
            expected_size_mb=None,  # Will be determined by HF verification
            verification_method="huggingface",
            download_trigger="auto",
            hf_cache_dir=str(get_models_dir())  # faster-whisper uses custom download_root
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No explicit cleanup needed for Faster Whisper
        self._model = None
