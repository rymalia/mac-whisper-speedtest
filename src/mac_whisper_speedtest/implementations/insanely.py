"""Insanely Fast Whisper implementation."""

import platform
import tempfile
import psutil
from typing import Any, Dict

import numpy as np
import soundfile as sf
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo


class InsanelyFastWhisperImplementation(WhisperImplementation):
    """Whisper implementation using Insanely Fast Whisper."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self.device_id = "mps" if platform.system() == "Darwin" else "cpu"
        # Apple Silicon optimizations:
        # - Adaptive batch sizing based on available memory
        # - compute_type="float16": Optimal for MPS device acceleration
        # - quantization="4bit": Enable 4-bit quantization for better memory efficiency
        self.batch_size = self._get_optimal_batch_size()
        self.compute_type = "float16"
        self.quantization = "4bit"  # Enable 4-bit quantization by default

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: Whisper implementation using Insanely Fast Whisper")
        self.log.info("====== ====== ====== ====== ====== ======")
    
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available system memory.

        Returns:
            Optimal batch size for the current system
        """
        try:
            # Get available memory in GB
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            if platform.system() == "Darwin" and self.device_id == "mps":
                # Apple Silicon unified memory optimization
                # More aggressive batch sizing for better performance
                if available_memory_gb >= 32:
                    batch_size = 16  # High-end Apple Silicon (M2/M3 Ultra, M4 Pro/Max)
                elif available_memory_gb >= 16:
                    batch_size = 12  # Mid-range Apple Silicon (M2/M3 Pro, M4)
                elif available_memory_gb >= 8:
                    batch_size = 10  # Base Apple Silicon (M1/M2/M3 base)
                elif available_memory_gb >= 4:
                    batch_size = 16  # More aggressive for Apple Silicon unified memory
                else:
                    batch_size = 12  # Still reasonable for low memory Apple Silicon

                self.log.info(f"Apple Silicon detected: {available_memory_gb:.1f}GB available, using batch_size={batch_size}")
            else:
                # Non-Apple Silicon systems
                if available_memory_gb >= 16:
                    batch_size = 24  # Original default for high-memory systems
                elif available_memory_gb >= 8:
                    batch_size = 16
                else:
                    batch_size = 8

                self.log.info(f"Non-Apple Silicon: {available_memory_gb:.1f}GB available, using batch_size={batch_size}")

            return batch_size

        except Exception as e:
            self.log.warning(f"Failed to determine optimal batch size: {e}, using default=12")
            return 12  # Safe default

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            import torch
            from transformers.pipelines import pipeline
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                # Fallback for older transformers versions
                BitsAndBytesConfig = None
            # Try to import flash attention check, with fallback
            try:
                from transformers.utils.import_utils import is_flash_attn_2_available
            except ImportError:
                try:
                    from transformers.utils import is_flash_attn_2_available
                except ImportError:
                    # If not available, define a fallback
                    is_flash_attn_2_available = lambda: False
        except ImportError:
            self.log.error("Failed to import required packages. Make sure transformers is installed.")
            raise

        self.model_name = self._map_model_name(model_name)
        self.log.info(f"Loading Insanely Fast Whisper model {self.model_name}")

        # Load the model using transformers pipeline
        # Apple Silicon optimization: Prefer SDPA over flash_attention_2 for better MPS performance
        # SDPA (Scaled Dot Product Attention) is more optimized for Apple Silicon MPS backend
        if platform.system() == "Darwin" and self.device_id == "mps":
            attn_implementation = "sdpa"
            self.log.info("Using SDPA attention implementation (optimized for Apple Silicon MPS)")
        else:
            attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            self.log.info(f"Using attention implementation: {attn_implementation}")

        # Additional Apple Silicon MPS optimizations
        model_kwargs: Dict[str, Any] = {"attn_implementation": attn_implementation}

        # Apple Silicon specific optimizations
        if platform.system() == "Darwin" and self.device_id == "mps":
            # Enable optimized memory layout for Apple Silicon
            model_kwargs["use_cache"] = True  # Enable KV cache for better performance
            model_kwargs["low_cpu_mem_usage"] = True  # Optimize for unified memory architecture

        # Configure quantization if available and enabled
        # Note: bitsandbytes is not supported on macOS, so 4-bit quantization will be skipped on Apple Silicon
        if self.quantization == "4bit" and BitsAndBytesConfig is not None:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                self.log.info("Using 4-bit quantization with BitsAndBytesConfig")
            except Exception as e:
                self.log.warning(f"Failed to configure 4-bit quantization: {e}. "
                               f"Note: bitsandbytes is not supported on macOS/Apple Silicon.")

        self._model = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch.float16,
            device=self.device_id,
            model_kwargs=model_kwargs,
        )

    def _get_model_map(self) -> Dict[str, str]:
        """Model name mappings for Insanely Fast Whisper.

        Maps standard Whisper model names to OpenAI HuggingFace repo IDs.
        Uses base class standardized pattern for consistency.
        """
        return {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3-turbo",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe

        Returns:
            The transcription result
        """
        if not hasattr(self, "_model") or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with Insanely Fast Whisper using model {self.model_name}")
        self.log.info("Using automatic language detection")

        # Convert numpy array to a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Save the audio data to a temporary file
            sf.write(temp_file.name, audio, 16000, format="WAV")

            # Run transcription using transformers pipeline with language detection
            # Apple Silicon optimization: Reduce chunk length for better memory efficiency
            chunk_length = 20 if (platform.system() == "Darwin" and self.device_id == "mps") else 30

            result = self._model(
                temp_file.name,
                chunk_length_s=chunk_length,
                batch_size=self.batch_size,
                return_timestamps=False,
                return_language=True,  # Enable language detection
                generate_kwargs={"task": "transcribe"}  # Ensure we transcribe in original language, not translate to English
            )

        # Handle different result formats from transformers pipeline
        text = ""
        detected_language = None
        segments = []

        if isinstance(result, dict):
            # Single result format
            text = result.get("text", "")
            detected_language = result.get("language")
            if "chunks" in result:
                segments = result["chunks"]
            elif "segments" in result:
                segments = result["segments"]
        elif isinstance(result, list) and len(result) > 0:
            # List of results format
            first_result = result[0]
            if isinstance(first_result, dict):
                text = first_result.get("text", "")
                detected_language = first_result.get("language")
                # Combine all text from list
                text = " ".join([r.get("text", "") for r in result if isinstance(r, dict)])
                segments = result
        else:
            # Fallback for unexpected formats
            self.log.warning(f"Unexpected result format: {type(result)}")
            text = str(result) if result else ""

        if detected_language:
            self.log.info(f"Detected language: {detected_language}")

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=detected_language,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self.model_name,
            "device_id": self.device_id,
            "batch_size": self.batch_size,
            "compute_type": self.compute_type,
            "quantization": self.quantization,
        }

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download."""
        # Use the same model mapping as load_model() to ensure consistency
        repo_id = self._map_model_name(model_name)

        return ModelInfo(
            model_name=model_name,
            repo_id=repo_id,
            cache_paths=[],
            expected_size_mb=None,  # Will be determined by HF verification
            verification_method="huggingface",
            download_trigger="auto",
            timeout_seconds=30 if "large" in model_name else 15
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No explicit cleanup needed
        self._model = None
