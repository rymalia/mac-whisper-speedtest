"""Parakeet MLX implementation."""

import platform
import tempfile
import os
from typing import Any, Dict

import numpy as np
import soundfile as sf
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir


class ParakeetMLXImplementation(WhisperImplementation):
    """Parakeet implementation using parakeet-mlx for Apple Silicon."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self._model = None
        self._hf_repo = None

        # Check if we're on macOS (MLX only works on Apple Silicon)
        if platform.system() != "Darwin":
            raise RuntimeError("Parakeet MLX is only supported on macOS with Apple Silicon")

    def load_model(self, model_name: str) -> None:
        """Load the model with the given name.

        Args:
            model_name: The name of the model to load
        """
        # Import here to avoid errors if not used
        try:
            from parakeet_mlx import from_pretrained
        except ImportError:
            self.log.error("Failed to import parakeet-mlx. Make sure it's installed.")
            raise

        self.model_name = model_name
        self.log.info(f"Loading Parakeet MLX model {self.model_name}")

        # Map model name to the format expected by parakeet-mlx
        model_map = {
            # Specific model names
            "parakeet-tdt-0.6b": "mlx-community/parakeet-tdt-0.6b-v2",
            "parakeet-tdt-0.6b-v2": "mlx-community/parakeet-tdt-0.6b-v2",
            "parakeet-tdt-1.1b": "mlx-community/parakeet-tdt-1.1b",
            "parakeet-ctc-0.6b": "mlx-community/parakeet-ctc-0.6b",
            "parakeet-ctc-1.1b": "mlx-community/parakeet-ctc-1.1b",
            # Size-based mappings (optimized for best performance using newest model)
            "tiny": "mlx-community/parakeet-tdt-0.6b-v2",
            "small": "mlx-community/parakeet-tdt-0.6b-v2",
            "base": "mlx-community/parakeet-tdt-0.6b-v2",
            "medium": "mlx-community/parakeet-tdt-0.6b-v2",
            "large": "mlx-community/parakeet-tdt-0.6b-v2",
            "large-v2": "mlx-community/parakeet-tdt-0.6b-v2",
            "large-v3": "mlx-community/parakeet-tdt-0.6b-v2",
        }

        # Get the appropriate model repo
        self._hf_repo = model_map.get(self.model_name, self.model_name)

        # If the model name doesn't start with mlx-community/, assume it's a direct HF repo
        if not self._hf_repo.startswith("mlx-community/") and "/" not in self._hf_repo:
            self._hf_repo = f"mlx-community/{self._hf_repo}"

        self.log.info(f"Loading Parakeet model from {self._hf_repo}")

        # Set HuggingFace cache directory to our models directory
        models_dir = get_models_dir()
        original_cache_dir = os.environ.get('HF_HOME')
        os.environ['HF_HOME'] = str(models_dir)

        try:
            # Load the model using parakeet-mlx
            self._model = from_pretrained(self._hf_repo)
            self.log.info(f"Successfully loaded Parakeet model: {type(self._model)}")
            self.log.info(f"Model cached in: {models_dir}")
        except Exception as e:
            self.log.error(f"Failed to load Parakeet model {self._hf_repo}: {e}")
            raise
        finally:
            # Restore original cache directory
            if original_cache_dir is not None:
                os.environ['HF_HOME'] = original_cache_dir
            else:
                os.environ.pop('HF_HOME', None)

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data.

        Args:
            audio: The audio data to transcribe (16kHz, float32)

        Returns:
            The transcription result
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.log.info(f"Transcribing with Parakeet MLX using model {self.model_name}")

        # parakeet-mlx expects file paths, so we need to save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            try:
                # Write audio data to temporary file
                # Ensure audio is in the correct format (16kHz, mono)
                sample_rate = 16000

                # Validate audio data
                if audio is None or len(audio) == 0:
                    raise ValueError("Audio data is empty or None")

                # Ensure audio is float32 and in valid range
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                # Clip audio to valid range to prevent issues
                audio = np.clip(audio, -1.0, 1.0)

                sf.write(temp_file.name, audio, sample_rate, format="WAV")

                # Verify the file was written successfully
                if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
                    raise RuntimeError(f"Failed to write audio to temporary file: {temp_file.name}")

                # Transcribe using parakeet-mlx
                result = self._model.transcribe(temp_file.name)

                # Validate result
                if result is None:
                    raise RuntimeError("Parakeet-mlx returned None result")

                # Extract text and segments from the result
                text = result.text if hasattr(result, 'text') else ""

                # Convert sentences to segments format if available
                segments = []
                if hasattr(result, 'sentences') and result.sentences:
                    for sentence in result.sentences:
                        segment = {
                            "text": sentence.text if hasattr(sentence, 'text') else str(sentence),
                            "start": sentence.start if hasattr(sentence, 'start') else 0.0,
                            "end": sentence.end if hasattr(sentence, 'end') else 0.0,
                        }
                        segments.append(segment)

                # Log transcription result for debugging
                if text:
                    self.log.info(f"Transcription completed: '{text[:50]}...' ({len(segments)} segments)")
                else:
                    self.log.warning("Parakeet-mlx returned empty transcription text")
                    # Log additional debug info when transcription is empty
                    if hasattr(result, 'sentences'):
                        self.log.info(f"Result has {len(result.sentences) if result.sentences else 0} sentences")

                # Parakeet TDT models are primarily English-focused
                # However, they may attempt to transcribe other languages
                detected_language = "en"  # Default to English as these models are English-trained

                # Simple heuristic: if the transcription contains non-English patterns,
                # we might be dealing with non-English input that was force-transcribed
                if text and self._contains_non_english_patterns(text):
                    detected_language = None  # Unknown/uncertain language
                    self.log.info("Detected possible non-English input - Parakeet TDT models are optimized for English")

                return TranscriptionResult(
                    text=text,
                    segments=segments,
                    language=detected_language,
                )
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass  # File might already be deleted

    def _contains_non_english_patterns(self, text: str) -> bool:
        """Simple heuristic to detect if text might contain non-English content."""
        if not text:
            return False

        # Check for common German words/patterns that might indicate non-English input
        german_indicators = [
            'und', 'der', 'die', 'das', 'ist', 'ein', 'eine', 'mit', 'von', 'zu',
            'auf', 'für', 'als', 'bei', 'nach', 'über', 'durch', 'um', 'vor',
            'ß', 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü'  # German special characters
        ]

        text_lower = text.lower()

        # Check for German indicators
        german_word_count = sum(1 for indicator in german_indicators if indicator in text_lower)
        if german_word_count > 2:  # Multiple German words detected
            return True

        # Check for patterns that suggest the model struggled with non-English input
        words = text.split()
        if len(words) > 0:
            short_word_ratio = len([word for word in words if len(word) < 3]) / len(words)
            if short_word_ratio > 0.4:  # Too many short words might indicate struggle
                return True

        return False

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self._hf_repo,
            "implementation": "parakeet-mlx",
            "platform": "Apple Silicon (MLX)",
        }

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        self._model = None
        self._hf_repo = None
