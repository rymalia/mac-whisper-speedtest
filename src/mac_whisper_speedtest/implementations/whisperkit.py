"""WhisperKit implementation using subprocess bridge to Swift framework."""

import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation
from mac_whisper_speedtest.utils import get_models_dir


class WhisperKitImplementation(WhisperImplementation):
    """WhisperKit implementation using Swift bridge to native WhisperKit framework."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self._bridge_path = None

        # Check if we're on macOS (WhisperKit only works on Apple platforms)
        if platform.system() != "Darwin":
            raise RuntimeError("WhisperKit is only supported on macOS")

        # Find the Swift bridge executable
        self._find_bridge_executable()

    def _find_bridge_executable(self):
        """Find the WhisperKit Swift bridge executable."""
        # Look for the bridge in the tools directory
        project_root = Path(__file__).parent.parent.parent.parent
        bridge_path = project_root / "tools" / "whisperkit-bridge" / ".build" / "release" / "whisperkit-bridge"

        if bridge_path.exists():
            self._bridge_path = str(bridge_path)
            self.log.info(f"Found WhisperKit bridge at: {self._bridge_path}")
        else:
            raise RuntimeError(f"WhisperKit bridge not found at {bridge_path}. "
                             f"Please build it first by running: cd tools/whisperkit-bridge && swift build -c release")

    def load_model(self, model_name: str) -> None:
        """Load the WhisperKit model (via Swift bridge).

        Args:
            model_name: Model size (tiny, base, small, medium, large)
        """
        if not self._bridge_path:
            raise RuntimeError("WhisperKit bridge not available")

        # Map model names to WhisperKit model names
        # Note: Turbo models are available but may have compatibility issues.
        # Use explicit model names like "large-v3-turbo" to access turbo variants.
        model_map = {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large-v3",  # Standard large-v3 for compatibility
            "large-v3": "large-v3",
            "large-v3-turbo": "large-v3-turbo",  # Explicit turbo model access
            "large-turbo": "large-v3-turbo"     # Alternative turbo access
        }
        
        self.model_name = model_map.get(model_name, model_name)

        # Log model information
        if model_name == "large":
            self.log.info(f"Using WhisperKit large-v3 model (requested: {model_name}, using: {self.model_name})")
            self.log.info("Note: For turbo performance, use 'large-v3-turbo' or 'large-turbo' explicitly")
        elif "turbo" in self.model_name:
            self.log.info(f"Using WhisperKit turbo model (requested: {model_name}, using: {self.model_name})")
        else:
            self.log.info(f"WhisperKit bridge ready for model: {self.model_name}")

        # Test that the bridge is working by running it with --help
        try:
            result = subprocess.run(
                [self._bridge_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.log.info("WhisperKit bridge is working correctly")
            else:
                raise RuntimeError(f"WhisperKit bridge test failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("WhisperKit bridge test timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to test WhisperKit bridge: {e}")

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data using WhisperKit Swift bridge.

        Args:
            audio: The audio data to transcribe (16kHz mono)

        Returns:
            The transcription result
        """
        if not self._bridge_path:
            raise RuntimeError("WhisperKit bridge not available. Call load_model() first.")

        self.log.info(f"Transcribing with WhisperKit via Swift bridge (model: {self.model_name})")

        try:
            # Preprocess audio to meet WhisperKit requirements
            processed_audio = self._preprocess_audio(audio)

            # Save audio to temporary file for the bridge
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Write audio as 16kHz mono WAV file
                sf.write(temp_path, processed_audio, 16000)

                # Call the Swift bridge (longer timeout for first run with model download)
                result = subprocess.run(
                    [self._bridge_path, temp_path, "--format", "json", "--model", self.model_name],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for model download on first run
                )

                if result.returncode != 0:
                    raise RuntimeError(f"WhisperKit bridge failed: {result.stderr}")

                # Parse JSON result
                output = json.loads(result.stdout)

                # Log the actual transcription time from Swift (excluding bridge overhead)
                transcription_time = output.get("transcription_time", 0)
                self.log.info(f"WhisperKit transcription time: {transcription_time:.4f}s")

                # Store the transcription time for the benchmark to use
                result_obj = TranscriptionResult(
                    text=output.get("text", ""),
                    segments=output.get("segments", []),
                    language=output.get("language", "en"),
                )

                # Add the actual transcription time as an attribute for the benchmark
                result_obj._transcription_time = transcription_time

                return result_obj

            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except:
                    pass

        except subprocess.TimeoutExpired:
            self.log.error("WhisperKit bridge timed out")
            raise RuntimeError("WhisperKit transcription timed out")
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to parse bridge output: {e}")
            raise RuntimeError(f"Failed to parse WhisperKit bridge output: {e}")
        except Exception as e:
            self.log.error(f"Transcription failed: {e}")
            raise RuntimeError(f"WhisperKit transcription failed: {e}")

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio to meet WhisperKit requirements.

        Args:
            audio: Input audio array

        Returns:
            Preprocessed audio array (16kHz, mono, float32, normalized)
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            self.log.debug(f"Converted audio to float32")

        # Ensure audio is 1D (mono)
        if audio.ndim > 1:
            if audio.shape[1] > 1:
                # Convert stereo to mono by averaging channels
                audio = np.mean(audio, axis=1)
                self.log.debug("Converted stereo audio to mono")
            else:
                audio = audio.flatten()

        # Normalize audio to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            self.log.debug(f"Normalized audio from max {max_val:.3f} to 1.0")
        elif max_val == 0.0:
            self.log.warning("Audio contains only silence")

        # Ensure minimum length (avoid empty audio)
        if len(audio) < 1600:  # 0.1 seconds at 16kHz
            self.log.warning(f"Audio is very short ({len(audio)} samples), padding with zeros")
            audio = np.pad(audio, (0, 1600 - len(audio)), mode='constant')

        self.log.debug(f"Preprocessed audio: shape={audio.shape}, dtype={audio.dtype}, "
                      f"range=[{np.min(audio):.3f}, {np.max(audio):.3f}]")

        return audio

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for this implementation."""
        return {
            "model": self.model_name or "base",
            "backend": "WhisperKit Swift Bridge",
            "platform": "Apple Silicon",
        }

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No persistent resources to clean up for the bridge implementation
        self.log.info("WhisperKit bridge implementation cleaned up")
