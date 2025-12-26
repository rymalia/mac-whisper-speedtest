"""WhisperKit implementation using subprocess bridge to Swift framework.

IMPORTANT: WhisperKit has a known issue with large model downloads (>1GB).
If downloads timeout, this implementation automatically creates the necessary
weights/ directories to allow subsequent download attempts to succeed.

See docs/MODEL_CACHING.md for detailed troubleshooting information.
"""

import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import structlog

from mac_whisper_speedtest.implementations.base import TranscriptionResult, WhisperImplementation, ModelInfo
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

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: WhisperKit Whisper implementation using Swift bridge to native WhisperKit framework")
        self.log.info("WhisperKit implementation using subprocess bridge to Swift framework")
        self.log.info("====== ====== ====== ====== ====== ======")
    
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

    def _ensure_weights_directories(self):
        """Pre-create weights/ directories to prevent download failures.

        WhisperKit has a bug where incomplete downloads leave partial model structures
        without creating the weights/ subdirectories. When retrying, downloads succeed
        but fail to move files to the non-existent weights/ directories.

        This workaround pre-creates the directories to ensure downloads can complete.
        """
        # Determine the model cache path
        # WhisperKit uses: ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/
        home = Path.home()

        # Build model-specific path based on model name
        # Standard models: openai_whisper-{model}
        # Distil models: distil-whisper_distil-{model}
        if "distil" in self.model_name:
            model_dir = f"distil-whisper_distil-{self.model_name.replace('distil-', '')}"
        else:
            model_dir = f"openai_whisper-{self.model_name}"

        model_cache_path = home / "Documents" / "huggingface" / "models" / "argmaxinc" / "whisperkit-coreml" / model_dir

        # Create weights directories for all three CoreML model components
        directories_to_create = [
            model_cache_path / "AudioEncoder.mlmodelc" / "weights",
            model_cache_path / "TextDecoder.mlmodelc" / "weights",
            model_cache_path / "MelSpectrogram.mlmodelc" / "weights",
        ]

        created_any = False
        for directory in directories_to_create:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created_any = True
                self.log.debug(f"Created weights directory: {directory}")

        if created_any:
            self.log.info(f"Pre-created weights directories for {model_dir} to prevent download failures")
        else:
            self.log.debug(f"Weights directories already exist for {model_dir}")

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

        # Workaround for WhisperKit download issue: pre-create weights/ directories
        # This prevents failures when incomplete downloads leave partial model structures
        self._ensure_weights_directories()

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

                # Call the Swift bridge with appropriate timeout
                # Large models (1.4GB+) need substantial time for initial download
                # Small/tiny models (<500MB) download quickly
                if "large" in self.model_name:
                    timeout = 900  # 15 minutes for large models (1.4GB download)
                elif "medium" in self.model_name:
                    timeout = 600  # 10 minutes for medium models
                else:
                    timeout = 300  # 5 minutes for small/tiny models

                result = subprocess.run(
                    [self._bridge_path, temp_path, "--format", "json", "--model", self.model_name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
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

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download."""
        from pathlib import Path

        # Map model names (same as in load_model)
        model_map = {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large-v3",
            "large-v3": "large-v3",
            "large-v3-turbo": "large-v3-turbo",
            "large-turbo": "large-v3-turbo"
        }

        mapped_name = model_map.get(model_name, model_name)

        # Build model-specific path
        if "distil" in mapped_name:
            model_dir = f"distil-whisper_distil-{mapped_name.replace('distil-', '')}"
        else:
            model_dir = f"openai_whisper-{mapped_name}"

        home = Path.home()
        model_cache_path = home / "Documents" / "huggingface" / "models" / "argmaxinc" / "whisperkit-coreml" / model_dir

        # Expected cache paths for the three model components
        cache_paths = [
            model_cache_path / "AudioEncoder.mlmodelc",
            model_cache_path / "TextDecoder.mlmodelc",
            model_cache_path / "MelSpectrogram.mlmodelc",
        ]

        # Expected sizes (in MB) for different models
        size_map = {
            "tiny": 76,
            "small": 467,
            "medium": 1400,
            "large-v3": 2900,
            "distil-large-v3": 229,
        }

        return ModelInfo(
            model_name=mapped_name,
            repo_id=None,  # WhisperKit downloads via its own mechanism
            cache_paths=cache_paths,
            expected_size_mb=size_map.get(mapped_name),
            verification_method="size",
            download_trigger="bridge"
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No persistent resources to clean up for the bridge implementation
        self.log.info("WhisperKit bridge implementation cleaned up")
