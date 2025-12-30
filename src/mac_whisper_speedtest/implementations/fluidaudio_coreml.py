"""FluidAudio CoreML implementation using subprocess bridge to Swift framework."""

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


class FluidAudioCoreMLImplementation(WhisperImplementation):
    """FluidAudio implementation using Swift bridge to native FluidAudio framework."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self.model_name = None
        self._bridge_path = None

        # Check if we're on macOS (FluidAudio only works on Apple platforms)
        if platform.system() != "Darwin":
            raise RuntimeError("FluidAudio is only supported on macOS")

        # Find the Swift bridge executable
        self._find_bridge_executable()

        self.log.info("====== ====== ====== ====== ====== ======")
        self.log.info("Implementation: FluidAudio Whisper implementation using Swift bridge to native FluidAudio framework")
        self.log.info("FluidAudio CoreML implementation using subprocess bridge to Swift framework")
        self.log.info("====== ====== ====== ====== ====== ======")

    def _find_bridge_executable(self):
        """Find the FluidAudio Swift bridge executable."""
        # Look for the bridge in the tools directory
        project_root = Path(__file__).parent.parent.parent.parent
        bridge_path = project_root / "tools" / "fluidaudio-bridge" / ".build" / "release" / "fluidaudio-bridge"

        if bridge_path.exists():
            self._bridge_path = str(bridge_path)
            self.log.info(f"Found FluidAudio bridge at: {self._bridge_path}")
        else:
            raise RuntimeError(f"FluidAudio bridge not found at {bridge_path}. "
                             f"Please build it first by running: cd tools/fluidaudio-bridge && swift build -c release")

    def load_model(self, model_name: str) -> None:
        """Load the FluidAudio model (via Swift bridge).

        Args:
            model_name: Model size (currently only supports 'small' equivalent)
        """
        if not self._bridge_path:
            raise RuntimeError("FluidAudio bridge not available")

        self.model_name = model_name
        self.log.info(f"FluidAudio bridge ready for model: {self.model_name}")

        # Test that the bridge is working by running it with --help
        try:
            result = subprocess.run(
                [self._bridge_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.log.info("FluidAudio bridge is working correctly")
            else:
                raise RuntimeError(f"FluidAudio bridge test failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("FluidAudio bridge test timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to test FluidAudio bridge: {e}")



    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe the given audio data using FluidAudio Swift bridge.

        Args:
            audio: The audio data to transcribe (16kHz mono)

        Returns:
            The transcription result
        """
        if not self._bridge_path:
            raise RuntimeError("FluidAudio bridge not available. Call load_model() first.")

        self.log.info(f"Transcribing with FluidAudio via Swift bridge")

        try:
            # Preprocess audio to meet FluidAudio requirements
            processed_audio = self._preprocess_audio(audio)

            # Save audio to temporary file for the bridge
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Write audio as 16kHz mono WAV file
                sf.write(temp_path, processed_audio, 16000)

                # Call the Swift bridge (longer timeout for first run with model download)
                result = subprocess.run(
                    [self._bridge_path, temp_path, "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for model download on first run
                )

                if result.returncode != 0:
                    raise RuntimeError(f"FluidAudio bridge failed: {result.stderr}")

                # Parse JSON result
                import json
                output = json.loads(result.stdout)

                # Log the actual transcription time from Swift (excluding bridge overhead)
                transcription_time = output.get("transcription_time", 0)
                processing_time = output.get("processing_time", 0)
                self.log.info(f"FluidAudio transcription time: {transcription_time:.4f}s "
                            f"(internal processing: {processing_time:.4f}s)")

                # Store the transcription time for the benchmark to use
                result_obj = TranscriptionResult(
                    text=output.get("text", ""),
                    segments=[],  # FluidAudio bridge doesn't provide segments yet
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
            self.log.error("FluidAudio bridge timed out")
            raise RuntimeError("FluidAudio transcription timed out")
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to parse bridge output: {e}")
            raise RuntimeError(f"Failed to parse FluidAudio bridge output: {e}")
        except Exception as e:
            self.log.error(f"Transcription failed: {e}")
            raise RuntimeError(f"FluidAudio transcription failed: {e}")

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio to meet CoreML model requirements.

        Args:
            audio: Input audio array

        Returns:
            Preprocessed audio array (16kHz, mono, float32, normalized)
        """

        self.log.debug("Preprocessing audio for FluidAudio CoreML")
        
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
            "model": f"fluidaudio-{self.model_name}" if self.model_name else "fluidaudio-asr",  # parakeet-tdt-0.6b-v3-coreml
            "backend": "FluidAudio Swift Bridge",
            "platform": "Apple Silicon",
        }

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information for verification/download.

        NOTE: FluidAudio uses a single fixed model (parakeet-tdt-0.6b-v3-coreml)
        regardless of the requested model_name parameter. This is an architectural
        limitation of the FluidAudio framework which doesn't support multiple model
        sizes like standard Whisper implementations.

        The parakeet-tdt-0.6b model is roughly equivalent to Whisper's small/base
        models in terms of size (~600M parameters).

        Args:
            model_name: Requested model size (ignored by FluidAudio)

        Returns:
            ModelInfo for the fixed parakeet-tdt-0.6b-v3-coreml model

        TODO: Timeout-based verification not yet implemented for bridge implementations.
              Bridge implementations currently use size-based verification only.
              Future enhancement: Add timeout_seconds field and integrate with
              check_models.py timeout verification (see HF implementations for pattern).
        """
        from pathlib import Path

        # Log for transparency when non-standard model sizes are requested
        # Parakeet ~600M params is roughly equivalent to small/base
        if model_name not in ["small", "base"]:
            self.log.info(
                f"FluidAudio uses parakeet-tdt-0.6b-v3-coreml for all model sizes "
                f"(requested: {model_name}). Parakeet is roughly equivalent to "
                f"Whisper small/base models."
            )

        # FluidAudio uses its own Parakeet model
        home = Path.home()
        model_dir = home / "Library" / "Application Support" / "FluidAudio" / "Models" / "parakeet-tdt-0.6b-v3-coreml"

        cache_paths = [
            model_dir / "Encoder.mlmodelc",
            model_dir / "Decoder.mlmodelc",
            model_dir / "Preprocessor.mlmodelc",
            model_dir / "JointDecision.mlmodelc",
        ]

        return ModelInfo(
            model_name="parakeet-tdt-0.6b-v3-coreml",
            repo_id="FluidInference/parakeet-tdt-0.6b-v3-coreml",
            cache_paths=cache_paths,
            expected_size_mb=460,  # ~425MB Encoder + 23MB Decoder + 12MB other
            verification_method="size",
            download_trigger="bridge"
        )

    def cleanup(self) -> None:
        """Clean up resources used by this implementation."""
        # No persistent resources to clean up for the bridge implementation
        self.log.info("FluidAudio bridge implementation cleaned up")
