"""Command-line interface for the Whisper benchmark tool."""

import asyncio
from typing import Optional

import numpy as np
import structlog
import typer

from mac_whisper_speedtest.audio import (
    get_default_device,
    record_audio,
    to_whisper_ndarray,
)
from mac_whisper_speedtest.benchmark import BenchmarkConfig, run_benchmark
from mac_whisper_speedtest.implementations import get_all_implementations
from mac_whisper_speedtest.utils import get_models_dir

log = structlog.get_logger(__name__)
app = typer.Typer()


def _check_coreml_availability() -> None:
    """Check if pywhispercpp has CoreML support and log warning if not.

    Only runs on macOS. Logs a helpful message if CoreML is not enabled,
    directing users to the build guide for 2-3x performance improvement.
    """
    import platform

    if platform.system() != "Darwin":
        return

    try:
        from pywhispercpp.model import Model

        info = Model.system_info()

        if "COREML = 0" in info:
            log.warning(
                "pywhispercpp built without CoreML support. "
                "For 2-3x speedup with WhisperCppCoreMLImplementation, "
                "see docs/optimizations_2026-01-13_pywhispercpp_CoreML_Build_Guide.md"
            )
    except ImportError:
        pass  # pywhispercpp not installed


# ─────────────────────────────────────────────────────────────
# Default values for CLI options
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "small"
DEFAULT_NUM_RUNS = 3
DEFAULT_AUDIO_FILE = "tests/jfk.wav"


def load_audio_file(file_path: str) -> np.ndarray:
    """Load audio file and convert to Whisper-compatible format (16kHz mono float32)."""
    import os

    import soundfile as sf

    # Validate file exists
    if not os.path.exists(file_path):
        raise typer.BadParameter(f"Audio file not found: {file_path}")

    try:
        print(f"Loading audio from: {file_path}")
        audio_data, sample_rate = sf.read(file_path, dtype="float32")
        print(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
    except Exception as e:
        raise typer.BadParameter(f"Failed to read audio file: {e}")

    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        print("Converted stereo to mono")

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        print(f"Resampled to 16kHz: {len(audio_data)} samples")

    return audio_data


@app.command()
def benchmark(
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m", help="Model size: tiny/base/small/medium/large"
    ),
    implementations: Optional[str] = typer.Option(
        None,
        "--implementations",
        "-i",
        help="Comma-separated implementation names to benchmark",
    ),
    num_runs: int = typer.Option(
        DEFAULT_NUM_RUNS, "--runs", "-n", help="Number of runs per implementation"
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        "-b",
        help="Non-interactive mode using audio file instead of microphone",
    ),
    audio_file: str = typer.Option(
        DEFAULT_AUDIO_FILE, "--audio", "-a", help="Audio file path for batch mode"
    ),
):
    """Benchmark different Whisper implementations on Apple Silicon."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    # Get implementations to benchmark
    all_impls = get_all_implementations()
    if implementations:
        # Filter implementations
        impl_names = [name.strip() for name in implementations.split(",")]
        impls_to_run = [impl for impl in all_impls if impl.__name__ in impl_names]
        if not impls_to_run:
            print(f"No valid implementations found in: {implementations}")
            print(f"Available implementations: {[impl.__name__ for impl in all_impls]}")
            return
    else:
        impls_to_run = all_impls

    # Auto-enable batch mode if --audio is explicitly provided with non-default value
    if audio_file != DEFAULT_AUDIO_FILE and not batch:
        print("Note: --audio provided, enabling batch mode automatically")
        batch = True

    if batch:
        # ─────────────────────────────────────────────────────────────
        # Non-interactive mode: load from file
        # ─────────────────────────────────────────────────────────────
        audio_data = load_audio_file(audio_file)
        print(f"Audio ready for Whisper: {len(audio_data)} samples")

        # Run benchmark
        config = BenchmarkConfig(
            model_name=model,
            implementations=impls_to_run,
            num_runs=num_runs,
            audio_data=audio_data,
        )

        print("\nStarting benchmark...")
        summary = asyncio.run(run_benchmark(config))
        summary.print_summary()
    else:
        # ─────────────────────────────────────────────────────────────
        # Interactive mode: record from microphone
        # ─────────────────────────────────────────────────────────────
        device_index, device_name = get_default_device()
        print(f"Using default input device: {device_name}")

        print("\nPress Enter to start recording. Press Enter again to stop recording...")
        input()
        print("Recording... Press Enter to stop.")

        # Create stop event for recording
        stop_event = asyncio.Event()

        # Run the async parts in a new event loop
        async def run_async_parts():
            # Start recording task
            record_task = asyncio.create_task(
                record_audio(
                    stop_event,
                    convert=to_whisper_ndarray,
                    device=device_index,
                )
            )

            # Wait for user to press Enter to stop recording
            await asyncio.get_event_loop().run_in_executor(None, input)
            stop_event.set()

            # Get recorded audio
            audio_data = await record_task
            print("Recording stopped. Starting benchmark...")

            # Run benchmark
            config = BenchmarkConfig(
                model_name=model,
                implementations=impls_to_run,
                num_runs=num_runs,
                audio_data=audio_data,
            )

            summary = await run_benchmark(config)
            return summary

        # Run the async parts and get the summary
        summary = asyncio.run(run_async_parts())
        summary.print_summary()


def main():
    """Entry point for the CLI."""
    # Create models directory at startup if it doesn't exist
    models_dir = get_models_dir()
    log.info(f"Models directory: {models_dir}")

    # Check if pywhispercpp has CoreML support and warn if not
    _check_coreml_availability()

    # Run the application
    app()


if __name__ == "__main__":
    main()
