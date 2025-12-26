"""Command-line interface for the Whisper benchmark tool."""

import asyncio
from typing import List, Optional

import structlog
import typer

from mac_whisper_speedtest.audio import (
    get_default_device,
    record_audio,
    to_whisper_ndarray,
)
from mac_whisper_speedtest.benchmark import BenchmarkConfig, run_benchmark
from mac_whisper_speedtest.check_models import ModelChecker
from mac_whisper_speedtest.implementations import get_all_implementations
from mac_whisper_speedtest.utils import get_models_dir

log = structlog.get_logger(__name__)
app = typer.Typer()


@app.command()
def benchmark(
    model: str = typer.Option("small", help="Model size to benchmark"),
    implementations: Optional[str] = typer.Option(
        None, help="Specific implementations to benchmark (comma-separated)"
    ),
    num_runs: int = typer.Option(3, help="Number of runs per implementation"),
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

    # Use default device
    device_index, device_name = get_default_device()
    print(f"Using default input device: {device_name}")

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

    # Record audio
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


@app.command(name="check-models")
def check_models(
    model: str = typer.Option("small", help="Model size to check"),
    implementations: Optional[str] = typer.Option(
        None, help="Specific implementations to check (comma-separated)"
    ),
    download: bool = typer.Option(False, help="Download missing models automatically"),
    selective: bool = typer.Option(False, help="Select specific implementations to download"),
):
    """Check model cache status and optionally download missing models."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    # Get implementations to check
    all_impls = get_all_implementations()
    if implementations:
        # Filter implementations
        impl_names = [name.strip() for name in implementations.split(",")]
        impls_to_check = [impl for impl in all_impls if impl.__name__ in impl_names]
        if not impls_to_check:
            print(f"No valid implementations found in: {implementations}")
            print(f"Available implementations: {[impl.__name__ for impl in all_impls]}")
            return
    else:
        impls_to_check = all_impls

    # Create model checker and check status
    checker = ModelChecker()
    statuses = checker.check_all_models(model, impls_to_check)

    # Print status table
    checker.print_status_table(statuses, model)
    checker.print_summary(statuses)

    # Handle download options
    if download or selective:
        async def run_download():
            if selective:
                # Interactive selection
                await checker.interactive_download(statuses, model, impls_to_check)
            else:
                # Download all missing
                await checker.download_missing_models(statuses, model, impls_to_check)

        asyncio.run(run_download())
    else:
        # Show interactive menu
        async def run_interactive():
            await checker.interactive_menu(statuses, model, impls_to_check)

        asyncio.run(run_interactive())


def main():
    """Entry point for the CLI."""
    # Create models directory at startup if it doesn't exist
    models_dir = get_models_dir()
    log.info(f"Models directory: {models_dir}")

    # Run the application
    app()


if __name__ == "__main__":
    main()
