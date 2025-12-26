"""Benchmark runner for Whisper implementations."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Optional

import numpy as np
import structlog

from mac_whisper_speedtest.implementations.base import BenchmarkResult, WhisperImplementation

log = structlog.get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str
    implementations: List[Type[WhisperImplementation]]
    num_runs: int = 3
    audio_data: Optional[np.ndarray] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    model_name: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def print_summary(self):
        """Print a summary of the benchmark results."""
        print(f"\n=== Benchmark Summary for '{self.model_name}' model ===\n")
        print(f"{'Implementation':<22} {'Avg Time (s)':<15} {'Parameters'}")
        print("-" * 80)

        # Sort results by average time
        sorted_results = sorted(
            self.results,
            key=lambda r: r.transcription_time
        )

        # Map implementation names to shorter versions
        name_map = {
            "WhisperCppCoreMLImplementation": "whisper.cpp",
            "MLXWhisperImplementation": "mlx-whisper",
            "InsanelyFastWhisperImplementation": "insanely-fast-whisper",
            "LightningWhisperMLXImplementation": "lightning-whisper-mlx",
            "FasterWhisperImplementation": "faster-whisper",
            "ParakeetMLXImplementation": "parakeet-mlx",
            "FluidAudioCoreMLImplementation": "fluidaudio-coreml",
            "WhisperKitImplementation": "whisperkit",
            "WhisperMPSImplementation": "whisper-mps",
        }

        for result in sorted_results:
            # Use the short name if available, otherwise use the original
            short_name = name_map.get(result.implementation, result.implementation)

            # Extract the actual model being used for clearer display
            actual_model = result.model_params.get("model", result.model_name)
            if isinstance(actual_model, str) and "/" in actual_model:
                # Show just the model name part for HF repos (e.g., "mlx-community/whisper-small" -> "whisper-small")
                model_display = actual_model.split("/")[-1]
            else:
                model_display = str(actual_model)

            # Create params string excluding the model (since we show it separately)
            params_dict = {k: v for k, v in result.model_params.items() if k != "model"}
            params_str = ", ".join([f"{k}={v}" for k, v in params_dict.items()])

            # Combine model and other params
            if params_str:
                full_params = f"model={model_display}, {params_str}"
            else:
                full_params = f"model={model_display}"

            print(f"{short_name:<22} {result.transcription_time:<15.4f} {full_params}")

            # Add transcription text below each result
            self._print_transcription(result.text)
            print()  # Add blank line between implementations

    def _print_transcription(self, text: str, max_length: int = 100) -> None:
        """Print transcription text with appropriate formatting and truncation.

        Args:
            text: The transcription text to display
            max_length: Maximum length before truncation (default: 100)
        """
        if not text or text.strip() == "":
            print("    (no transcription)")
            return

        # Clean up the text (remove extra whitespace, newlines)
        cleaned_text = " ".join(text.strip().split())

        # Truncate if necessary
        if len(cleaned_text) > max_length:
            truncated_text = cleaned_text[:max_length].rstrip() + "..."
        else:
            truncated_text = cleaned_text

        # Print with indentation and quotes for clarity
        print(f'    "{truncated_text}"')




async def run_benchmark(config: BenchmarkConfig) -> BenchmarkSummary:
    """Run the benchmark with the given configuration.

    Args:
        config: The benchmark configuration

    Returns:
        A summary of the benchmark results
    """
    if config.audio_data is None:
        raise ValueError("Audio data is required for benchmarking")

    summary = BenchmarkSummary(model_name=config.model_name)

    for impl_class in config.implementations:
        impl_name = impl_class.__name__
        log.info("====== ====== ====== ====== ====== ====== ======")
        log.info(f"Benchmarking {impl_name} with model {config.model_name}")

        try:
            # Create implementation instance
            implementation = impl_class()

            # Load the model (not timed)
            log.info(f"Loading model for {impl_name}")
            implementation.load_model(config.model_name)

            # Run multiple times and average
            total_time = 0.0
            last_result = None
            for run in range(config.num_runs):
                log.info(f"Run {run+1}/{config.num_runs} for {impl_name}")

                # Time the transcription
                start_time = time.time()
                result = await implementation.transcribe(config.audio_data)
                end_time = time.time()

                # Use internal transcription time if available (e.g., from FluidAudio bridge)
                # This excludes bridge/subprocess overhead for more accurate measurements
                if hasattr(result, '_transcription_time'):
                    run_time = result._transcription_time
                    log.info(f"Using internal transcription time: {run_time:.4f}s "
                           f"(total with overhead: {end_time - start_time:.4f}s)")
                else:
                    run_time = end_time - start_time
                total_time += run_time
                last_result = result  # Keep the last result for text

                log.info(f"Run {run+1} completed in {run_time:.4f} seconds")
                log.info(f"Transcription: {result.text[:50]}...")

            # Calculate average time
            avg_time = total_time / config.num_runs

            # Add result to summary (use last result's text)
            summary.results.append(BenchmarkResult(
                implementation=impl_name,
                model_name=config.model_name,
                model_params=implementation.get_params(),
                transcription_time=avg_time,
                text=last_result.text if last_result else ""
            ))

            log.info(f"Average time for {impl_name}: {avg_time:.4f} seconds")

            # Clean up
            if hasattr(implementation, 'cleanup') and callable(implementation.cleanup):
                implementation.cleanup()

        except Exception as e:
            log.error(f"Error benchmarking {impl_name}: {e}", exc_info=True)
            # Add failed result
            summary.results.append(BenchmarkResult(
                implementation=impl_name,
                model_name=config.model_name,
                model_params={"error": str(e)},
                transcription_time=float('inf'),
                text=""
            ))

    return summary
