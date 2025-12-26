#!/usr/bin/env python3
"""Test script to run benchmark with a pre-recorded audio file."""

import asyncio
import soundfile as sf
import structlog

from mac_whisper_speedtest.audio import to_whisper_ndarray
from mac_whisper_speedtest.benchmark import BenchmarkConfig, run_benchmark
from mac_whisper_speedtest.implementations import get_all_implementations

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ]
)

async def main():
    # Load test audio file
    audio_path = "tools/whisperkit-bridge/.build/checkouts/WhisperKit/Tests/WhisperKitTests/Resources/jfk.wav"
    print(f"Loading audio from: {audio_path}")

    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    print(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")

    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        print(f"Resampled to 16kHz: {len(audio_data)} samples")

    # Audio is already in float32 format normalized to [-1, 1]
    whisper_audio = audio_data
    print(f"Audio ready for Whisper: {len(whisper_audio)} samples")

    # Get all implementations
    all_impls = get_all_implementations()
    print(f"\nAvailable implementations: {len(all_impls)}")
    for impl in all_impls:
        print(f"  - {impl.__name__}")

    # Run benchmark
    config = BenchmarkConfig(
        model_name="small",
        implementations=all_impls,
        num_runs=1,  # Just 1 run for testing
        audio_data=whisper_audio,
    )

    print("\nStarting benchmark...")
    summary = await run_benchmark(config)
    summary.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
