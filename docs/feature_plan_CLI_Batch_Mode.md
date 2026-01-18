# Feature Plan: CLI Batch Mode (Non-Interactive Benchmarking)

**Status:** Ready for implementation
**Created:** 2026-01-14
**Purpose:** Guide a fresh Claude agent session to implement this feature

---

## Problem Statement

The benchmark tool currently has three entry points with overlapping functionality:

| File | Location | Interactive? | Parameters | Issue |
|------|----------|--------------|------------|-------|
| `cli.py` | `src/.../` | Yes (microphone) | `--model`, `--impl`, `--runs` via typer | No file input option |
| `test_benchmark.py` | project root | No (file) | Hardcoded | Redundant, no params |
| `test_benchmark2.py` | project root | No (file) | `sys.argv` positional | Hacky, no --help |

**Goal:** Consolidate into a single CLI with both interactive and non-interactive (batch) modes.

---

## Current Architecture

```
.venv/bin/mac-whisper-speedtest  →  cli.py  →  benchmark.py
                                       ↑
                                   (microphone only)

python3 test_benchmark2.py       →  benchmark.py
                                       ↑
                                   (file only, sys.argv parsing)
```

**Key insight:** `cli.py` already uses `typer` with proper `--help` support. The test scripts are workarounds for a missing `--audio` flag.

---

## Decisions Made

### 1. Mode Switch: `--batch` Flag

**Decision:** Use explicit `--batch` / `-b` flag to enable non-interactive mode.

**Rationale:**
- Clearer intent than implicit detection
- Allows `--audio` to have a default value
- Users can run `--batch` without specifying audio file

### 2. Default Audio File

**Decision:** Default to `tests/jfk.wav` when `--batch` is used without `--audio`.

**Usage:**
```bash
mac-whisper-speedtest --batch                    # Uses tests/jfk.wav
mac-whisper-speedtest --batch --audio custom.wav # Uses custom file
```

### 3. Centralize All Defaults

**Decision:** Define all defaults as constants at the top of `cli.py`.

**Rationale:**
- Current defaults are scattered and inconsistent (num_runs: 3 vs 2 vs 5)
- Single source of truth
- Easy to find and modify

```python
# cli.py (top of file)
DEFAULT_MODEL = "small"
DEFAULT_NUM_RUNS = 3
DEFAULT_AUDIO_FILE = "tests/jfk.wav"
```

### 4. Cleanup Root Scripts

**Decision:** Delete `test_benchmark.py` and `test_benchmark2.py` after verification.

**Rationale:** They become obsolete once `--batch` is implemented.

---

## Implementation Plan

### Step 1: Add Constants Block

Location: `src/mac_whisper_speedtest/cli.py` (after imports)

```python
# ─────────────────────────────────────────────────────────────
# Default values for CLI options
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "small"
DEFAULT_NUM_RUNS = 3
DEFAULT_AUDIO_FILE = "tests/jfk.wav"
```

### Step 2: Update CLI Options

Update the `@app.command()` function signature in `cli.py`:

```python
@app.command()
def benchmark(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Model size: tiny/base/small/medium/large"),
    num_runs: int = typer.Option(DEFAULT_NUM_RUNS, "--runs", "-n", help="Number of runs per implementation"),
    implementations: Optional[str] = typer.Option(None, "--implementations", "-i", help="Comma-separated implementation names"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Non-interactive mode using audio file instead of microphone"),
    audio_file: str = typer.Option(DEFAULT_AUDIO_FILE, "--audio", "-a", help="Audio file path for batch mode"),
):
```

**Note:** Keep `--implementations` (not `--impl`) to maintain backwards compatibility with existing scripts. Add `-i` as a short form.

### Step 3: Add Audio File Loading Function

Add helper function in `cli.py`:

```python
def load_audio_file(file_path: str) -> np.ndarray:
    """Load audio file and convert to Whisper-compatible format."""
    import os
    import soundfile as sf

    # Validate file exists
    if not os.path.exists(file_path):
        raise typer.BadParameter(f"Audio file not found: {file_path}")

    try:
        print(f"Loading audio from: {file_path}")
        audio_data, sample_rate = sf.read(file_path, dtype='float32')
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
```

### Step 4: Add Mode Branching Logic

In the `benchmark()` function, replace/wrap the recording logic:

```python
# Auto-enable batch mode if --audio is explicitly provided with non-default value
if audio_file != DEFAULT_AUDIO_FILE and not batch:
    print(f"Note: --audio provided, enabling batch mode automatically")
    batch = True

if batch:
    # Non-interactive mode: load from file
    audio_data = load_audio_file(audio_file)

    # Run benchmark directly (no async recording needed)
    config = BenchmarkConfig(
        model_name=model,
        implementations=impls_to_run,
        num_runs=num_runs,
        audio_data=audio_data,
    )
    summary = asyncio.run(run_benchmark(config))
    summary.print_summary()
else:
    # Interactive mode: record from microphone (existing code)
    print("\nPress Enter to start recording. Press Enter again to stop recording...")
    # ... existing microphone recording code ...
    # (ends with summary.print_summary())
```

**Design note:** If user provides `--audio custom.wav` without `--batch`, we auto-enable batch mode rather than ignoring the flag or erroring out. This is the most intuitive behavior.

### Step 5: Add Required Import

```python
import numpy as np  # For type hint in load_audio_file
```

### Step 6: Verify and Clean Up

1. Test all modes (see verification checklist below)
2. Delete `test_benchmark.py`
3. Delete `test_benchmark2.py`
4. Update CLAUDE.md

---

## Verification Checklist

Run these commands after implementation:

```bash
# Verify --help shows all options
.venv/bin/mac-whisper-speedtest --help

# Batch mode with default audio
.venv/bin/mac-whisper-speedtest --batch

# Batch mode with custom audio
.venv/bin/mac-whisper-speedtest --batch --audio tests/ted_60.wav

# Batch mode with all options
.venv/bin/mac-whisper-speedtest -b -m large -n 1 -i "MLXWhisperImplementation"

# Interactive mode still works (requires microphone)
.venv/bin/mac-whisper-speedtest
```

---

## CLAUDE.md Updates

After implementation, update the "Run Benchmarks" section in CLAUDE.md:

```bash
### Run Benchmarks

# Interactive mode (microphone)
.venv/bin/mac-whisper-speedtest
.venv/bin/mac-whisper-speedtest --model large --runs 5

# Non-interactive (batch) mode - for CI/agents
.venv/bin/mac-whisper-speedtest --batch                           # Default audio (tests/jfk.wav)
.venv/bin/mac-whisper-speedtest --batch --audio tests/ted_60.wav  # Custom audio
.venv/bin/mac-whisper-speedtest -b -m large -n 1 -i "MLXWhisperImplementation"
```

Also update the "Non-Interactive Benchmarking" section to reference the new `--batch` flag instead of `test_benchmark.py`.

---

## Files Summary

| File | Action |
|------|--------|
| `src/mac_whisper_speedtest/cli.py` | **Modify**: Add constants, --batch, --audio, load_audio_file() |
| `test_benchmark.py` | **Delete** after verification |
| `test_benchmark2.py` | **Delete** after verification |
| `CLAUDE.md` | **Update**: Document new --batch usage |

---

## Notes for Implementing Agent

1. **Read cli.py first** to understand the current structure
2. **The async recording logic** is complex - the batch mode path should be simpler (no async needed for file loading)
3. **soundfile and librosa** are already dependencies - verify in pyproject.toml, check test_benchmark.py for usage patterns
4. **Test with actual benchmarks** not just --help - ensure transcription actually works in batch mode
5. **Relative path caveat**: `DEFAULT_AUDIO_FILE = "tests/jfk.wav"` is relative to CWD. If file not found, the error handling will catch it with a clear message.

---

## Open Considerations

These are minor decisions the implementing agent can make:

| Question | Options | Recommendation |
|----------|---------|----------------|
| Keep `--implementations` or change to `--impl`? | Keep long form, add `-i` short | Keep `--implementations` for backwards compat, add `-i` |
| What if default audio file missing? | Error with helpful message | Already handled by `load_audio_file()` validation |
| Import soundfile/librosa at top or inline? | Top-level vs lazy import | Keep inline (lazy) in `load_audio_file()` to match existing pattern |
