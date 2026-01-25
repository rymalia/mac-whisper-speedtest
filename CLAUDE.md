# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking tool that compares 9 different ASR (speech-to-text) implementations on Apple Silicon Macs **most of them based on Whisper**. Measures transcription performance while displaying actual transcription text for quality comparison.

## Session Continuity (READ THIS FIRST)

**At the start of every new session**, check for session summary documents:

```bash
ls -t docs/SESSION_SUMMARY*.md | head -1
```

If a session summary exists, **read it immediately** before doing any other work. These documents contain:
- Unfinished work and next steps from the previous session
- Current state of important files (especially anything in `.state/`)
- Warnings about files that may have been edited between sessions
- Context that prevents repeating already-completed investigations

**Confirm to the user you have reviewed this session summary in your opening greeting of the session**


## Session Summaries

**IMPORTANT:** At the end of each coding session, generate a comprehensive session summary and save it to `docs/SESSION_SUMMARY_YYYY-MM-DD_{VERY SHORT DESCRIPTOR}.md`.

The summary should include:
- **Key Decisions Made**: Strategic choices and rationale
- **Files Modified**: List of all changed files with descriptions
- **Issues Fixed**: Problems identified and solutions implemented
- **Testing Performed**: Verification and validation steps
- **Summary Statistics**: Lines changed, bugs fixed, etc.
- **Unfinished work**: notes and next steps on things that didn't get finished 
- Context that prevents repeating already-completed investigations

A great example of this is: `SESSION_SUMMARY_2026-01-25_MLX_Upgrade_0.30.3.md`


## Commands

### Setup
```bash
uv sync                                                    # Install Python dependencies
cd tools/whisperkit-bridge && swift build -c release       # Build WhisperKit bridge (required)
cd tools/fluidaudio-bridge && swift build -c release       # Build FluidAudio bridge (optional)
```

### Optional: CoreML Acceleration for whisper.cpp

Default `uv sync` installs pywhispercpp **without** CoreML (Metal GPU fallback). For 2-3x speedup:

```bash
uv pip uninstall pywhispercpp && uv cache clean pywhispercpp
WHISPER_COREML=ON WHISPER_COREML_ALLOW_FALLBACK=ON \
  uv pip install --no-cache git+https://github.com/absadiki/pywhispercpp@v1.4.1
```

**Performance**: Small 1.87x faster, Large 2.82x faster. See [CoreML Build Guide](docs/optimizations_2026-01-13_pywhispercpp_CoreML_Build_Guide.md).

### Run Benchmarks

**Interactive mode (microphone):**
```bash
.venv/bin/mac-whisper-speedtest                           # Default: small model, all implementations
.venv/bin/mac-whisper-speedtest -m large -n 5             # Large model, 5 runs
.venv/bin/mac-whisper-speedtest -i "WhisperKitImplementation,MLXWhisperImplementation"
```

**Non-interactive (batch) mode — for CI/agents:**
```bash
.venv/bin/mac-whisper-speedtest --batch                   # Uses default audio (tests/jfk.wav)
.venv/bin/mac-whisper-speedtest --batch --audio tests/ted_60.wav
.venv/bin/mac-whisper-speedtest -b -m large -n 1 -i "MLXWhisperImplementation"
```

**CLI flags:** `-m` model, `-n` runs, `-i` implementations, `-b` batch, `-a` audio

### Tests
```bash
pytest tests/ -v                                          # Run all tests
pytest tests/test_model_params.py -v                      # Model parameter validation
pytest tests/test_parakeet_integration.py -v              # Parakeet integration tests
pytest tests/test_model_params.py::TestModelParams::test_all_implementations_include_model_in_params -v  # Single test
```

## Architecture

### Data Flow
```
CLI (typer) → Audio Recording (PyAudio @ 16kHz) → Benchmark Runner (async)
    → Load each Implementation → Run transcription N times → Measure timing → Display results
```

### Key Components

- **`cli.py`**: Entry point, handles `--model`, `--implementations`, `--runs`, `--batch`, `--audio` args
- **`benchmark.py`**: Core benchmarking logic with `BenchmarkConfig` and `BenchmarkSummary` dataclasses
- **`audio.py`**: Records audio at 16kHz/16-bit mono, converts to float32 numpy arrays
- **`implementations/__init__.py`**: Dynamic registry with conditional imports based on available packages
- **`implementations/base.py`**: Abstract `WhisperImplementation` base class

### Implementation Pattern
All 9 implementations inherit from `WhisperImplementation` and must implement:
- `load_model(model_name)` - Load model, handle name mapping and fallbacks
- `async transcribe(audio)` - Return `TranscriptionResult` with text, segments, language
- `get_params()` - Return dict with `"model"` key and other config info

### Swift Bridges
`tools/whisperkit-bridge/` and `tools/fluidaudio-bridge/` are Swift CLI tools that communicate via JSON over subprocess. They measure internal transcription time (excluding audio loading overhead).

## Implementations

| Category | Implementations |
|----------|----------------|
| Native CoreML | WhisperKit, FluidAudio CoreML |
| MLX-Accelerated | mlx-whisper, Parakeet MLX, lightning-whisper-mlx |
| GPU/MPS | insanely-fast-whisper, whisper-mps |
| CPU | faster-whisper, whisper.cpp+CoreML |

## Requirements

- macOS 14.0+ with Apple Silicon
- Python 3.11+
- Swift 5.9+ / Xcode 15.0+ (for bridges)


## Non-Interactive Benchmarking (Recommended for Agents/CI)

Use `--batch` mode to bypass microphone recording:
```bash
.venv/bin/mac-whisper-speedtest --batch                   # Uses tests/jfk.wav
.venv/bin/mac-whisper-speedtest -b --audio tests/ted_60.wav -n 1
```

This enables:
- **Automated testing** in CI/CD pipelines
- **Agent-based development** (Claude agents cannot access microphones or respond to prompts)
- **Reproducible debugging** (same audio input = deterministic comparison across runs)

Test audio files in `tests/`:
| File | Size | Description |
|------|------|-------------|
| `jfk.wav` | 352 KB | Classic JFK speech sample (Whisper demo standard) |
| `ted_60.wav` | 1.9 MB | 60-second TED talk audio |
| `ted_60_stereo_32.wav` | 15 MB | Stereo 32-bit version (tests audio preprocessing) |

**Why this matters for Claude agents:** Interactive mode requires pressing Enter to start/stop recording and access to microphone hardware—neither of which is available in agent environments. Always use `--batch` mode.

## Model File Verification Guidelines

**IMPORTANT**: Never assume how model files got into any folder. The user frequently:
- Copies/pastes model files between folders during testing
- Copies known-good model folders into the HF cache for reuse across projects

**Verification Method**: The ONLY reliable way to verify which process downloads to which location:
1. **ASK PERMISSION** before deleting any model files
2. Delete (or rename) the expected folder
3. Run the process
4. Observe which folders are created

**User's Rename Trick**: To test without deleting, rename folders by inserting `__OFF__`:
- `tiny` → `ti__OFF__ny`
- `models--Systran--faster-whisper-small` → `models--Systran--faster-whisper-sm__OFF__all`

This makes the folder "invisible" to the process, triggering a fresh download attempt.

**User Preference**: The common HuggingFace cache (`~/.cache/huggingface/hub/`) is PREFERRED because:
- Multiple projects can share the same downloaded models
- Avoids duplicate downloads of large model files
- Libraries that use obscure local cache folders are problematic

## Empirical Testing Requirements

**CRITICAL: Never claim empirical verification without proof.**

Claude agents have a tendency to fabricate test results by inferring behavior from code analysis and presenting it as if tests were actually run. This is unacceptable.

### Rule 1: Show Your Work

Any "Empirical Test Results" section MUST include:
- The **actual Bash commands** you ran (visible in conversation history)
- The **actual terminal output** you received
- If you didn't run tests, explicitly state: `"⚠️ NOT EMPIRICALLY VERIFIED — CODE ANALYSIS ONLY"`

### Rule 2: Ask Permission First

Before deleting or renaming ANY model files for testing:
1. Ask: "May I delete/rename [specific path] to test cache behavior?"
2. **Wait for explicit user confirmation**
3. Only proceed after user says yes

### Rule 3: Mark Unverified Sections Clearly

If you cannot or did not perform empirical testing, mark sections with:
```markdown
> **⚠️ NOT EMPIRICALLY VERIFIED — CODE ANALYSIS ONLY**
>
> The information below was inferred from reading source code, NOT from running tests.
```

### Rule 4: Never Fabricate Results

Do NOT:
- Write fake "Test Date" entries
- Claim you deleted folders when you didn't
- Present code-inferred behavior as observed behavior
- Copy test result formats from other documents and fill in plausible values

### What Counts as Empirical Verification

✅ **Verified**: You ran a Bash command and showed the output
✅ **Verified**: You used the rename trick (`__OFF__`) and observed behavior
✅ **Verified**: Terminal output is visible in the conversation

❌ **Not Verified**: You read the code and inferred the behavior
❌ **Not Verified**: You copied results from another document
❌ **Not Verified**: You assumed behavior based on library documentation

## GitHub Repository Research

When researching version changes, release notes, or changelogs for a dependency hosted on GitHub:

### URL Patterns (in order of preference)

1. **`/releases`** — GitHub's native release page (most reliable, most common)
   - Example: `https://github.com/ml-explore/mlx/releases`
   - Contains tagged releases with full release notes
   - Often includes assets, changelogs, and breaking change notices

2. **`/blob/main/CHANGELOG.md`** — Manual changelog file
   - Example: `https://github.com/org/repo/blob/main/CHANGELOG.md`
   - Some projects maintain this instead of (or in addition to) GitHub releases
   - Also check: `HISTORY.md`, `NEWS.md`, `CHANGES.md`

3. **`/blob/main/MIGRATION.md`** — Migration guides for breaking changes
   - Less common, but valuable when upgrading across major versions

4. **Package registry pages** — PyPI, npm, crates.io often link to changelogs
   - Example: `https://pypi.org/project/mlx/#history`

### Decision Tree

```
Need release/changelog info for a GitHub repo?
│
├─► Try /releases first
│   ├─► Found releases? Use them
│   └─► No releases or sparse notes? Continue...
│
├─► Try /blob/main/CHANGELOG.md
│   ├─► Found? Use it
│   └─► 404? Try HISTORY.md, NEWS.md, CHANGES.md
│
└─► Check package registry (PyPI, npm, etc.)
    └─► Often has release history or links to docs
```

### Why This Matters

Many modern projects use GitHub Releases exclusively and don't maintain a separate CHANGELOG file. Always try `/releases` before assuming a CHANGELOG.md exists.
