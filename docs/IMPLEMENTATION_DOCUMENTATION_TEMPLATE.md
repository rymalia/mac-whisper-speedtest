# Implementation Documentation Template

Use this template prompt in new Claude Code sessions to generate detailed model documentation for each Whisper implementation.

## Template Prompt

Copy and paste this prompt, replacing `{IMPLEMENTATION_NAME}` with the target implementation:

---

You are an veteran expert python coder and Swift coder. Please trace the execution flow for {IMPLEMENTATION_NAME} and document everything related to the model used.

## Tasks

### 1. Trace Benchmark Execution Flow
Trace this command:
```bash
.venv/bin/python3 test_benchmark.py medium 1 {IMPLEMENTATION_NAME}
```

Document:
- Entry point through test_benchmark.py
- How load_model() is called
- What library/module actually handles the download
- The HuggingFace repo ID or download URL used
- Cache locations (both HF cache and any local/custom cache)
- Files downloaded (weights, config, etc.)
- How transcribe() loads the model

### 2. Trace check-models Command Flow
Trace this command:
```bash
.venv/bin/mac-whisper-speedtest check-models --model medium --implementations {IMPLEMENTATION_NAME}
```

Document:
- CLI entry through cli.py
- How get_model_info() is called
- What _get_model_map() returns (if implemented)
- What repo_id check-models looks for in HF cache
- Verification flow (_check_hf_cache, _verify_hf_model, _verify_by_loading)

### 3. Identify Variant Mismatches
Compare:
- What repo/model get_model_info() reports (used by check-models)
- What repo/model load_model() actually downloads (used by benchmark)

If there's a mismatch, document it in a "Known Issue" section with:
- The specific repos that differ
- Impact on check-models accuracy
- Root cause
- Potential fix options

### 4. Output Documentation
Write all findings to: docs/model_details_{IMPLEMENTATION_NAME}.md

Structure:
1. Title and intro
2. "## Benchmark Execution Flow" - with command and numbered steps
3. "## Summary Table" - requested model, actual repo, URLs, cache locations, files
4. "## Model Mapping Reference" (if applicable)
5. "## Notes"
6. "## check-models Command Flow" - with command and numbered steps
7. "## check-models Summary Table"
8. "## Known Issue: Variant Mismatch" (if any mismatch found)

Key files to read:
- src/mac_whisper_speedtest/implementations/{implementation_file}.py
- src/mac_whisper_speedtest/benchmark.py
- src/mac_whisper_speedtest/check_models.py
- src/mac_whisper_speedtest/cli.py
- The underlying library in .venv/lib/python3.12/site-packages/
```

---

## Implementation Names

Use these exact class names when substituting `{IMPLEMENTATION_NAME}`:

| Implementation | File | Backend |
|----------------|------|---------|
| `LightningWhisperMLXImplementation` | lightning.py | MLX (lightning-whisper-mlx) |
| `MLXWhisperImplementation` | mlx.py | MLX (mlx-whisper) |
| `ParakeetMLXImplementation` | parakeet_mlx.py | MLX (parakeet-mlx) |
| `InsanelyFastWhisperImplementation` | insanely.py | PyTorch MPS (transformers) |
| `WhisperMPSImplementation` | whisper_mps.py | PyTorch MPS (whisper-mps) |
| `FasterWhisperImplementation` | faster.py | CTranslate2 CPU |
| `WhisperCppCoreMLImplementation` | coreml.py | whisper.cpp + CoreML |
| `WhisperKitImplementation` | whisperkit.py | Swift bridge (WhisperKit) |
| `FluidAudioCoreMLImplementation` | fluidaudio_coreml.py | Swift bridge (FluidAudio) |

## Completed Documentation

Track which implementations have been documented:

- [x] `LightningWhisperMLXImplementation` - docs/model_details_LightningWhisperMLXImplementation.md
- [x] `MLXWhisperImplementation` - docs/model_details_MLXWhisperImplementation.md
- [x] `ParakeetMLXImplementation` - docs/model_details_ParakeetMLXImplementation.md
- [x] `InsanelyFastWhisperImplementation` - docs/model_details_InsanelyFastWhisperImplementation.md
- [x] `WhisperMPSImplementation` - docs/model_details_WhisperMPSImplementation.md
- [x] `FasterWhisperImplementation` - docs/model_details_FasterWhisperImplementation.md
- [x] `WhisperCppCoreMLImplementation` - docs/model_details_WhisperCppCoreMLImplementation.md
- [x] `WhisperKitImplementation` - docs/model_details_WhisperKitImplementation.md
- [x] `FluidAudioCoreMLImplementation` - docs/model_details_FluidAudioCoreMLImplementation.md

## Example Output

See `docs/model_details_LightningWhisperMLXImplementation.md` for the expected output format and level of detail.
