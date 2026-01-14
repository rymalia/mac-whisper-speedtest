# Implementation Documentation Instructions Template

Use this template prompt in new Claude Code sessions to generate detailed model documentation for each Whisper implementation in this project.

> **WARNING TO CLAUDE AGENTS**: You have a tendency to infer behavior from
> code and present it as empirical observation. This is unacceptable for this
> documentation. You MUST run the actual commands and include real terminal
> output. If you cannot run the commands, you MUST mark the documentation as
> incomplete and explain why. **Code analysis alone is NOT sufficient.**

## Template Prompt

Follow this prompt, replacing `{IMPLEMENTATION_NAME}` with the target implementation:

---

You are an expert veteran coder: very experienced developing and debugging python & Swift. You are also very experienced working with Large Language Models, specifically downloading and managing model files in a local environment. Please trace the execution flow for {IMPLEMENTATION_NAME} and document everything related to the model used and movements of data or files.

## Tasks

### 1. Trace Benchmark Execution Flow (Code Analysis)

Trace this command by reading source code:
```bash
.venv/bin/python3 test_benchmark2.py small 2 {IMPLEMENTATION_NAME}
```

And also trace a variant using the large model option:
```bash
.venv/bin/python3 test_benchmark2.py large 2 {IMPLEMENTATION_NAME}
```

Document:
- Entry point through test_benchmark2.py
- How load_model() is called - are any parameters passed to it?
- Does the model size name parameter get mapped to another value as the model name at any point?
- What library/module actually handles the download (make special note)
- The HuggingFace repo ID or download URL used
- Cache locations (both HF cache and any local/custom cache)
- Files downloaded (weights, config, etc.)
- How transcribe() loads the model

### 2. Empirical Testing (MANDATORY - DO NOT SKIP)

**This section is REQUIRED before documentation is considered complete.**

You must ACTUALLY run the benchmark and observe model file behavior. **Test BOTH model sizes** because they can have significantly different behaviors:

| Size | Why Test It |
|------|-------------|
| `small` | Quick baseline (~500MB), validates basic functionality |
| `large` | Exposes timeout issues, download problems (~2-3GB), stress tests caching |

> **CRITICAL LESSON LEARNED**: In WhisperKit, the `small` model (487MB) downloads fine within 300s timeout, but `large` (2.9GB) **always fails** due to timeout. Testing only `small` would have missed this P0 bug.

#### Test Procedure (Repeat for BOTH small AND large):

**Phase A: Test `small` model**
1. **Locate** the expected cache folder for this implementation (from code analysis)
2. **ASK PERMISSION** before renaming: "May I rename [folder path] to test cache behavior?"
3. **Rename** the small model folder: `small` → `sma__OFF__ll`
4. **Run**: `.venv/bin/python3 test_benchmark2.py small 1 {IMPLEMENTATION_NAME}`
5. **Observe and document**:
   - Exact terminal output showing download progress
   - Use `ls -la [cache_folder]` to show files/folders created
   - Total download size and time
6. **Restore** the folder name after testing

**Phase B: Test `large` model**
1. **Rename** the large model folder: `large` → `lar__OFF__ge` (or `large-v3` → `large__OFF__-v3`)
2. **Run**: `.venv/bin/python3 test_benchmark2.py large 1 {IMPLEMENTATION_NAME}`
3. **Document**: Same observations as Phase A
4. **IMPORTANT**: If the large model times out, this is a **critical finding** - document the timeout and try running the underlying tool directly (without Python timeout) to complete the download
5. **Restore** the folder name after testing

**Phase C: Test cached behavior**
1. With models now downloaded, run both sizes again
2. Verify no re-download occurs and document cached performance

#### Three Model States to Test (for each size):
- **No local model exists**: Does it only download from remote or check other local caches first?
- **Complete local model exists**: How does it verify completeness or as-expected?
- **Partial/incomplete model exists**: Does it wipe & start over or continue the download?

**You MUST include actual terminal output for BOTH model sizes in the documentation.**

#### What To Do When Large Model Times Out

If the large model download exceeds the Python subprocess timeout (common for 2-3GB models):

1. **Document the timeout as a P0 finding** - This is critical information
2. **Clean up any partial cache** - Incomplete files can cause permanent failure:
   ```bash
   rm -rf [cache_folder]/[model_name]
   rm -f /private/var/folders/*/T/CFNetworkDownload_*.tmp  # macOS temp files
   ```
3. **Run the underlying tool directly** (bypassing Python timeout):
   - For Swift bridges: `tools/{bridge}/.build/release/{bridge} tests/jfk.wav --model large-v3`
   - For Python libraries: Call the library directly in a Python REPL with no timeout
4. **Wait for download to complete** (may take 15-30+ minutes for large models)
5. **Re-run the Python benchmark** to verify cached behavior works
6. **Document both**: The timeout failure AND the successful cached run

> **Example from WhisperKit**: The 300s timeout caused the 2.9GB large model to always fail. Running the Swift bridge directly (without Python timeout) took ~27 minutes to complete. After caching, subsequent Python benchmark runs completed in ~7 seconds.

### 3. Identify Issues and Propose Improvements

Based on your code analysis and empirical testing, identify issues and propose concrete improvements. Consider:

#### Issue Categories to Look For:
- **Timeout issues**: Are timeouts sufficient for model downloads? First-run scenarios?
- **Cache behavior**: Does it use standard HuggingFace cache or custom location? Can models be shared?
- **Error recovery**: What happens on partial downloads, missing files, or corruption?
- **User experience**: Is there progress feedback? Warnings for long operations?
- **Feature gaps**: Does it ignore parameters? Missing functionality vs other implementations?
- **Resource efficiency**: Redundant downloads? Unnecessary disk usage?

#### For Each Issue Found, Document:
1. **Problem**: Clear description of the issue
2. **Impact**: How it affects users or system behavior
3. **Location**: File and line number(s) where the issue exists
4. **Recommended Fix**: Concrete code changes (with code samples where helpful)
5. **Effort Estimate**: 1 line / ~20 lines / ~50 lines / Medium / Large
6. **Priority**: P0 (Critical), P1 (High), P2 (Medium), P3 (Low/Future)

#### Priority Guidelines:
- **P0 (Critical)**: Blocks normal operation (e.g., timeouts that always fail)
- **P1 (High)**: Significant UX issues or missing essential functionality
- **P2 (Medium)**: Improvements that save time/bandwidth or improve reliability
- **P3 (Low/Future)**: Nice-to-have features, ecosystem consistency, future-proofing

### 4. Did We Miss Anything?

If the Claude agent has any other relevant findings or insights not accounted for in the output document format listed below, **PLEASE POINT THAT OUT** so the document format can be improved.

### 5. Output Documentation

Write all findings to: `docs/model_details_{IMPLEMENTATION_NAME}.md`

Structure:
1. Title and intro
2. "## File Reference Legend" - reference to indicate key files as: [PROJECT], [LIBRARY], [BRIDGE]
3. "## Key Questions Answered" - table summarizing answers to investigation questions with evidence references
4. "## Benchmark Execution Flow" - with command and numbered steps
5. "## Summary Table" - requested model, actual repo, URLs, cache locations, files
6. "## Model Mapping Reference" (if applicable)
7. "## Notes"
8. "## Key Source Files"
9. "## Empirical Test Results" - **MUST include**:
   - Date of test
   - **Subsections for BOTH sizes**: "### Small Model Tests" and "### Large Model Tests"
   - Exact command(s) run (copy from conversation)
   - Terminal output (copy/paste actual output)
   - `ls -la` output showing created files/folders
   - File sizes observed for each model
   - Download times and any timeout issues
   - Cached run performance (second run without download)
   - **If NOT tested**: explicit `> ⚠️ NOT EMPIRICALLY VERIFIED — CODE ANALYSIS ONLY` warning at section top
   - **If large model timed out**: document as critical finding with workaround
10. "## Known Issues / Conflicts Discovered" (explain for fix-it lists or workarounds)
11. "## Recommended Improvements" - **REQUIRED** - includes:
    - Individual improvement proposals with Problem/Impact/Location/Fix/Effort/Priority
    - Code samples for proposed fixes (both quick fixes and better solutions)
    - "## Priority Summary" table with columns: Priority | Improvement | Effort | Impact | Status
    - "## Implementation Order Recommendation" with phased approach and checkboxes

Key files to read:
- test_benchmark2.py
- src/mac_whisper_speedtest/benchmark.py
- src/mac_whisper_speedtest/cli.py
- src/mac_whisper_speedtest/implementations/{implementation_file}.py
- The underlying library in .venv/lib/python3.12/site-packages/

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

---

## Completion Criteria

A documentation file is ONLY considered complete when:
- [ ] Code analysis flow is documented for both `small` and `large` model paths
- [ ] Benchmark was ACTUALLY run for **BOTH** `small` AND `large` models (not just analyzed)
- [ ] Terminal output from benchmark runs is included for **BOTH sizes**
- [ ] Model file locations were verified with `ls` commands for **BOTH sizes**
- [ ] The "Empirical Test Results" section contains actual observed data (not inferred)
- [ ] "Key Questions Answered" table is included near the top of the document
- [ ] "Recommended Improvements" section includes at least one improvement proposal
- [ ] Priority Summary table is included with effort estimates and status tracking
- [ ] Implementation Order Recommendation is included with phased checkboxes
- [ ] If `large` model timed out, this is documented as a P0 issue with workaround

**DO NOT mark as [x] in the checklist below until ALL criteria above are met.**

---

## Completed Documentation

Track which implementations have been documented:

- [ ] `LightningWhisperMLXImplementation` - docs/model_details_LightningWhisperMLXImplementation.md
- [x] `MLXWhisperImplementation` - docs/model_details_MLXWhisperImplementation.md
- [x] `ParakeetMLXImplementation` - docs/model_details_ParakeetMLXImplementation.md
- [x] `InsanelyFastWhisperImplementation` - docs/model_details_InsanelyFastWhisperImplementation.md
- [x] `WhisperMPSImplementation` - docs/model_details_WhisperMPSImplementation.md
- [x] `FasterWhisperImplementation` - docs/model_details_FasterWhisperImplementation.md
- [x] `WhisperCppCoreMLImplementation` - docs/model_details_WhisperCppCoreMLImplementation.md
- [x] `WhisperKitImplementation` - docs/model_details_WhisperKitImplementation.md
- [x] `FluidAudioCoreMLImplementation` - docs/model_details_FluidAudioCoreMLImplementation.md

## Example Output

See the following completed documentation files for expected format and level of detail:

| Document | Best Example For |
|----------|------------------|
| `docs/model_details_WhisperKitImplementation_large.md` | **Gold standard for large model testing** - comprehensive timeout analysis, partial download behavior, empirical verification of caching bugs, CDN timeout handling |
| `docs/model_details_MLXWhisperImplementation.md` | Basic structure |

