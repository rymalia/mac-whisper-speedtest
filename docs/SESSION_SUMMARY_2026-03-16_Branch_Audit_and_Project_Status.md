# Session Summary: Branch Audit & Project Status Review

**Date:** 2026-03-16
**Purpose:** Comprehensive audit of all git branches, project status assessment, and planning next steps after a ~7 week gap since last session (2026-01-28)

---

## Session Context

I've reviewed all session summaries from the 2026-01-13 through 2026-01-28 development sprint, the CLAUDE.md, GEMINI.md, README.md, codebase exploration doc, version audit, and all untracked scratch files. This document synthesizes the full picture.

---

## Git Branch Audit

### Branch Topology

```
main (3b76780) ─── 21 commits ahead of dev ─── Active primary branch
│
├── feature/cli-batch-mode (562acc7) ─── ancestor of main ─── MERGED
├── planning (a342927) ─── ancestor of main ─── MERGED
│
dev (e854fa7) ─── 13 commits on diverged path ─── STALE
├── feature/check-models (305e80c) ─── same commit ─┐
├── feature/model-cache-manager (305e80c) ─── same commit ─┤── ALL IDENTICAL
├── feature/templating-add-implementation (305e80c) ─── same commit ─┤
└── feature/update-integration-versions (305e80c) ─── same commit ─┘
```

### Branch Details

| Branch | HEAD | Status | Assessment |
|--------|------|--------|------------|
| **`main`** | `3b76780` | Active, 1 commit ahead of `origin/main` | Primary working branch. Contains all upgrade initiative work, batch mode, CoreML improvements, research docs. |
| **`dev`** | `e854fa7` | Stale, diverged | Early fork work: initial M3 setup, check-models CLI, model name refactoring, whisper-mps MLX discovery, implementation docs. All work superseded by main's structured approach. Remote backup at `origin/dev`. |
| **`feature/cli-batch-mode`** | `562acc7` | Merged | Commit is ancestor of main. Batch mode landed as `61a2af8`. **Safe to delete.** |
| **`planning`** | `a342927` | Merged | Commit is ancestor of main. Codebase exploration landed. **Safe to delete.** |
| **`feature/check-models`** | `305e80c` | Stale placeholder | Identical to 3 other feature branches. Check-models work was done on dev, superseded by main's batch mode. **Safe to delete.** |
| **`feature/model-cache-manager`** | `305e80c` | Never started | Placeholder only. Model handling documented in `feat_model-handling-issues.md` but not yet implemented. **Safe to delete.** |
| **`feature/templating-add-implementation`** | `305e80c` | Never started | Placeholder only. Covered by CODEBASE_EXPLORATION.md's "Adding a New Implementation" section. **Safe to delete.** |
| **`feature/update-integration-versions`** | `305e80c` | Superseded | The upgrade initiative (MLX, WhisperKit, FluidAudio, HuggingFace) was completed directly on main. **Safe to delete.** |

### Recommended Git Cleanup Commands

```bash
# Delete local branches that are already merged or superseded
git branch -d feature/cli-batch-mode
git branch -d planning

# Delete local branches that were never developed (force needed since they diverged)
git branch -D feature/check-models
git branch -D feature/model-cache-manager
git branch -D feature/templating-add-implementation
git branch -D feature/update-integration-versions

# Optional: rename dev to experiments (as previously discussed)
git branch -m dev experiments

# Optional: push the unpushed main commit to origin
git push origin main

# Optional: delete remote dev after confirming you don't need it
# git push origin --delete dev
```

### Untracked Files in Working Tree

| File | Assessment |
|------|------------|
| `.vscode/` | IDE settings, gitignored territory |
| `GEMINI.md` | Gemini CLI project file — keep untracked or commit |
| `feat_model-handling-issues.md` | Active planning doc for model pre-loader feature — keep |
| `scratch_next-deep-dive.md` | Investigation prompts for WhisperKit large model + LightningWhisperMLX — keep as reference |
| `tests/ted_60.m4a` | Test audio file (m4a format) — potentially commit alongside existing wav files |

---

## What Was Accomplished (2026-01-13 → 2026-01-28)

### Infrastructure
- **Batch mode** (`--batch`, `--audio` flags) — enables CI/CD and agent-driven benchmarking
- **CoreML auto-download** — `.mlmodelc` models fetched from HuggingFace automatically
- **Runtime CoreML detection** — warns if pywhispercpp lacks CoreML support
- **104 tests** across 8 test files

### Dependency Upgrades (Upgrade Initiative)

| Package | Before | After | Impact |
|---------|--------|-------|--------|
| MLX + MLX-Metal | 0.27.1 | 0.30.3 | Neural Accelerator support, mxfp4 quantization |
| WhisperKit (Swift) | 0.13.1 | 0.15.0 | struct→class, swift-transformers 1.1.6 |
| FluidAudio (Swift) | 0.1.0 | 0.10.0 | **4.6x faster**, API rewrite, Swift 6 |
| huggingface-hub | 0.30.2 | 1.3.4 | Major version jump |
| transformers | 4.51.3 | 5.0.0 | Major version, required insanely.py fix |
| accelerate | 1.6.0 | 1.12.0 | HW acceleration improvements |
| safetensors | 0.5.3 | 0.7.0 | Model loading |

### Research Completed
- **Moonshine ASR** — full analysis + implementation plan (`feature_plan_moonshine_implementation.md`)
- **mlx-audio / VibeVoice** — full analysis + implementation plan (`feature_plan_vibevoice_implementation.md`)
- **pywhispercpp CoreML** — build guide with empirical benchmarks showing 1.9-2.8x speedup

---

## Where We Left Off

The last session (2026-01-28) completed research on Moonshine. The project was in a "research complete, ready to implement" state for two new ASR engines. There is also remaining dependency upgrade work.

### Unpushed Work
- 1 commit on main (`3b76780 docs: add ASR research...`) not yet on `origin/main`

---

## Planned Next Steps (from session summaries)

### New Implementations (Ready to Build)

1. **MoonshineOnnxImplementation** — ONNX Runtime, 5-15x faster on short audio, plan at `feature_plan_moonshine_implementation.md`
2. **VibeVoiceImplementation** — LLM-based ASR (Qwen2 7B) with speaker diarization, plan at `feature_plan_vibevoice_implementation.md`

### Remaining Dependency Upgrades (from Version Audit)

| Package | Current | Latest* | Priority | Risk |
|---------|---------|---------|----------|------|
| `torch` + `torchaudio` | 2.6.0 | 2.9.1+ | MEDIUM | Medium |
| `faster-whisper` | 1.1.1 | 1.2.1+ | MEDIUM | Low |
| `parakeet-mlx` | 0.3.5 | 0.5.0+ | MEDIUM | Low |
| `coremltools` | 8.3.0 | 9.0+ | MEDIUM | Medium |
| `whisper-mps` | 0.0.7 | 0.0.10+ | MEDIUM | Low |

*Versions from the Jan audit — likely newer releases exist now after 7 weeks.

### Feature Ideas (from scratch files)

- **Model pre-loader script** — documented in `feat_model-handling-issues.md`. Download all needed models before benchmarking to avoid timeout issues during runs.
- **Streaming benchmark mode** — mentioned in GEMINI.md roadmap

### Known Issues (Unresolved)

| Issue | Severity | Description |
|-------|----------|-------------|
| P0 | High | 300s Python timeout insufficient for large models on slow networks |
| P0 | High | No download completeness check (swift-transformers HubApi) |
| P1 | Medium | No download resume capability |
| P2 | Low | Orphaned temp files from failed downloads |
| Test | Low | `test_model_name_mapping_examples` wrong expectation for LightningWhisperMLX |

---

## My Recommendations

### Priority 1: Housekeeping (Quick Wins)

1. **Clean up branches** — Run the git commands above. 6 stale branches are pure noise.
2. **Push unpushed commit** — `3b76780` is sitting locally.
3. **Decide on `GEMINI.md`** — Either gitignore it or commit it. Having it untracked indefinitely is ambiguous.

### Priority 2: Implement Moonshine (Best Next Feature)

Moonshine is the stronger candidate to implement first over VibeVoice because:
- **Lightweight dependency** — just `useful-moonshine-onnx` (ONNX Runtime)
- **Genuinely different architecture** — proportional compute makes it 5-15x faster on short audio, which will be eye-catching in benchmark results
- **Simpler integration** — text-only output, no speaker diarization to parse
- **Cross-platform** — ONNX Runtime works everywhere, not Apple-only

VibeVoice is interesting but it's a 9B parameter model requiring 8-32GB RAM, which changes the benchmarking dynamics significantly.

### Priority 3: Model Pre-Loader (Addresses P0 Issues)

The `feat_model-handling-issues.md` doc outlines a real pain point. Rather than a standalone script, I'd suggest:
- Add a `--prefetch` CLI flag that runs `load_model()` on all selected implementations without timing, reporting download status
- This integrates naturally with the existing CLI rather than being a separate tool
- Addresses the P0 timeout issues by separating "download" from "benchmark"

### Priority 4: Fresh Dependency Audit

It's been ~7 weeks since the last audit. Before upgrading individual packages, run a fresh `uv lock --upgrade --dry-run` to see what's changed. The torch/torchaudio jump (2.6→2.9+) would benefit `insanely-fast-whisper` through MPS backend improvements, but is a large download and medium risk.

### Lower Priority (Nice to Have)

- **`faster-whisper` 1.1.1→1.2.1** — low risk, adds distil-large-v3.5 support. Quick win.
- **Streaming benchmark mode** — interesting but large scope; Moonshine's streaming patterns (documented in research) could inform the design.
- **Extract shared audio preprocessing** — the CODEBASE_EXPLORATION doc notes ~60 lines duplicated between whisperkit.py and fluidaudio_coreml.py. Minor but clean.

---

## Session Statistics

- **Files reviewed:** ~15 docs, CLAUDE.md, GEMINI.md, README.md, pyproject.toml
- **Branches audited:** 8 local + 2 remote-only
- **Branch cleanup identified:** 6 branches safe to delete, 1 to optionally rename
- **Gap since last session:** ~47 days (2026-01-28 → 2026-03-16)
