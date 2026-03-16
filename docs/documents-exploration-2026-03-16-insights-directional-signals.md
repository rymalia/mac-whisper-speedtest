# mac-whisper-speedtest Documents Exploration

**Date:** 2026-03-16
**Method:** QMD hybrid search (lex + vec) across the full 64-document / 21,500-line collection
**Scope:** Problems, plans, architecture, performance data, user preferences, meta-patterns

---

## QMD Exploration of Documentation: Insights & Directional Signals

After searching the `mac-whisper-speedtest` collection from multiple angles, 9 distinct signals emerged. Together they paint a picture of a project that has done exceptional research and documentation work, but now faces a strategic fork: consolidate and simplify, or continue expanding.

---

### Signal 1: The Documentation Is the Real Product

This project has **59 markdown docs totaling 21,500 lines** — roughly **4x more documentation than source code**. The docs aren't filler; they contain empirical test results, architectural analysis, and bug discoveries that don't exist anywhere else. The session summaries form a coherent narrative across 18 sessions from Jan 7 through Jan 28.

**Document inventory:**

| Category | Count | Total Lines | Notes |
|----------|-------|-------------|-------|
| Session summaries | 25 | ~3,800 | Chronological project history |
| Model detail deep dives | 15 | ~8,500 | Per-implementation investigations |
| Upgrade plans | 5 | ~2,900 | All marked DONE |
| Feature plans | 5 | ~2,000 | Mix of done and planned |
| Research docs | 2 | ~1,400 | Moonshine + mlx-audio |
| Superseded (`docs/crap/`) | 4 | ~2,175 | Earlier attempts, redone |
| Other (explainers, guides, templates) | 3 | ~990 | Reusable reference material |

**The directional signal:** This project's value isn't just the benchmarking tool — it's the **knowledge base of how 9 different ASR implementations actually behave on Apple Silicon**, including their quirks, download failures, and real performance data. If you ever wanted to publish findings or help others choose an ASR engine, the docs are already 80% of the way there.

---

### Signal 2: Model Download Reliability Is the #1 Unsolved Problem

The strongest signal across all docs is the **model download pain point**. It surfaces in:

- `feat_model-handling-issues.md` (dedicated planning doc, sitting untracked in repo root)
- WhisperKit large model deep dive (300s timeout fails for 2.9GB download)
- FluidAudio deep dive (P0: 300s Python timeout < 22 min download)
- WhisperKit documentation merge ("**First-run download always fails** due to timeout")
- Hub library caching bug ("a 4K folder was treated as a complete 1.7GB model" — **permanent failure state**)
- Parakeet deep dive (`CAS service error: UnexpectedEof`)

**Every implementation handles downloads differently. None of them handle it well for large models.** This is the thread that connects your `feat_model-handling-issues.md` planning doc, the P0 bugs in multiple session summaries, and the `scratch_next-deep-dive.md` investigation prompts.

**Catalogued P0/P1 download bugs (still open):**

| Implementation | Bug | Severity |
|---------------|-----|----------|
| WhisperKit (large) | 300s Python timeout < download time → always fails on first run | P0 |
| FluidAudio | 300s Python timeout < 22 min download | P0 |
| WhisperKit | Partial download (4K folder) treated as complete → permanent failure state | P0 |
| FluidAudio | Resume not actually implemented despite code structure suggesting it | P1 |
| WhisperMPS | No resume for interrupted 3GB downloads | P2 |
| All | No download completeness checks | P0 |

**Directional signal:** A `--prefetch` command or model pre-loader would address the single biggest user-facing pain point. It would also make the benchmarking tool much more reliable for first-time users.

---

### Signal 3: mlx-audio Could Be a Strategic Pivot Point

The `research_mlx-audio_analysis.md` contains a buried bombshell:

> **mlx-audio could replace 3 of our 9 implementations** (MLXWhisper, ParakeetMLX, LightningWhisperMLX) with a unified interface

This isn't just a "nice to have" — it's a potential architectural simplification. Those 3 implementations each wrap a different library that all ultimately run on MLX. mlx-audio implements the same models natively in MLX with a unified loader, config-driven quantization, and streaming support built in.

**What an mlx-audio consolidation would look like:**

```
Current (9 implementations, 9 dependencies):
  MLXWhisper ──→ mlx-whisper lib
  ParakeetMLX ──→ parakeet-mlx lib
  LightningMLX ──→ lightning-whisper-mlx lib
  + 6 others (unchanged)

After consolidation (8+ implementations, 7 dependencies):
  MLXAudioWhisper ──→ mlx-audio (native Whisper)
  MLXAudioParakeet ──→ mlx-audio (native Parakeet)
  MLXAudioMoonshine ──→ mlx-audio (native Moonshine)    ← NEW
  MLXAudioVibeVoice ──→ mlx-audio (native VibeVoice)    ← NEW
  + 6 others (unchanged)
```

**Directional signal:** Rather than adding Moonshine and VibeVoice as implementations #10 and #11 (further increasing the maintenance surface), consider adding them *through mlx-audio* which already supports both. This would give you:
- Moonshine, VibeVoice, Whisper, and Parakeet through **one dependency** (`mlx-audio`)
- Streaming support for free
- Potentially replacing 3 existing wrappers with thinner adapters

This is the biggest architectural decision ahead: **grow wider (more implementations) or grow deeper (consolidate on mlx-audio)?**

---

### Signal 4: There's a "docs/crap" Folder (Documentation Debt)

4 superseded docs live in `docs/crap/` (2,175 lines total). These are earlier attempts at FluidAudio, LightningWhisperMLX, and WhisperKit documentation that were redone. They're still indexed by QMD and could cause confusion in searches.

Beyond that, there are multiple documents covering the same ground:
- WhisperKit has **3 separate model_details docs** (small, large, and combined) = 2,070 lines
- The version audit has **5 separate upgrade plan docs** that are now all marked DONE
- The `feature_plan_CLI_Batch_Mode.md` (271 lines) documents work that shipped in January

**Docs cleanup estimate:**

| Action | Lines Removed | Risk |
|--------|---------------|------|
| Delete `docs/crap/` (4 files) | ~2,175 | None — all superseded |
| Archive completed upgrade plans to `docs/archive/` | ~2,900 | Low — still accessible |
| Consolidate WhisperKit model_details (3→1) | ~1,000 | Low — merge, don't delete |
| Archive completed feature plans | ~700 | Low |
| **Total reduction** | **~6,775 (~31%)** | |

**Directional signal:** A docs cleanup pass could cut ~30% of the doc volume without losing information. This makes QMD searches more precise and the project easier to navigate.

---

### Signal 5: The Claude Agent Trust Problem Is Real

A uniquely strong signal: CLAUDE.md has an entire section on **"Empirical Testing Requirements"** with 4 rules about not fabricating results. The implementation documentation template has a **WARNING TO CLAUDE AGENTS** header. Multiple session summaries reference the problem:

> "Claude Opus 4.5 instead of Sonnet 4.5 — SO MUCH BETTER" (dev branch commit message)

The `__OFF__` rename trick, the "show your work" mandate, the "mark unverified sections clearly" rule — these all evolved from actual incidents where AI agents presented code analysis as empirical testing.

**The evolution of trust guardrails:**

1. **Jan 7-10:** Early sessions, no guardrails. Agent fabrication incidents occurred.
2. **Jan 10:** `IMPLEMENTATION_DOCUMENTATION_TEMPLATE.md` created with "WARNING TO CLAUDE AGENTS"
3. **Jan 11+:** CLAUDE.md updated with 4-rule empirical testing framework
4. **Jan 12:** User discovers Opus 4.5 >> Sonnet 4.5 for investigation honesty
5. **Jan 13+:** All session summaries include explicit verification status

**Directional signal:** This hard-won knowledge about working with AI agents on empirical testing could itself be valuable to share. The `IMPLEMENTATION_DOCUMENTATION_TEMPLATE.md` + CLAUDE.md guardrails are a reusable pattern for any project that needs agents to do honest testing.

---

### Signal 6: Streaming Is the Unexplored Frontier

Streaming/real-time ASR appears in:
- GEMINI.md roadmap: "Add streaming benchmark mode"
- Moonshine research: detailed streaming architecture with Silero VAD
- mlx-audio research: native streaming support
- FluidAudio: "real-time streaming ASR with ~110x RTF"
- FluidAudio v0.10.0: added "Parakeet EOU streaming" capability

But **zero implementation work has been done on streaming**. The current architecture is batch-only (record → transcribe → measure).

**What a streaming benchmark would measure:**

| Metric | Batch (current) | Streaming (new) |
|--------|-----------------|-----------------|
| Latency | Total transcription time | Time-to-first-word |
| Throughput | Words per second (post-hoc) | Real-time factor (RTF) |
| Quality | Final transcript accuracy | Intermediate stability (how much text "jumps") |
| Memory | Peak during full-file processing | Sustained during continuous stream |

**Directional signal:** Streaming would be a major differentiation — "which ASR engine gives you the best real-time experience on Apple Silicon?" is a question nobody else has benchmarked systematically. Moonshine's proportional compute (5-15x faster on short segments) makes it especially interesting for streaming benchmarks.

---

### Signal 7: The Deep Dive Investigation Pattern Has Coverage Gaps

Session summaries show deep dive investigations completed for:
- MLXWhisper (Jan 11)
- WhisperKit small (Jan 10) + large (Jan 11) + merged (Jan 13)
- FasterWhisper (Jan 12)
- FluidAudioCoreML (Jan 12)
- InsanelyFastWhisper (Jan 12)
- WhisperMPS (Jan 12)
- LightningWhisperMLX (Jan 7 + Jan 12)
- Parakeet (Jan 11)
- WhisperCppCoreML (has model_details doc but **no session summary**)

The `scratch_next-deep-dive.md` file contains **unexecuted investigation prompts** for WhisperKit large model timeout behavior and a LightningWhisperMLX redo. These were queued but never run.

**Directional signal:** The investigation coverage is nearly complete (8/9 with session summaries). WhisperCppCoreML is the odd one out — it has a model_details doc but no proper deep dive session. The scratch file suggests you wanted to go deeper on timeout behavior specifically.

---

### Signal 8: No Post-Upgrade Benchmark Baseline Exists

All performance data in the docs predates the major upgrade initiative:
- The **README example output** was generated before FluidAudio 0.10 (4.6x faster), MLX 0.30.3, WhisperKit 0.15, and transformers 5.0
- The **CoreML build guide benchmarks** were run on Jan 13 (pre-upgrade)
- The **Apple Silicon optimizations doc** references lightning-whisper-mlx at 2.97s and faster-whisper at 12.48s — likely stale numbers

After upgrading 6+ major dependencies (some with dramatic performance changes), **there is no saved benchmark run showing the current state of the world.**

**What's missing:**

| What | Last Measured | What Changed Since |
|------|-------------|-------------------|
| FluidAudio timing | Pre-0.10 (~0.9s?) | 4.6x faster after upgrade |
| MLX implementations | Pre-0.30 | MLX 0.30.3 with Neural Accelerator support |
| WhisperKit timing | Pre-0.15 | +7% regression noted in upgrade summary |
| InsanelyFastWhisper | Pre-transformers-5.0 | dtype change, quantization disabled on macOS |
| All implementations head-to-head | Never saved post-upgrade | Multiple variables changed simultaneously |

**Directional signal:** Run a fresh `--batch -m small -n 3` and `--batch -m large -n 3` benchmark and save the results. This would give you a clean post-upgrade baseline, update the README's example output, and validate (or invalidate) the performance claims made during individual upgrade sessions.

---

### Signal 9: The Project's Three Phases Reveal a Natural Rhythm

Mapping the 25 session summaries chronologically reveals three distinct phases:

```
Phase 1: EXPLORATION (Jan 7-14) — 12 sessions
├── Deep dive investigations of all 9 implementations
├── Created model_details docs with empirical verification
├── Discovered P0 timeout bugs, caching bugs, fabrication problems
├── Built the documentation template and trust guardrails
└── Culminated in version audit and upgrade planning

Phase 2: INFRASTRUCTURE (Jan 14-26) — 8 sessions
├── Git worktree cleanup and folder reorganization
├── CLI batch mode (enabling agent-driven and CI benchmarking)
├── CoreML auto-download and runtime detection
├── Dependency upgrades: MLX, WhisperKit, FluidAudio, HuggingFace
└── Culminated in upgrade initiative milestone doc

Phase 3: RESEARCH (Jan 26-28) — 2 sessions
├── mlx-audio analysis (discovered consolidation opportunity)
├── Moonshine analysis (proportional compute, streaming patterns)
└── Created implementation plans for both

—— 47-day gap ——

Phase 4: ??? (Mar 16+)
```

**Directional signal:** The rhythm suggests the project alternates between **outward exploration** and **inward consolidation**. Phase 1 explored outward (9 implementations), Phase 2 consolidated inward (upgrades, infrastructure), Phase 3 explored outward again (new ASR research). The natural Phase 4 would be another consolidation: clean up docs, run post-upgrade benchmarks, and make the mlx-audio architectural decision before expanding further.

---

## Summary: Where the Signals Point

| Priority | Direction | Why | Effort |
|----------|-----------|-----|--------|
| **1** | Post-upgrade benchmark baseline | No saved results after major upgrades; README is stale | Low (one command + save output) |
| **2** | Model pre-loader (`--prefetch`) | Addresses #1 pain point across 6+ documents | Medium |
| **3** | Evaluate mlx-audio as unified backend | Could simplify 3→1 implementations AND add Moonshine + VibeVoice | Medium (research spike) |
| **4** | Docs cleanup | 31% volume reduction, remove `crap/`, archive completed plans | Low |
| **5** | Implement Moonshine (standalone or via mlx-audio) | Most interesting new implementation, ready-to-execute plan exists | Medium |
| **6** | Streaming benchmark prototype | Unexplored frontier, multiple docs point toward it | High |
| **7** | Git branch cleanup | 6 stale branches are pure noise | Trivial |

---

## Meta-Observations

**Project evolution pattern:** The QMD collection revealed that this project evolved through three distinct phases — (1) implementation exploration & documentation (Jan 7-14), (2) infrastructure & upgrades (Jan 14-26), (3) new ASR research (Jan 26-28). The natural next phase is consolidation before the next expansion.

**Documentation as institutional memory:** The 64-document collection at 21,500 lines is unusually rich for a benchmarking tool. The documents aren't just notes — they're an institutional knowledge base about ASR on Apple Silicon that has independent value beyond the tool itself.

**The unformalized pain point:** The `feat_model-handling-issues.md` file sitting untracked in the repo root (not in `docs/`) was written as a quick brain dump and never formalized — but it articulates the single most important problem in the project. It deserves to be the next feature plan.

**Search methodology note:** QMD's hybrid search (lex + vec combined) was particularly effective here because the project uses inconsistent terminology across docs — "P0 bug" in some places, "timeout" in others, "first-run download always fails" in yet another. Lexical search alone would have missed the connections; semantic search alone would have missed the exact quotes. The combination surfaced the "model download reliability" signal across 6 different documents that each described the same underlying problem differently.
