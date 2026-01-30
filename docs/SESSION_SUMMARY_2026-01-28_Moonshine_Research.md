# Session Summary: Moonshine Research & Analysis

**Date:** 2026-01-28
**Purpose:** Research and document Moonshine ASR for potential integration into mac-whisper-speedtest

---

## Executive Summary

Completed a thorough research analysis of the Moonshine speech-to-text project, similar to the previous mlx-audio analysis. Created comprehensive documentation covering architecture, ONNX vs MLX comparison, streaming STT patterns, and an implementation plan for adding Moonshine as the 10th ASR implementation.

---

## Key Deliverables

### Files Created

| File | Purpose |
|------|---------|
| `docs/research_moonshine_analysis.md` | Comprehensive codebase analysis (~500 lines) |
| `docs/feature_plan_moonshine_implementation.md` | Implementation plan for MoonshineOnnxImplementation |
| `docs/SESSION_SUMMARY_2026-01-28_Moonshine_Research.md` | This file |

### Research Scope

1. **Moonshine codebase exploration** at `/Users/rymalia/projects/moonshine`
2. **ONNX vs MLX runtime comparison** (architectural and performance)
3. **Streaming STT deep dive** — how `live_captions.py` achieves real-time transcription
4. **Proportional compute explanation** — why Moonshine is 5-15x faster on short audio
5. **Implementation plan** for adding to mac-whisper-speedtest

---

## Key Findings

### Moonshine Overview

- **Family of ASR models** optimized for fast, on-device transcription
- **9 model variants**: 2 English (tiny/base) + 7 language-specific
- **ONNX-first approach**: Cross-platform, lightweight deployment
- **Processes audio 5-15x faster** than Whisper on short segments

### Proportional Compute (Key Insight)

| Audio Length | Whisper | Moonshine | Why |
|--------------|---------|-----------|-----|
| 1 second | ~500ms | ~17ms | Whisper pads to 30s; Moonshine processes only input |
| 5 seconds | ~500ms | ~83ms | Compute scales linearly with input length |
| 30 seconds | ~500ms | ~500ms | Same performance at full window |

**Technical reason:** Moonshine uses **RoPE (Rotary Position Embeddings)** which handle variable-length sequences, while Whisper uses **absolute position embeddings** trained on fixed 30-second windows.

### Streaming Architecture

Moonshine's `live_captions.py` demonstrates a well-designed streaming pattern:

1. **Silero VAD** for voice activity detection (512-sample chunks)
2. **Lookback buffer** (160ms) to capture audio before VAD triggers
3. **Periodic re-transcription** (every 200ms) for "live" feel
4. **Forced truncation** (15s max) to prevent hallucination

**Can other models stream?** Yes, but less efficiently. Whisper's fixed 30-second processing makes it poorly suited for real-time streaming on short chunks.

### ONNX vs MLX Comparison

| Aspect | ONNX (Moonshine) | MLX (Apple Silicon) |
|--------|-----------------|---------------------|
| Target hardware | Any platform | Apple Silicon only |
| Model format | Static `.onnx` files | Dynamic Python objects |
| Cross-platform | ✅ Excellent | ❌ Apple only |
| Position embeddings | RoPE (variable length) | Varies by model |

---

## Implementation Plan Summary

### Recommended Approach

Create `MoonshineOnnxImplementation` as the 10th implementation:
- Use `useful-moonshine-onnx` package (ONNX Runtime)
- Map model names: `"tiny"` → `"moonshine/tiny"`
- Return `TranscriptionResult` with empty segments (Moonshine doesn't provide timestamps)

### Files to Create/Modify

| File | Action |
|------|--------|
| `implementations/moonshine_onnx.py` | CREATE |
| `implementations/__init__.py` | MODIFY |
| `pyproject.toml` | MODIFY (add dependency) |
| `tests/test_moonshine_integration.py` | CREATE |

### Dependencies

```toml
"useful-moonshine-onnx>=0.1.0"
```

Brings: `onnxruntime`, `huggingface_hub`, `librosa`, `tokenizers`

---

## Unfinished Work / Next Steps

### Ready to Implement (When Ready)

1. **Add Moonshine implementation** — Follow `feature_plan_moonshine_implementation.md`
2. **Add VibeVoice implementation** — Follow `feature_plan_vibevoice_implementation.md` (from previous session)

### Future Enhancements

1. **Streaming mode** — Add `--stream` CLI option using Moonshine's streaming patterns
2. **Non-English benchmarks** — Test multilingual Moonshine variants
3. **MLX backend** — Add native MLX implementation if performance benefits justify it

---

## External References

| Resource | Location |
|----------|----------|
| Moonshine local repo | `/Users/rymalia/projects/moonshine` |
| Moonshine GitHub | `https://github.com/moonshine-ai/moonshine` |
| HuggingFace models | `UsefulSensors/moonshine` |
| Live captions demo | `moonshine/demo/moonshine-onnx/live_captions.py` |

---

## Session Statistics

- **Duration:** ~1 session
- **Files created:** 3 (2 research docs + 1 session summary)
- **Research scope:** Complete codebase analysis of Moonshine project
- **Implementation status:** Planning complete, ready for implementation
