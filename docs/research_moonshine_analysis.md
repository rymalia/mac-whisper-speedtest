# Moonshine Codebase Analysis

**Date:** 2026-01-28
**Purpose:** Comparative analysis of Moonshine vs mac-whisper-speedtest, focused on ONNX runtime, streaming STT architecture, and integration planning

---

## Executive Summary

**Moonshine** is a family of speech-to-text models optimized for **fast, on-device ASR**. Unlike Whisper which processes fixed 30-second chunks, Moonshine scales compute with input length—processing audio **5-15x faster** while achieving similar or better accuracy.

**Key Finding:** Moonshine is an excellent candidate for our 10th implementation. It offers:
- Native ONNX support (cross-platform, lightweight)
- Significantly faster inference than Whisper on short audio
- A well-documented streaming architecture we can learn from
- Multiple language variants with the same interface

**Primary integration path:** `moonshine-onnx` (the ONNX runtime package)

---

## Project Structure

```
/Users/rymalia/projects/moonshine/
├── moonshine/                          # Keras implementation (deprecated for non-English)
│   ├── model.py                        # Core architecture (encoder/decoder/RoPE)
│   ├── transcribe.py                   # High-level API
│   └── tools/convert_to_onnx.py        # Keras → ONNX conversion
│
├── moonshine-onnx/                     # ONNX runtime implementation (RECOMMENDED)
│   ├── src/
│   │   ├── model.py                    # MoonshineOnnxModel class
│   │   ├── transcribe.py               # transcribe() API
│   │   └── assets/tokenizer.json       # Shared tokenizer (1.9MB)
│   └── requirements.txt                # [onnxruntime, huggingface_hub, librosa, tokenizers]
│
├── demo/
│   ├── moonshine-onnx/
│   │   ├── live_captions.py            # Microphone + VAD streaming demo
│   │   └── live_captions_web.py        # FastRTC/Gradio WebRTC demo
│   └── moonshine-web/                  # Browser-based ONNX Runtime Web demo
│
└── README.md                           # Main documentation
```

---

## Architecture Comparison

### mac-whisper-speedtest (Our Project)

```
┌─────────────────────────────────────────────────────────────┐
│                    WhisperImplementation (ABC)               │
│  - load_model(model_name)                                    │
│  - async transcribe(audio) → TranscriptionResult             │
│  - get_params() → dict                                       │
└─────────────────────────────────────────────────────────────┘
                              │
     ┌────────────┬───────────┼───────────┬────────────┐
     ▼            ▼           ▼           ▼            ▼
┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ faster- │ │mlx-whisper│ │ whisper │ │parakeet-│ │lightning│
│ whisper │ │  (lib)   │ │  kit    │ │   mlx   │ │ whisper │
└─────────┘ └──────────┘ └─────────┘ └─────────┘ └─────────┘
   ▲            ▲            ▲            ▲            ▲
   │            │            │            │            │
 [3rd party libraries - each with its own API]
```

**Pattern:** Adapter wrappers around different libraries

### Moonshine Architecture

```
Audio (16kHz) → Preprocessor → Encoder → Decoder → Tokens → Text
                 (Conv1D)     (RoPE)    (KV cache)   (tokenizers)

┌─────────────────────────────────────────────────────────────┐
│ AudioPreprocessor (Conv1D stack)                             │
│   Conv1D(dim, k=127, s=64) → Tanh → GroupNorm →              │
│   Conv1D(2×dim, k=7, s=3) → GELU →                           │
│   Conv1D(dim, k=3, s=2) → GELU                               │
│   Total downsampling: 64 × 3 × 2 = 384x                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Encoder (Transformer with RoPE)                              │
│   - n_layers: 6 (tiny) or 8 (base)                           │
│   - Rotary position embeddings                               │
│   - Feed-forward: GELU (encoder) vs SwiGLU (decoder)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Decoder (Autoregressive with KV cache)                       │
│   - Causal self-attention + Cross-attention to encoder       │
│   - KV cache for efficient token-by-token generation         │
│   - uncached_call (first token) vs cached_call (rest)        │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** Unlike Whisper's fixed 30-second chunks, Moonshine processes audio proportionally to its length. This is why short audio is so much faster.

---

## Proportional Compute: Why Moonshine Is Faster

This section explains the fundamental architectural difference that makes Moonshine 5-15x faster than Whisper on short audio.

### Fixed Compute (Whisper)

Whisper was designed around a **fixed 30-second context window**. Regardless of input length:

```
Any audio input → Pad/chunk to 30 seconds → Process full 30-second window
```

Even if you give Whisper 1 second of audio, it:
1. Pads the audio to 30 seconds (with silence)
2. Computes a fixed-size mel spectrogram: `[80, 3000]` (80 mel bins × 3000 time frames)
3. Runs the full encoder on all 30 seconds worth of features
4. Processes the entire padded input through the model

```python
# Whisper's approach - audio is always padded/trimmed to 30 seconds
mel = whisper.log_mel_spectrogram(audio, n_mels=80)  # Fixed size: [80, 3000]
```

### Proportional Compute (Moonshine)

Moonshine processes **only the audio you give it**. The compute cost scales linearly with input length:

```
1 second audio  → Process 1 second  → Fast
5 second audio  → Process 5 seconds → Medium
30 second audio → Process 30 seconds → Same as Whisper
```

This is achieved through strided convolutions that preserve proportionality:

```python
# Moonshine's audio preprocessor - output size proportional to input
# Conv1D(k=127, s=64) → Conv1D(k=7, s=3) → Conv1D(k=3, s=2)
# Total stride: 64 × 3 × 2 = 384
# Output length = input_samples / 384 (proportional to input!)
```

### The Math

If processing 30 seconds takes ~500ms:

| Audio Length | Whisper | Moonshine | Speedup |
|--------------|---------|-----------|---------|
| 1 second | ~500ms (full 30s processing) | ~500ms × (1/30) ≈ **17ms** | **~30x** |
| 5 seconds | ~500ms (full 30s processing) | ~500ms × (5/30) ≈ **83ms** | **~6x** |
| 10 seconds | ~500ms (full 30s processing) | ~500ms × (10/30) ≈ **167ms** | **~3x** |
| 30 seconds | ~500ms | ~500ms | **~1x** |

**Key insight:** Moonshine's "5-15x faster" claim specifically applies to **short audio segments**, which is exactly what streaming transcription requires.

### Why Whisper Uses Fixed Windows

Whisper was trained on 30-second chunks from a massive dataset. The model architecture—particularly its **absolute position embeddings**—expects this fixed context size. The positions are learned during training and don't generalize to different lengths without modification.

### Why Moonshine Handles Variable Lengths

Moonshine uses **Rotary Position Embeddings (RoPE)** instead of absolute position embeddings:

```python
class RotaryEmbedding(Model):
    def __init__(self, dim, base=10000):
        # inv_freq computed from dimension, not sequence length
        self.inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))

    def call(self, seq_len):
        # Position embeddings generated dynamically for any sequence length
        freqs = einsum("i, j -> i j", arange(seq_len), self.inv_freq)
        return stack((freqs, freqs), axis=-1)
```

RoPE encodes position information as rotations in the embedding space, which naturally handles variable-length sequences without retraining. This architectural choice is what enables proportional compute.

### Implications for Streaming

For streaming transcription, you're constantly processing short audio chunks (1-5 seconds). The compute difference is dramatic:

| Scenario | Whisper Latency | Moonshine Latency |
|----------|-----------------|-------------------|
| Process 1s chunk every 1s | ~500ms (too slow!) | ~17ms (real-time) |
| Process 2s chunk every 2s | ~500ms (too slow!) | ~33ms (real-time) |
| Process 5s chunk every 5s | ~500ms (borderline) | ~83ms (comfortable) |

This is why Moonshine is particularly well-suited for real-time streaming applications.

---

## Model Variants

| Model | Parameters | Languages | Token Rate | Use Case |
|-------|-----------|-----------|------------|----------|
| `moonshine/tiny` | 27M | English | 6/sec | Fast, lightweight |
| `moonshine/base` | 62M | English | 6/sec | Higher accuracy |
| `moonshine/tiny-ar` | 27M | Arabic | 13/sec | Non-English |
| `moonshine/tiny-zh` | 27M | Chinese | 13/sec | Non-English |
| `moonshine/tiny-ja` | 27M | Japanese | 13/sec | Non-English |
| `moonshine/tiny-ko` | 27M | Korean | 13/sec | Non-English |
| `moonshine/tiny-uk` | 27M | Ukrainian | 8/sec | Non-English |
| `moonshine/tiny-vi` | 27M | Vietnamese | 13/sec | Non-English |
| `moonshine/base-es` | 62M | Spanish | 6/sec | Non-English |

**Note:** Non-English models have higher token rates due to different tokenization densities.

### Architecture Details

| Model | dim | inner_dim | heads | enc_layers | dec_layers |
|-------|-----|-----------|-------|------------|------------|
| tiny | 288 | 288 | 8 | 6 | 6 |
| base | 416 | 416 | 8 | 8 | 8 |

---

## ONNX vs MLX Comparison

This is a comparison of Moonshine's ONNX approach vs MLX-based implementations (like mlx-whisper, mlx-audio).

### Runtime Philosophy

| Aspect | ONNX (Moonshine) | MLX (Apple Silicon) |
|--------|-----------------|---------------------|
| **Target hardware** | Any (CPU, CUDA, CoreML, DirectML) | Apple Silicon only |
| **Execution** | Compiled graph, no Python overhead | Lazy evaluation, just-in-time |
| **Model format** | Static `.onnx` files | Dynamic Python objects |
| **Memory** | Pre-allocated, fixed graph | Unified memory, dynamic allocation |
| **Quantization** | Separate model files | Built-in `nn.quantize()` |
| **Distribution** | HuggingFace + ONNX files | HuggingFace + SafeTensors |

### Inference Pipeline

**ONNX (Moonshine):**
```python
class MoonshineOnnxModel:
    def __init__(self, model_name):
        self.encoder = onnxruntime.InferenceSession(encoder_path)
        self.decoder = onnxruntime.InferenceSession(decoder_path)

    def generate(self, audio):
        # Encoder: single pass
        last_hidden_state = self.encoder.run(None, {"input_values": audio})[0]

        # Decoder: autoregressive loop with KV cache
        for i in range(max_len):
            logits, *present_kv = self.decoder.run(None, {
                "input_ids": [[next_token]],
                "encoder_hidden_states": last_hidden_state,
                "use_cache_branch": [i > 0],
                **past_key_values
            })
            next_token = logits[0, -1].argmax()
```

**MLX (typical pattern):**
```python
class MLXModel(nn.Module):
    def generate(self, audio):
        # Encoder
        hidden = self.encoder(audio)

        # Decoder with MLX's lazy evaluation
        cache = KVCache()
        for _ in range(max_len):
            logits = self.decoder(tokens, hidden, cache)
            next_token = mx.argmax(logits[-1])
            mx.eval(next_token)  # Force evaluation
```

### Performance Characteristics

| Metric | ONNX | MLX |
|--------|------|-----|
| **First inference** | Slow (graph optimization) | Slow (JIT compilation) |
| **Subsequent inference** | Fast (optimized graph) | Fast (cached kernels) |
| **Memory efficiency** | Pre-allocated, predictable | Dynamic, unified memory |
| **Cross-platform** | ✅ Excellent | ❌ Apple only |
| **GPU utilization** | Via execution providers | Native Metal |

### Why ONNX for Moonshine Integration

1. **Cross-platform consistency** — Same code runs on any hardware
2. **No additional dependencies** — `onnxruntime` is lightweight (~30MB)
3. **Mature ecosystem** — Extensive tooling, optimization, debugging
4. **Reference implementation** — Moonshine team recommends ONNX for production
5. **CoreML fallback** — ONNX Runtime can use CoreML on macOS

---

## Streaming STT Architecture (Deep Dive)

This section documents how Moonshine's `live_captions.py` achieves real-time streaming transcription. This architecture could be adapted for mac-whisper-speedtest.

### Core Pipeline

```
Microphone → Audio Chunks (512 samples) → VAD → Speech Buffer → Moonshine → Text
                   ↓                        ↓
            sounddevice.InputStream    Silero VAD
            (callback → Queue)         (detects speech start/end)
```

### Critical Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHUNK_SIZE` | 512 | Silero VAD requires exactly 512 samples at 16kHz (~32ms) |
| `LOOKBACK_CHUNKS` | 5 | Keep 5 chunks (160ms) before VAD triggers to capture word beginnings |
| `MIN_REFRESH_SECS` | 0.2 | How often to re-transcribe during active speech for "live" feel |
| `MAX_SPEECH_SECS` | 15 | Force-end recording to prevent hallucination on long segments |

### Why Streaming Works (and What Makes It Different from Batch)

**Batch transcription (our current approach):**
```
Record complete audio → Load model → Transcribe all at once → Return text
```

**Streaming transcription (Moonshine demo):**
```
Start listening → Detect speech start → Buffer audio →
  → Periodic re-transcription (every 200ms) →
  → Detect speech end → Final transcription → Display
```

**Key differences:**

1. **VAD-driven segmentation** — Speech is automatically segmented by silence detection, not fixed windows
2. **Incremental updates** — Users see partial transcription as they speak
3. **Lookback buffer** — Captures audio *before* VAD triggers (prevents losing word beginnings)
4. **Forced truncation** — Long segments (>15s) are split to prevent hallucination

### VAD Integration Pattern

Moonshine uses **Silero VAD** (a separate, lightweight voice activity detector):

```python
from silero_vad import VADIterator, load_silero_vad

vad_model = load_silero_vad(onnx=True)  # ONNX is 4-5x faster on CPU
vad_iterator = VADIterator(
    model=vad_model,
    sampling_rate=16000,
    threshold=0.5,              # Speech detection sensitivity
    min_silence_duration_ms=300  # Silence before "end" triggers
)

# Returns {"start": sample_idx} or {"end": sample_idx} or empty dict
speech_dict = vad_iterator(chunk)
```

**Hysteresis logic:**
- Speech **starts** when probability > `threshold`
- Speech **ends** when probability < `threshold - 0.15` for `min_silence_duration_ms`
- The 0.15 gap prevents rapid on/off flickering

### Lookback Buffer Pattern

VAD triggers *after* speech starts, so you lose the first phonemes. The lookback buffer solves this:

```python
lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE  # 2560 samples = 160ms

speech = np.concatenate((speech, chunk))
if not recording:
    speech = speech[-lookback_size:]  # Keep only recent audio when idle
```

When VAD triggers "start", the buffer already contains ~160ms of audio *before* the trigger point.

### Two-Mode Transcription Strategy

1. **Periodic refresh** (during active speech):
   ```python
   if recording and (time.time() - start_time) > MIN_REFRESH_SECS:
       print_captions(transcribe(speech))  # Interim result
       start_time = time.time()
   ```
   Note: Partial words cause unpredictable transcriptions (expected behavior).

2. **End-of-utterance** (when VAD detects silence):
   ```python
   if "end" in speech_dict:
       caption_cache.append(transcribe(speech))  # Final, reliable result
   ```

### Soft Reset for Forced Truncation

When `MAX_SPEECH_SECS` is exceeded, you must reset VAD state without affecting model weights:

```python
def soft_reset(vad_iterator):
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.current_sample = 0
```

This allows continuation of the audio stream without reloading the VAD model.

### Can Other Models Stream?

**Q: Could we do streaming with Whisper models?**

**A: Yes, with caveats.**

| Model Type | Streaming Capability | Notes |
|------------|---------------------|-------|
| **Moonshine** | ✅ Native | Scales with input length, fast on short segments |
| **Whisper (any variant)** | ⚠️ Inefficient | Fixed 30-second processing regardless of input length |
| **VibeVoice** | ✅ Has `stream_transcribe()` | LLM-based, outputs tokens progressively |
| **faster-whisper** | ⚠️ Possible | Uses CTranslate2, supports chunked processing |

**Why Moonshine is better for streaming:**
- **Proportional compute:** 1 second of audio = fast; 30 seconds = slower but proportional
- **Whisper's problem:** Even 1 second of audio triggers 30-second chunk processing overhead

**Practical streaming for Whisper:**
You can still build streaming UIs with Whisper by:
1. Using VAD to segment speech
2. Transcribing each segment independently
3. Concatenating results

But the latency will be higher than Moonshine's approach.

### Hallucination Mitigation

Moonshine (like all ASR models) can hallucinate in these scenarios:
- **Truncated words** — Partial audio at segment boundaries
- **Long segments** — >15 seconds without pause
- **Background noise** — Non-speech interpreted as speech

**Mitigations in `live_captions.py`:**
1. Lookback buffer prevents word-start truncation
2. `MAX_SPEECH_SECS` forces segmentation
3. VAD filters non-speech (threshold=0.5)
4. Final cached transcriptions (at utterance end) are more reliable than interim ones

---

## Comparison with Our Implementations

| Feature | Moonshine ONNX | Our Implementations |
|---------|---------------|---------------------|
| **Runtime** | ONNX Runtime | Varies (MLX, CoreML, PyTorch, etc.) |
| **Sample rate** | 16kHz (strict) | 16kHz (converted if needed) |
| **Max duration** | 64 seconds | Varies by implementation |
| **Segments** | No (text only) | Yes (with timestamps) |
| **Language** | 8 languages | English (most models) |
| **Streaming** | Demo available | Not implemented |

### Output Format Comparison

**Our TranscriptionResult:**
```python
@dataclass
class TranscriptionResult:
    text: str
    segments: List = field(default_factory=list)  # [{start, end, text}]
    language: Optional[str] = None
```

**Moonshine output:**
```python
# Just returns list of strings
["Ever tried ever failed, no matter try again, fail again, fail better."]
```

**Note:** Moonshine doesn't return timestamps/segments. We'd need to adapt our `TranscriptionResult` or leave segments empty.

---

## Performance Benchmarks

From Moonshine's documentation and papers:

### Speed Comparison (vs Whisper)

| Audio Length | Whisper | Moonshine | Speedup |
|-------------|---------|-----------|---------|
| 1 second | ~500ms | ~30ms | ~15x |
| 5 seconds | ~500ms | ~100ms | ~5x |
| 10 seconds | ~500ms | ~200ms | ~2.5x |
| 30 seconds | ~500ms | ~500ms | ~1x |

**Key insight:** Moonshine's advantage is on **short audio**. For 30-second chunks, performance is similar.

### Accuracy (Word Error Rate)

| Model | English WER | Params |
|-------|------------|--------|
| Moonshine tiny | 12.66% | 27M |
| Moonshine base | 10.07% | 62M |
| Whisper tiny | 12.81% | 39M |
| Whisper base | 10.32% | 74M |

**Moonshine achieves similar or better accuracy with fewer parameters.**

---

## Integration Plan Summary

### Recommended Approach

Create `MoonshineOnnxImplementation` that:
1. Uses `moonshine-onnx` package (ONNX runtime)
2. Maps model names to Moonshine variants
3. Returns `TranscriptionResult` with empty segments (no timestamp support)
4. Supports both English and non-English variants

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `implementations/moonshine_onnx.py` | CREATE | New implementation |
| `implementations/__init__.py` | MODIFY | Register implementation |
| `pyproject.toml` | MODIFY | Add `useful-moonshine-onnx` dependency |
| `tests/test_moonshine_integration.py` | CREATE | Integration tests |

### Model Name Mapping

```python
MODEL_MAP = {
    # English
    "moonshine-tiny": "moonshine/tiny",
    "moonshine-base": "moonshine/base",
    # Non-English (explicit language codes)
    "moonshine-tiny-ar": "moonshine/tiny-ar",  # Arabic
    "moonshine-tiny-zh": "moonshine/tiny-zh",  # Chinese
    "moonshine-tiny-ja": "moonshine/tiny-ja",  # Japanese
    "moonshine-tiny-ko": "moonshine/tiny-ko",  # Korean
    "moonshine-tiny-uk": "moonshine/tiny-uk",  # Ukrainian
    "moonshine-tiny-vi": "moonshine/tiny-vi",  # Vietnamese
    "moonshine-base-es": "moonshine/base-es",  # Spanish
}
```

---

## Future Streaming Enhancement

Based on this research, here's how we could add streaming to mac-whisper-speedtest:

### Phase 1: Research Complete ✅
- Understand Moonshine's streaming architecture
- Document VAD integration patterns
- Identify dependencies (silero-vad, sounddevice)

### Phase 2: Streaming Infrastructure (Future)
1. Add `StreamingTranscriptionResult` dataclass with partial/final flags
2. Create `StreamingWhisperImplementation` base class
3. Implement `MoonshineStreamingImplementation`
4. Add CLI mode for streaming benchmark (`--stream`)

### Phase 3: Expand to Other Models (Future)
1. Test streaming patterns with faster-whisper
2. Explore mlx-audio's streaming capabilities
3. Create unified streaming interface

---

## Appendix: Key File Locations

| Purpose | Path |
|---------|------|
| ONNX model class | `moonshine-onnx/src/model.py` |
| ONNX transcribe API | `moonshine-onnx/src/transcribe.py` |
| Keras model (reference) | `moonshine/model.py` |
| Live captions demo | `demo/moonshine-onnx/live_captions.py` |
| WebRTC demo | `demo/moonshine-onnx/live_captions_web.py` |
| Tokenizer | `moonshine-onnx/src/assets/tokenizer.json` |
| HuggingFace repo | `UsefulSensors/moonshine` |

---

## Summary Statistics

| Metric | Moonshine | mac-whisper-speedtest |
|--------|-----------|----------------------|
| Model variants | 9 (2 English + 7 language) | 9 implementations |
| Runtime | ONNX (cross-platform) | Mixed (MLX, CoreML, PyTorch) |
| Sample rate | 16kHz (strict) | 16kHz (flexible) |
| Max duration | 64 seconds | Varies |
| Streaming | Demo available | Not implemented |
| Output | Text only | Text + segments |
| Primary purpose | Fast on-device ASR | Benchmarking |
