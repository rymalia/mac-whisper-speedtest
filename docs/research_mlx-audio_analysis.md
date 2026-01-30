# mlx-audio Codebase Analysis

**Date:** 2026-01-26
**Purpose:** Comparative analysis of mlx-audio vs mac-whisper-speedtest, focused on STT features and actionable insights

---

## Executive Summary

**mlx-audio** is a **production-grade, native MLX implementation** of audio models (STT, TTS, STS). It's fundamentally different from mac-whisper-speedtest: we *wrap* existing libraries for benchmarking, while mlx-audio *implements* the models directly in MLX.

**Key Finding:** mlx-audio could replace 3 of our 9 implementations (MLXWhisper, ParakeetMLX, LightningWhisperMLX) with a unified interface, but it's not a drop-in replacement—it's an alternative approach to the problem.

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

### mlx-audio

```
┌─────────────────────────────────────────────────────────────┐
│                    base_load_model()                         │
│  1. Download from HuggingFace (snapshot_download)            │
│  2. Load config.json                                         │
│  3. Dynamic import: mlx_audio.stt.models.{model_type}        │
│  4. ModelConfig.from_dict(config)                            │
│  5. Load & sanitize weights                                  │
│  6. Apply quantization (if specified)                        │
│  7. model.generate(audio) → STTOutput                        │
└─────────────────────────────────────────────────────────────┘
                              │
     ┌────────────┬───────────┼───────────┬────────────┐
     ▼            ▼           ▼           ▼            ▼
┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ whisper │ │ parakeet │ │vibevoice│ │ voxtral │ │ wav2vec │
│ (native)│ │ (native) │ │ (native)│ │ (native)│ │ (native)│
└─────────┘ └──────────┘ └─────────┘ └─────────┘ └─────────┘
   ▲            ▲            ▲            ▲            ▲
   │            │            │            │            │
 [ALL implemented directly in MLX - no external deps]
```

**Pattern:** Unified loader with native implementations

---

## What Would mlx-audio Replace?

| Our Implementation | Could Use mlx-audio? | Notes |
|-------------------|---------------------|-------|
| MLXWhisperImplementation | ✅ Yes | mlx-audio has native Whisper |
| ParakeetMLXImplementation | ✅ Yes | mlx-audio has native Parakeet (TDT/RNNT/CTC) |
| LightningWhisperMLXImplementation | ✅ Yes | mlx-audio Whisper covers this |
| FasterWhisperImplementation | ❌ No | CTranslate2 backend, not MLX |
| WhisperCppCoreMLImplementation | ❌ No | whisper.cpp + CoreML, different stack |
| InsanelyFastWhisperImplementation | ❌ No | PyTorch/Transformers pipeline |
| FluidAudioCoreMLImplementation | ❌ No | CoreML native, not MLX |
| WhisperKitImplementation | ❌ No | Swift/CoreML native |
| WhisperMPSImplementation | ❌ No | PyTorch MPS backend |

**Verdict:** 3 of 9 implementations could be consolidated into mlx-audio.

---

## Output Format Comparison

### Our TranscriptionResult
```python
@dataclass
class TranscriptionResult:
    text: str
    segments: List = field(default_factory=list)
    language: Optional[str] = None
```

### mlx-audio STTOutput
```python
@dataclass
class STTOutput:
    text: str
    segments: List[dict] = None
    language: str = None
    prompt_tokens: int = 0           # ← NEW: Token metrics
    generation_tokens: int = 0       # ← NEW
    total_tokens: int = 0            # ← NEW
    prompt_tps: float = 0.0          # ← NEW: Tokens per second
    generation_tps: float = 0.0      # ← NEW
    total_time: float = 0.0          # ← NEW: Internal timing
```

**Insight:** mlx-audio captures LLM-style metrics (tokens per second) because some models like VibeVoice are LLM-based. This could be useful for benchmarking.

---

## Model Integration Pattern Comparison

### Our Approach: Per-Implementation Custom Code

Each implementation in `implementations/` has its own:
- Model loading logic
- Name mapping (e.g., "large" → "whisper-large-v3")
- Audio preprocessing
- Error handling

```python
# implementations/__init__.py - Manual registration
if importlib.util.find_spec("pywhispercpp"):
    from ...coreml import WhisperCppCoreMLImplementation
    available_implementations.append(WhisperCppCoreMLImplementation)
```

### mlx-audio Approach: Config-Driven Dynamic Loading

The core idea: **models describe themselves via `config.json`**, and a single loader function handles everything.

#### The Loading Flow

```
User calls: load("mlx-community/whisper-large-v3-turbo-asr-fp16")
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. RESOLVE PATH                                                  │
│    - Is it local? Use directly                                   │
│    - Is it HuggingFace repo? → snapshot_download()               │
│    Result: /Users/.../hub/models--mlx-community--whisper-.../    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. READ config.json                                              │
│    {                                                             │
│      "model_type": "whisper",        ← Determines which module   │
│      "n_mels": 128,                                              │
│      "n_vocab": 51866,                                           │
│      "quantization": {"bits": 4, "group_size": 64}  ← Optional   │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DYNAMIC IMPORT                                                │
│    model_type = "whisper"                                        │
│    arch = importlib.import_module("mlx_audio.stt.models.whisper")│
│                                                                  │
│    This gives us access to:                                      │
│    - arch.Model (the nn.Module class)                            │
│    - arch.ModelConfig (dataclass for config)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. BUILD MODEL                                                   │
│    model_config = arch.ModelConfig.from_dict(config)             │
│    model = arch.Model(model_config)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. LOAD WEIGHTS (with optional hooks)                            │
│    weights = mx.load("model.safetensors")                        │
│                                                                  │
│    if model.sanitize:        # Optional: rename weight keys      │
│        weights = model.sanitize(weights)                         │
│                                                                  │
│    if config.quantization:   # Optional: quantize layers         │
│        nn.quantize(model, bits=4, ...)                           │
│                                                                  │
│    model.load_weights(weights)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. POST-LOAD HOOK (optional)                                     │
│    if model.post_load_hook:                                      │
│        model = model.post_load_hook(model, model_path)           │
│    # e.g., initialize KV cache, load tokenizer, etc.             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        Ready to use!
              model.generate(audio) → STTOutput
```

#### The Key Code (from `mlx_audio/utils.py:316-404`)

```python
def base_load_model(model_path, category, model_remapping, lazy=False, strict=False, **kwargs):
    # Step 1: Resolve to local path (download if needed)
    if isinstance(model_path, str):
        model_path = get_model_path(model_path, ...)

    # Step 2: Load config
    config = load_config(model_path)

    # Step 3: Determine model type and import dynamically
    model_type = config.get("model_type", None)
    model_class, model_type = get_model_class(model_type, ...)
    #           ↑ This does: importlib.import_module(f"mlx_audio.{category}.models.{model_type}")

    # Step 4: Build model from config
    model_config = model_class.ModelConfig.from_dict(config)
    model = model_class.Model(model_config)

    # Step 5: Load weights with hooks
    weights = load_weights(model_path)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    apply_quantization(model, config, weights, ...)
    model.load_weights(list(weights.items()), strict=strict)

    # Step 6: Post-load hook
    if hasattr(model_class.Model, "post_load_hook"):
        model = model_class.Model.post_load_hook(model, model_path)

    return model
```

#### What Each Model Must Provide

Each model in `mlx_audio/stt/models/{model_type}/` exports:

```python
# __init__.py
from .whisper import Model, ModelConfig

# whisper.py
@dataclass
class ModelConfig:
    n_mels: int = 128
    n_vocab: int = 51866
    # ... other params

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        self.encoder = AudioEncoder(config.n_mels, ...)

    def generate(self, audio, **kwargs) -> STTOutput:
        # Transcription logic
        ...

    # Optional hooks
    def sanitize(self, weights: dict) -> dict:
        return {k.replace("model.", ""): v for k, v in weights.items()}

    @staticmethod
    def post_load_hook(model, model_path):
        model.tokenizer = load_tokenizer(model_path)
        return model
```

#### Why This Is Powerful

Adding a new model requires **zero changes to the loading code**:
1. Create `mlx_audio/stt/models/newmodel/`
2. Implement `Model` and `ModelConfig`
3. Upload weights + `config.json` to HuggingFace

The loader automatically discovers and loads it based on `"model_type": "newmodel"` in config.json.

#### Why We Can't Fully Adopt This

Our implementations wrap **external libraries** with different APIs:
- `mlx_whisper.load_model()`
- `parakeet_mlx.from_pretrained()`
- `faster_whisper.WhisperModel()`
- subprocess calls to Swift binaries (WhisperKit)

mlx-audio can use config-driven loading because they **own all the implementations**. Every model follows the same `Model`/`ModelConfig` pattern.

**Key Difference:** mlx-audio models are self-describing via `config.json`. The loader doesn't need to know implementation details.

---

## Features mlx-audio Has That We Don't

### 1. Streaming Transcription
```python
for result in model.generate_streaming(audio, chunk_duration=1.0):
    print(result.text)  # Real-time output as audio is processed
```
We currently process entire audio files at once.

### 2. Multiple Output Formats
```bash
mlx_audio.stt.generate --format srt   # Subtitles with timestamps
mlx_audio.stt.generate --format vtt   # WebVTT
mlx_audio.stt.generate --format json  # Structured data
```
We only output text + segments.

### 3. Built-in Quantization Support
```python
# Specified in config.json, automatically applied on load
"quantization": {"group_size": 64, "bits": 4}
```
We don't have quantization controls.

### 4. Speaker Diarization (VibeVoice)
```python
# VibeVoice model supports speaker identification
result = model.generate(audio)
# result.segments includes speaker IDs
```

### 5. Context/Hotword Injection
```bash
mlx_audio.stt.generate --context "technical terms, acronyms"
```
Improves accuracy for domain-specific vocabulary.

### 6. OpenAI-Compatible API Server
```bash
mlx_audio.server --port 8000
curl -X POST http://localhost:8000/v1/audio/transcriptions ...
```
Drop-in replacement for OpenAI's API.

---

## What mlx-audio Does Well (Patterns Worth Adopting)

### 1. Unified Model Loading
**Problem in our code:** Each implementation has bespoke loading logic.

**mlx-audio solution:**
```python
# stt/utils.py - Simple wrapper
def load(model_path, **kwargs):
    return base_load_model(model_path, category="stt", ...)

# Usage
model = load("mlx-community/whisper-large-v3-turbo-asr-fp16")
```

**Actionable insight:** We could create a similar pattern for MLX-based implementations.

### 2. Model Lifecycle Hooks
```python
class Model(nn.Module):
    def sanitize(weights):      # Transform weight names during load
        ...

    def model_quant_predicate(path, module):  # Control quantization
        ...

    def post_load_hook(model, path):  # Post-processing
        ...
```

**Actionable insight:** Extensibility points that models can optionally implement.

### 3. HuggingFace-First Model Distribution
All models live in `mlx-community/` on HuggingFace Hub:
- `mlx-community/whisper-large-v3-turbo-asr-fp16`
- `mlx-community/parakeet-tdt-0.6b-asr-fp16`

Auto-download on first use via `snapshot_download()`.

**Actionable insight:** We could cache models the same way instead of relying on each library's cache behavior.

### 4. Rich Timing Metrics
```python
@dataclass
class STTOutput:
    prompt_tps: float = 0.0      # Prefill tokens/sec
    generation_tps: float = 0.0  # Decode tokens/sec
    total_time: float = 0.0      # Wall clock time
```

**Actionable insight:** Our `BenchmarkResult` only captures `transcription_time`. We could add more granular metrics.

---

## What We Do That mlx-audio Doesn't

### 1. Cross-Library Benchmarking
We compare 9 fundamentally different libraries. mlx-audio only tests its own MLX implementations.

### 2. Non-MLX Implementations
We support CoreML-native (WhisperKit, FluidAudio), CPU-only (faster-whisper), and PyTorch (insanely-fast-whisper). mlx-audio is MLX-only.

### 3. Model Name Normalization
We handle the chaos of different naming conventions:
- `"large"` → `"whisper-large-v3"` (MLX)
- `"large"` → `"large-v3"` (WhisperKit)
- `"large"` → `"Systran/faster-whisper-large-v3"` (faster-whisper)

mlx-audio expects explicit HuggingFace repo IDs.

### 4. Interactive Recording Mode
Our benchmark can record from microphone. mlx-audio is file-only (though has stdin support).

---

## Potential Integration Paths

### Option A: Use mlx-audio as an Implementation
Create `MLXAudioImplementation` that wraps mlx-audio's unified loader:

```python
class MLXAudioImplementation(WhisperImplementation):
    def load_model(self, model_name):
        from mlx_audio.stt import load
        repo = self._map_model_name(model_name)  # "large" → "mlx-community/whisper-large-v3..."
        self.model = load(repo)

    async def transcribe(self, audio):
        result = self.model.generate(audio)
        return TranscriptionResult(text=result.text, ...)
```

**Pros:** Access to all mlx-audio models (Whisper, Parakeet, VibeVoice, etc.)
**Cons:** Another dependency; loses individual implementation comparison

### Option B: Adopt Their Patterns
Keep our architecture but adopt:
1. `STTOutput` with timing metrics → Add to `TranscriptionResult`
2. Model lifecycle hooks → Add `sanitize()`, `post_load_hook()` to base class
3. Unified config loading → Centralize HuggingFace download logic

**Pros:** Improves our code without changing purpose
**Cons:** Significant refactor

### Option C: Add New Models They Support
mlx-audio supports models we don't:
- **VibeVoice** (Microsoft's 9B LLM-based ASR with speaker diarization)
- **Voxtral** (Mistral's speech model)
- **GLM-ASR**

Could add `VibeVoiceImplementation` using their code as reference.

**Pros:** More models to benchmark
**Cons:** These are MLX-only, overlaps with existing MLX implementations

---

## Recommended Actions

### High Priority (Low Effort, High Value)

1. **Adopt `total_time` in STTOutput**
   - Add internal timing to `TranscriptionResult`
   - mlx-audio captures this; useful for detailed benchmarks

2. **Add streaming support flag**
   - Track which implementations support streaming
   - Future-proofs for real-time benchmarks

### Medium Priority

3. **Centralize HuggingFace cache logic**
   - Currently each implementation handles its own model downloads
   - mlx-audio's `get_model_path()` + `snapshot_download()` pattern is cleaner

4. **Consider output format options**
   - SRT/VTT export would make the tool more useful for practical evaluation

### Low Priority (Research)

5. **Evaluate adding VibeVoice**
   - 9B parameter LLM-based ASR with speaker diarization
   - Would be interesting to benchmark against Whisper variants

6. **Study their quantization approach**
   - Config-driven quantization is elegant
   - Could inform future optimization work

---

## Appendix: Key File Locations in mlx-audio

| Purpose | Path |
|---------|------|
| Unified model loader | `mlx_audio/utils.py` (lines 316-404) |
| STT utilities | `mlx_audio/stt/utils.py` |
| STTOutput dataclass | `mlx_audio/stt/models/base.py` |
| Whisper implementation | `mlx_audio/stt/models/whisper/whisper.py` |
| Parakeet implementation | `mlx_audio/stt/models/parakeet/parakeet.py` |
| VibeVoice (LLM-based) | `mlx_audio/stt/models/vibevoice_asr/vibevoice_asr.py` |
| CLI entry point | `mlx_audio/stt/generate.py` |
| API server | `mlx_audio/server.py` |

---

## Summary Statistics

| Metric | mac-whisper-speedtest | mlx-audio |
|--------|----------------------|-----------|
| STT implementations | 9 (6 libraries) | 6 (all native MLX) |
| Backend diversity | High (CoreML, MLX, PyTorch, CPU) | Low (MLX only) |
| Model loading | Per-implementation | Unified config-driven |
| Output format | Text + segments | Text + segments + timing + tokens |
| Streaming support | No | Yes (Whisper, Parakeet) |
| Quantization | Per-library | Built-in, config-driven |
| API server | No | Yes (OpenAI-compatible) |
| Primary purpose | Benchmarking | Production inference |

---

## Deep Dive: VibeVoice-ASR

VibeVoice is the most architecturally interesting model in mlx-audio—a **9B parameter LLM-based ASR** that fundamentally differs from Whisper-style encoder-decoder models.

### Architecture Overview

```
Raw Audio (24kHz)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│ AcousticTokenizer   │              │ SemanticTokenizer   │
│ (32M params)        │              │ (32M params)        │
│ Output: [B,T',64]   │              │ Output: [B,T',128]  │
└─────────────────────┘              └─────────────────────┘
       │                                      │
       ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│ AcousticConnector   │              │ SemanticConnector   │
│ MLP → [B,T',3584]   │              │ MLP → [B,T',3584]   │
└─────────────────────┘              └─────────────────────┘
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
            ┌─────────────────────┐
            │ Merge with Text     │
            │ Embeddings at       │
            │ <|box_start|> tokens│
            └─────────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ Qwen2 LM (7B)       │
            │ 28 layers, 3584 dim │
            │ Generates JSON      │
            └─────────────────────┘
                      │
                      ▼
            JSON with transcription,
            speaker IDs, timestamps
```

### What "LLM-based ASR" Means

Traditional ASR (Whisper, Parakeet): Audio → Encoder → Decoder → Text

VibeVoice: Audio → Tokenizers → Embeddings → **LLM generates structured JSON**

The Qwen2 LLM is prompted with a conversation template:
```
System: "You are a helpful assistant that transcribes audio..."
User: [speech embeddings inserted at special token positions]
      "This is a X seconds audio. Please transcribe with: Start time, End time, Speaker ID, Content"

Model generates:
[
  {"Start time": 0, "End time": 5.2, "Speaker ID": 0, "Content": "Hello everyone..."},
  {"Start time": 5.5, "End time": 9.8, "Speaker ID": 1, "Content": "Thank you..."}
]
```

### Key Technical Details

| Aspect | Value |
|--------|-------|
| **Sample Rate** | 24kHz (not 16kHz like Whisper) |
| **Max Duration** | 59 minutes |
| **Compression** | 3200 samples per token (~7.5ms) |
| **Model Size** | ~9B parameters total |
| **Speed** | ~100-200 tokens/sec on M1 Pro |

### Model Variants

| Model | Bits | Approx Size | Min RAM |
|-------|------|-------------|---------|
| `mlx-community/VibeVoice-ASR-4bit` | 4-bit | ~5 GB | 8 GB |
| `mlx-community/VibeVoice-ASR-5bit` | 5-bit | ~6 GB | 10 GB |
| `mlx-community/VibeVoice-ASR-6bit` | 6-bit | ~7 GB | **12 GB (default)** |
| `mlx-community/VibeVoice-ASR-8bit` | 8-bit | ~10 GB | 16 GB |
| `mlx-community/VibeVoice-ASR-bf16` | bf16 | ~18 GB | 32 GB |

### Special Features

1. **Speaker Diarization** — Built into the model, no separate processing
2. **Context Injection** — Pass domain terms to improve accuracy:
   ```python
   model.generate(audio, context="machine learning, neural networks")
   ```
3. **Streaming** — Token-by-token generation via `stream_transcribe()`
4. **Quantization** — Only LM layers quantized (encoders stay high-fidelity)

### Output Format

```python
result = model.generate(audio)

result.text        # Raw JSON string
result.segments    # Parsed list:
# [
#   {"start": 0.0, "end": 5.2, "speaker_id": 0, "text": "Hello everyone..."},
#   {"start": 5.5, "end": 9.8, "speaker_id": 1, "text": "Thank you..."},
# ]
result.total_time        # Wall clock time
result.generation_tps    # Tokens per second
```

---

## Planning: Add VibeVoiceImplementation

### Goal

Add VibeVoice as the **10th ASR implementation** in mac-whisper-speedtest, enabling benchmarking of LLM-based ASR against traditional Whisper-style models.

### Implementation Plan

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `implementations/vibevoice.py` | CREATE | New implementation class |
| `implementations/__init__.py` | MODIFY | Register in available_implementations |
| `pyproject.toml` | MODIFY | Add `mlx-audio` dependency |

#### Core Implementation

```python
# implementations/vibevoice.py
class VibeVoiceImplementation(WhisperImplementation):
    """VibeVoice ASR implementation using mlx-audio."""

    def __init__(self):
        self.log = structlog.get_logger(__name__)
        self._model = None
        if platform.system() != "Darwin":
            raise RuntimeError("VibeVoice requires macOS with Apple Silicon")

    def load_model(self, model_name: str) -> None:
        from mlx_audio.stt import load

        # Map explicit quantization variants
        model_map = {
            "vibevoice-4bit": "mlx-community/VibeVoice-ASR-4bit",
            "vibevoice-5bit": "mlx-community/VibeVoice-ASR-5bit",
            "vibevoice-6bit": "mlx-community/VibeVoice-ASR-6bit",
            "vibevoice-8bit": "mlx-community/VibeVoice-ASR-8bit",
            "vibevoice-bf16": "mlx-community/VibeVoice-ASR-bf16",
        }
        # Default to 6-bit (good balance of quality and memory)
        self._hf_repo = model_map.get(model_name, "mlx-community/VibeVoice-ASR-6bit")
        self.log.info(f"Loading VibeVoice from {self._hf_repo}")
        self._model = load(self._hf_repo)

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        # CRITICAL: Resample 16kHz → 24kHz
        audio_24k = self._resample(audio, 16000, 24000)

        result = self._model.generate(
            audio=audio_24k,
            max_tokens=8192,
            temperature=0.0
        )

        return TranscriptionResult(
            text=result.text,
            segments=result.segments,  # Includes speaker_id
            language=result.language
        )

    def _resample(self, audio, orig_sr, target_sr):
        from scipy.signal import resample_poly
        gcd = np.gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // gcd, orig_sr // gcd)
```

#### Sample Rate Handling

**Critical issue:** Our benchmark uses 16kHz audio; VibeVoice requires 24kHz.

**Solution:** Resample in the implementation (scipy already a dependency via mlx-audio).

#### Model Mapping

Unlike Whisper (tiny/small/medium/large variants), VibeVoice has **multiple quantization levels**. Default to 6-bit for a good balance of quality and memory:

```python
model_map = {
    # Explicit quantization variants
    "vibevoice-4bit": "mlx-community/VibeVoice-ASR-4bit",
    "vibevoice-5bit": "mlx-community/VibeVoice-ASR-5bit",
    "vibevoice-6bit": "mlx-community/VibeVoice-ASR-6bit",
    "vibevoice-8bit": "mlx-community/VibeVoice-ASR-8bit",
    "vibevoice-bf16": "mlx-community/VibeVoice-ASR-bf16",
}
# Default everything else to 6-bit
DEFAULT_REPO = "mlx-community/VibeVoice-ASR-6bit"
```

#### Dependencies

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "mlx-audio>=0.2.0",
]
```

Note: mlx-audio brings `mlx-lm==0.30.5` and `transformers>=5.0.0` (we already have 5.0.0).

#### Registration

```python
# implementations/__init__.py
try:
    if importlib.util.find_spec("mlx_audio"):
        import platform
        if platform.system() == "Darwin":
            from mac_whisper_speedtest.implementations.vibevoice import VibeVoiceImplementation
            available_implementations.append(VibeVoiceImplementation)
except ImportError as e:
    logger.warning(f"Failed to import VibeVoiceImplementation: {e}")
```

### Memory Considerations

| Variant | Approx Size | Min RAM |
|---------|-------------|---------|
| 4-bit | ~5 GB | 8 GB |
| 5-bit | ~6 GB | 10 GB |
| **6-bit (default)** | ~7 GB | 12 GB |
| 8-bit | ~10 GB | 16 GB |
| bf16 (full) | ~18 GB | 32 GB |

**Using 6-bit as default** provides a good balance of quality and memory usage.

**Recommendation:** Add memory check in `load_model()`:
```python
import psutil
available_mem = psutil.virtual_memory().available / (1024**3)
min_mem = {"4bit": 8, "5bit": 10, "6bit": 12, "8bit": 16, "bf16": 32}
for variant, req in min_mem.items():
    if variant in self._hf_repo and available_mem < req:
        self.log.warning(f"VibeVoice-{variant} needs ~{req}GB, only {available_mem:.1f}GB available")
        break
```

### Verification Plan

1. **Unit test:** `tests/test_vibevoice_integration.py`
2. **Batch benchmark:**
   ```bash
   .venv/bin/mac-whisper-speedtest -b -n 1 -i "VibeVoiceImplementation"
   ```
3. **Verify speaker_id in segments**
4. **Memory monitoring during load/inference**

### Future Enhancements (Not in MVP)

| Feature | Description |
|---------|-------------|
| Context injection | `--context "domain terms"` CLI flag |
| Streaming mode | Real-time transcription output |
| Speaker count hint | Pre-specify expected speakers |

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Large model (9B) | Document requirements, add warning |
| Sample rate mismatch | Resample in implementation |
| mlx-audio API changes | Pin version |
| Slower than Whisper | Expected - document this |
