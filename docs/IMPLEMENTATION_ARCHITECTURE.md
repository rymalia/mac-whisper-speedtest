# Implementation Architecture Comparison

This document explains the architectural differences between the 9 Whisper implementations in mac-whisper-speedtest, helping developers understand why they perform differently and what trade-offs each approach makes.

## Overview

All implementations transcribe speech using Whisper-based models, but they use fundamentally different architectures, libraries, and integration patterns. Understanding these differences helps explain:

- **Performance characteristics** - Why some are faster than others
- **Warning messages** - Why InsanelyFastWhisper shows transformers warnings while others don't
- **Setup requirements** - Why some need Swift bridges while others are pure Python
- **Compatibility** - Why some are macOS-only while others are cross-platform
- **Optimization potential** - What trade-offs each architecture makes

## Quick Reference Table

| Implementation | Backend | Integration | Platform | Speed Tier | Library Dependency |
|----------------|---------|-------------|----------|------------|-------------------|
| FluidAudio-CoreML | CoreML + ANE | Swift Bridge | macOS only | ⚡⚡⚡ Fastest | FluidAudio (Swift) |
| WhisperKit | CoreML + ANE | Swift Bridge | macOS only | ⚡⚡ Very Fast | WhisperKit (Swift) |
| mlx-whisper | Apple MLX | Python Direct | macOS only | ⚡ Fast | mlx-whisper wrapper |
| whisper.cpp | C++ + CoreML | Python Bindings | Cross-platform | ⚡ Fast | pywhispercpp wrapper |
| parakeet-mlx | Apple MLX | Python Direct | macOS only | ⚡ Fast | parakeet-mlx wrapper |
| insanely-fast-whisper | PyTorch MPS | Python Direct | Cross-platform | Medium | **transformers (direct)** |
| lightning-whisper-mlx | Apple MLX | Python Direct | macOS only | Medium | lightning-whisper-mlx wrapper |
| faster-whisper | CTranslate2 (CPU) | Python Direct | Cross-platform | Medium | faster-whisper wrapper |
| whisper-mps | PyTorch MPS | Python Direct | macOS only | Slow | whisper-mps wrapper |

## Architecture Categories

### 1. Library Dependencies: Transformers vs Wrappers

This is the **most important architectural difference** and explains why warning messages differ between implementations.

#### Direct Transformers Usage (1 implementation)

**InsanelyFastWhisperImplementation** is the ONLY implementation that uses the raw HuggingFace `transformers` library directly:

```python
# insanely.py lines 142-148
from transformers.pipelines import pipeline

self._model = pipeline(
    "automatic-speech-recognition",
    model=self.model_name,
    torch_dtype=torch.float16,
    device=self.device_id,
    model_kwargs=model_kwargs,
)
```

**Characteristics:**
- ✅ Most flexible - direct access to all HuggingFace model parameters
- ✅ Latest features - always has newest transformers capabilities
- ⚠️ Verbose warnings - shows all internal transformers deprecation notices
- ⚠️ More configuration - requires manual attention mask, decoder ID management
- ⚠️ Breaking changes - subject to transformers API changes

**Why it shows warnings:**
- `FutureWarning: inputs → input_features` - Internal transformers API evolution
- `forced_decoder_ids conflict` - Direct exposure to model internals
- `attention_mask not set` - Low-level parameter management required

#### Wrapper Library Usage (8 implementations)

All other implementations use **wrapper libraries** that abstract away transformers complexity:

**MLX-based wrappers** (mlx-whisper, lightning-whisper-mlx, parakeet-mlx):
```python
# mlx.py line 144
from mlx_whisper import transcribe

result = transcribe(
    audio=audio,
    path_or_hf_repo=self._model_path,
    temperature=0.0,
    language=self.language,
    task="transcribe"
)
```

**CTranslate2 wrapper** (faster-whisper):
```python
# faster.py line 212
from faster_whisper import WhisperModel

segments, info = self._model.transcribe(
    audio,
    beam_size=self.beam_size,
    language=self.language,
    vad_filter=True,
)
```

**PyTorch MPS wrapper** (whisper-mps):
```python
# whisper_mps.py line 89
from whisper_mps.whisper.transcribe import transcribe

result = transcribe(
    audio=audio,
    model=self.model_name,
    temperature=0.0,
    language=self.language,
)
```

**Characteristics:**
- ✅ Clean API - high-level interface hides complexity
- ✅ No warnings - wrappers handle attention masks, decoder IDs internally
- ✅ Stable - wrappers shield from transformers API changes
- ✅ Optimized defaults - pre-configured for common use cases
- ⚠️ Less flexible - can't access low-level transformers features
- ⚠️ Update lag - may not have latest model features immediately

### 2. Backend Technologies

#### Native Swift + CoreML (2 implementations)

**FluidAudio-CoreML** and **WhisperKit** use native macOS frameworks:

**Architecture:**
```
Python → Subprocess → Swift Binary → CoreML → Apple Neural Engine
   ↓                                                      ↓
Save WAV                                            Hardware acceleration
   ↓                                                      ↓
Read JSON ← Internal timing ← Bridge overhead ← Transcription result
```

**Key files:**
- `implementations/fluidaudio_coreml.py` - Python wrapper
- `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift` - Swift executable
- Similar structure for WhisperKit

**Characteristics:**
- ⚡⚡⚡ Fastest performance - direct Apple Neural Engine access
- ✅ Zero Python ML overhead - no PyTorch, no TensorFlow
- ✅ Battery efficient - optimized for Apple hardware
- ✅ Internal timing - reports transcription time excluding bridge overhead
- ⚠️ macOS only - requires Swift Package Manager, Xcode
- ⚠️ Setup complexity - requires building Swift bridges
- ⚠️ Debugging difficulty - errors span Python/Swift boundary

**Bridge Pattern:**
```python
# fluidaudio_coreml.py lines 85-110 (simplified)
result = subprocess.run(
    [bridge_path, "--audio", temp_audio_path, "--model", model_name],
    capture_output=True,
    text=True
)
response = json.loads(result.stdout)
transcription_time = response["transcriptionTime"]  # Internal timing
```

#### Apple MLX Framework (3 implementations)

**mlx-whisper**, **lightning-whisper-mlx**, **parakeet-mlx** use Apple's MLX framework:

**Architecture:**
```
Python → MLX Python API → MLX C++ Core → Apple Metal → GPU
   ↓                                                      ↓
NumPy array                                        Unified memory
   ↓                                                      ↓
Direct processing ← Quantized models ← MLX optimizations
```

**Characteristics:**
- ⚡ Fast - optimized for Apple Silicon unified memory
- ✅ Quantization support - 4-bit, 8-bit models for memory efficiency
- ✅ Pure Python - no subprocess overhead
- ✅ Unified memory - efficient data transfer between CPU/GPU
- ⚠️ macOS only - MLX is Apple Silicon exclusive
- ⚠️ Model availability - depends on mlx-community conversions

**MLX Advantages:**
- Designed specifically for Apple Silicon architecture
- Optimized memory access patterns for unified memory
- Lightweight compared to PyTorch/TensorFlow

#### PyTorch MPS (2 implementations)

**insanely-fast-whisper**, **whisper-mps** use PyTorch's Metal Performance Shaders backend:

**Architecture:**
```
Python → PyTorch → MPS Backend → Apple Metal → GPU
   ↓                                              ↓
transformers                                Float16/quantization
   ↓                                              ↓
Direct API ← Attention mechanisms ← GPU kernels
```

**Characteristics:**
- ⚡ Medium speed - GPU acceleration via Metal
- ✅ Cross-platform PyTorch - can fall back to CPU/CUDA
- ✅ HuggingFace ecosystem - access to all transformers models
- ⚠️ Memory overhead - PyTorch is heavier than MLX
- ⚠️ MPS limitations - some operations fall back to CPU

**MPS vs MLX:**
- MPS: General PyTorch backend, broader compatibility
- MLX: Apple-specific, more optimized for unified memory

#### CTranslate2 CPU (1 implementation)

**faster-whisper** uses CTranslate2 for optimized CPU inference:

**Architecture:**
```
Python → CTranslate2 → Quantized models → Apple Accelerate → CPU
   ↓                                                          ↓
faster_whisper wrapper                               Performance cores
   ↓                                                          ↓
Beam search ← VAD filtering ← Multi-threaded inference
```

**Characteristics:**
- ✅ CPU-optimized - leverages Apple Accelerate framework
- ✅ Low memory - int8 quantization by default
- ✅ Cross-platform - works on any CPU
- ✅ Mature - battle-tested CTranslate2 engine
- ⚠️ No GPU on macOS - cannot use Apple Neural Engine or Metal
- ⚠️ Slower than GPU - limited by CPU performance

**Why no GPU on Apple Silicon:**
- CTranslate2 supports CUDA (NVIDIA) but not MPS (Apple)
- Relies on CPU SIMD optimizations via Apple Accelerate
- Thread optimization helps but can't match GPU/ANE speed

#### C++ + CoreML (1 implementation)

**whisper.cpp** uses optimized C++ with optional CoreML:

**Architecture:**
```
Python → pywhispercpp → whisper.cpp (C++) → CoreML → ANE
   ↓                                                    ↓
Bindings                                        GGML models
   ↓                                                    ↓
Minimal overhead ← Quantization ← Direct memory access
```

**Characteristics:**
- ⚡ Fast - efficient C++ implementation
- ✅ CoreML acceleration - optional Apple Neural Engine usage
- ✅ Minimal overhead - lightweight C++ core
- ✅ Cross-platform - works everywhere, optimized for Apple
- ⚠️ Binary dependencies - requires compiled libraries
- ⚠️ Manual model download - GGML models not auto-downloaded

### 3. Integration Patterns

#### Pattern A: Direct Python Libraries (6 implementations)

**MLX-based**, **PyTorch MPS**, **faster-whisper** use standard Python imports:

```python
# Direct import and usage
from library import model_or_function
result = model_or_function(audio, **params)
```

**Advantages:**
- Simple debugging - Python stack traces
- Easy development - standard Python workflow
- Direct control - all parameters accessible

**Trade-offs:**
- Python overhead - GIL, interpreter costs
- Library dependencies - large dependency trees

#### Pattern B: Swift Subprocess Bridges (2 implementations)

**WhisperKit**, **FluidAudio** use subprocess communication:

```python
# Subprocess pattern
subprocess.run([swift_binary, "--audio", audio_path])
# Parse JSON output
result = json.loads(stdout)
```

**Advantages:**
- Native performance - no Python ML overhead
- Internal timing - accurate performance measurement
- Hardware optimization - direct CoreML/ANE access

**Trade-offs:**
- IPC overhead - subprocess creation, file I/O
- Complex debugging - errors span language boundary
- Platform-specific - macOS + Swift toolchain required

**Bridge Overhead Mitigation:**
Both implementations report `_transcription_time` attribute that excludes:
- Audio file writing
- Subprocess creation
- JSON parsing
- Only measures actual transcription

See `benchmark.py` lines 151-156 for how this is handled.

#### Pattern C: C++ Bindings (1 implementation)

**whisper.cpp** uses compiled Python bindings:

```python
# C bindings via pywhispercpp
from pywhispercpp import Model
model = Model(model_path)
result = model.transcribe(audio)
```

**Advantages:**
- Native speed - C++ performance
- Memory efficient - direct memory access
- Minimal Python overhead

**Trade-offs:**
- Build complexity - requires C++ compilation
- Limited Python integration - less Pythonic API

## Implementation Deep Dives

### FluidAudio-CoreML (FASTEST)

**Architecture:** Swift + CoreML + Parakeet model
**Backend:** Apple Neural Engine
**Integration:** Swift subprocess bridge

**Why it's fastest:**
1. **Native Swift** - zero Python ML overhead
2. **Apple Neural Engine** - hardware acceleration
3. **Parakeet model** - optimized architecture (Parakeet TDT 0.6B v3)
4. **CoreML compilation** - ahead-of-time model optimization
5. **Minimal runtime** - no JIT compilation or warm-up

**Key implementation details:**
- Uses FluidAudio's streaming ASR framework
- Models stored in `~/Library/Application Support/io.fluid.FluidAudio/`
- Requires one-time model fix via `fix_models.sh`
- Reports internal timing excluding subprocess overhead

**Architecture file:** `implementations/fluidaudio_coreml.py`
**Bridge file:** `tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift`

### WhisperKit (Second Fastest)

**Architecture:** Swift + CoreML + Whisper
**Backend:** Apple Neural Engine
**Integration:** Swift subprocess bridge

**Performance characteristics:**
1. **Native Swift** - same benefits as FluidAudio
2. **Apple Neural Engine** - hardware acceleration
3. **Standard Whisper models** - broader compatibility
4. **CoreML optimized** - compiled model graphs

**Differences from FluidAudio:**
- Uses official OpenAI Whisper architecture
- Different model format (WhisperKit-specific)
- Slightly slower but more compatible
- No model fix required

**Architecture file:** `implementations/whisperkit.py`
**Bridge file:** `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift`

### mlx-whisper

**Architecture:** Python + Apple MLX
**Backend:** Apple Metal GPU
**Integration:** Direct Python

**Performance characteristics:**
1. **MLX framework** - optimized for unified memory
2. **Quantization** - 4-bit/8-bit models reduce memory bandwidth
3. **Direct processing** - NumPy array input, no file I/O
4. **Pure Python** - no subprocess overhead

**Key implementation details:**
- Automatically downloads from HuggingFace mlx-community
- Supports quantized models (4-bit default for small)
- Fallback chain for missing quantized models
- Direct `mlx_whisper.transcribe()` call

**Architecture file:** `implementations/mlx.py`

### whisper.cpp + CoreML

**Architecture:** C++ + Python bindings + CoreML
**Backend:** C++ core with optional ANE
**Integration:** Python bindings (pywhispercpp)

**Performance characteristics:**
1. **C++ implementation** - highly optimized inference
2. **Optional CoreML** - can use Apple Neural Engine
3. **GGML quantization** - efficient model formats
4. **Minimal dependencies** - lightweight runtime

**Key implementation details:**
- Uses pywhispercpp Python wrapper
- GGML model format (different from PyTorch)
- CoreML acceleration optional but recommended
- Manual model management

**Architecture file:** `implementations/coreml.py`

### parakeet-mlx

**Architecture:** Python + Apple MLX + Parakeet model
**Backend:** Apple Metal GPU
**Integration:** Direct Python

**Performance characteristics:**
1. **MLX framework** - same as mlx-whisper
2. **Parakeet TDT model** - NVIDIA's optimized architecture
3. **Alternative approach** - different from FluidAudio's Parakeet CoreML

**Differences from FluidAudio:**
- Same Parakeet model family, different backend (MLX vs CoreML)
- Pure Python vs Swift bridge
- Slower but easier setup

**Architecture file:** `implementations/parakeet_mlx.py`

### insanely-fast-whisper

**Architecture:** Python + HuggingFace transformers + PyTorch MPS
**Backend:** PyTorch Metal Performance Shaders
**Integration:** Direct Python (transformers pipeline)

**Performance characteristics:**
1. **Direct transformers** - no abstraction layer
2. **MPS acceleration** - PyTorch GPU backend
3. **SDPA attention** - optimized for Apple Silicon
4. **Adaptive batching** - memory-aware batch sizes

**Key implementation details:**
- ONLY implementation using raw transformers API
- Shows all transformers warnings (attention mask, decoder IDs)
- Most flexible for advanced configuration
- Adaptive batch size based on available memory

**Why it shows warnings:**
```python
# Line 194: Direct transformers parameters
generate_kwargs={"task": "transcribe"}
# Conflicts with model's forced_decoder_ids

# Line 193: Return language requires attention mask
return_language=True
# But pipeline doesn't auto-generate attention mask
```

**Architecture file:** `implementations/insanely.py`

### lightning-whisper-mlx

**Architecture:** Python + Apple MLX + LightningWhisperMLX wrapper
**Backend:** Apple Metal GPU
**Integration:** Direct Python (high-level wrapper)

**Performance characteristics:**
1. **MLX framework** - unified memory optimization
2. **Wrapper library** - abstracts MLX complexity
3. **4-bit quantization** - optional memory savings
4. **Batch processing** - configurable batch sizes

**Key implementation details:**
- High-level API: `LightningWhisperMLX(model, batch_size, quant)`
- Maps "large" → "large-v3" automatically
- Simple interface hides MLX internals

**Architecture file:** `implementations/lightning.py`

### faster-whisper

**Architecture:** Python + CTranslate2
**Backend:** CPU-only (Apple Accelerate framework)
**Integration:** Direct Python (faster_whisper wrapper)

**Performance characteristics:**
1. **CTranslate2** - optimized CPU inference engine
2. **No GPU on macOS** - cannot use Metal/ANE
3. **Multi-threaded** - intelligent core allocation
4. **int8 quantization** - memory efficient

**Apple Silicon optimizations:**
- Dynamic thread count based on P-cores/E-cores detection
- Uses `system_profiler` to detect core configuration
- Falls back to 75% of total cores if detection fails

**Why slower on Apple Silicon:**
- CTranslate2 doesn't support MPS (Apple GPU)
- Limited to CPU inference via Apple Accelerate
- Can't leverage Neural Engine or Metal

**Architecture file:** `implementations/faster.py`
**See:** Lines 62-125 for CPU thread optimization logic

### whisper-mps

**Architecture:** Python + whisper-mps + PyTorch MPS
**Backend:** PyTorch Metal Performance Shaders
**Integration:** Direct Python (whisper-mps wrapper)

**Performance characteristics:**
1. **PyTorch MPS** - GPU acceleration via Metal
2. **Wrapper abstraction** - simplified API
3. **Automatic MPS** - auto-detects and uses GPU

**Key implementation details:**
- Wrapper around original OpenAI Whisper with MPS support
- Downloads models from `openaipublic.azureedge.net` (not HuggingFace)
- Automatic MPS device selection when available
- Slowest implementation despite GPU acceleration

**Why relatively slow:**
- Less optimized than MLX for Apple Silicon
- PyTorch overhead on MPS backend
- Not specialized for unified memory architecture

**Architecture file:** `implementations/whisper_mps.py`

## Key Architectural Trade-offs

### Speed vs Compatibility

| Priority | Recommended Implementations | Trade-off |
|----------|---------------------------|-----------|
| **Maximum Speed** | FluidAudio, WhisperKit | macOS-only, complex setup |
| **Fast + Portable** | mlx-whisper, whisper.cpp | Good speed, broader compatibility |
| **Maximum Compatibility** | insanely-fast-whisper, faster-whisper | Works everywhere, slower |

### Native vs Portable

**Native (macOS-only):**
- FluidAudio, WhisperKit, MLX-based (mlx-whisper, lightning, parakeet)
- Fastest on Apple Silicon
- Can't run on Linux/Windows

**Portable (cross-platform):**
- insanely-fast-whisper, faster-whisper, whisper.cpp
- Run anywhere
- Slower on Apple Silicon (except whisper.cpp)

### Complexity vs Performance

**Simple Setup:**
- mlx-whisper: `pip install mlx-whisper` → done
- insanely-fast-whisper: `pip install insanely-fast-whisper` → done

**Complex Setup, Better Performance:**
- FluidAudio: Build Swift bridge + run model fix script
- WhisperKit: Build Swift bridge + Xcode dependencies

### Direct Control vs Abstraction

**Low-level (insanely-fast-whisper):**
- Full transformers API access
- Maximum flexibility
- More warnings, more configuration

**High-level (wrappers):**
- Clean API, sensible defaults
- Less flexibility
- No warnings, less control

## Warning Messages Explained

### Why Only InsanelyFastWhisper Shows Warnings

**Root Cause:** It's the only implementation using the raw `transformers` library directly.

**Specific Warnings:**

1. **`FutureWarning: inputs → input_features`**
   - **Source:** Internal transformers API evolution
   - **Why others don't show it:** Wrappers use updated API or suppress warnings
   - **Impact:** None currently, will break in future transformers version

2. **`forced_decoder_ids conflict`**
   - **Source:** Line 194: `generate_kwargs={"task": "transcribe"}` conflicts with model defaults
   - **Why others don't show it:** Wrappers handle decoder IDs internally
   - **Impact:** None - explicit task takes priority

3. **`attention_mask not set`**
   - **Source:** Line 193: `return_language=True` without explicit attention mask
   - **Why others don't show it:** Wrappers auto-generate attention masks
   - **Impact:** Minor edge case issues at audio boundaries

**Wrapper libraries handle these automatically:**
```python
# MLX wrapper (mlx_whisper) - no warnings
result = transcribe(audio=audio, language=None, task="transcribe")
# Internally handles all decoder IDs, attention masks

# Faster-whisper wrapper - no warnings
segments, info = model.transcribe(audio, language=None)
# CTranslate2 backend manages all internals
```

## Choosing the Right Implementation

### Decision Tree

```
Do you need maximum speed?
├─ YES, macOS-only is OK
│  └─ Use: FluidAudio-CoreML (0.08s)
│     Alternative: WhisperKit (0.43s)
│
└─ Need cross-platform?
   ├─ Want GPU acceleration?
   │  ├─ macOS: mlx-whisper (0.71s)
   │  └─ Other: insanely-fast-whisper (1.24s)
   │
   └─ CPU-only is fine?
      └─ faster-whisper (2.11s)
         Alternative: whisper.cpp (0.86s)
```

### Use Case Recommendations

**Real-time applications:** FluidAudio-CoreML or WhisperKit
- Lowest latency, hardware accelerated

**Batch processing:** mlx-whisper or whisper.cpp
- Good throughput, simpler setup than Swift bridges

**Development/experimentation:** insanely-fast-whisper
- Direct transformers access, maximum flexibility

**Production (cross-platform):** faster-whisper
- Stable, mature, works everywhere

**Memory-constrained:** mlx-whisper with 4-bit quantization
- Efficient memory usage, good speed

## Future Architecture Considerations

### Emerging Patterns

1. **Distilled models** (Parakeet) - Faster than standard Whisper
2. **Quantization** - 4-bit/8-bit becoming standard
3. **Streaming inference** - Real-time processing (FluidAudio)
4. **Neural Engine optimization** - Direct ANE access (CoreML implementations)

### Adding New Implementations

When adding new implementations, consider:

1. **Backend category** - MLX, PyTorch, Swift, C++?
2. **Integration pattern** - Direct Python, subprocess, bindings?
3. **Platform requirements** - macOS-only or cross-platform?
4. **Warning handling** - Raw transformers or wrapper?

See `CLAUDE.md` "Adding New Implementations" section for code patterns.

## Version Status Audit

This section tracks the current state of all implementation dependencies to help maintain the project with the latest stable versions.

### Swift Bridge Dependencies

| Package | Your Version | Latest Available | Status | GitHub |
|---------|-------------|------------------|---------|---------|
| **WhisperKit** | 0.13.1 | **0.15.0** (Dec 13, 2025) | ⚠️ **2 versions behind** | [Releases](https://github.com/argmaxinc/WhisperKit/releases) |
| **FluidAudio** | 0.7.12 | **0.8.0** (Dec 18, 2025) | ⚠️ **1 version behind** | [Releases](https://github.com/FluidInference/FluidAudio/releases) |

**Location:** `tools/whisperkit-bridge/Package.swift` and `tools/fluidaudio-bridge/Package.swift`

**How to update:**
```bash
# Edit Package.swift files to update version numbers, then:
cd tools/whisperkit-bridge && swift build -c release && cd ../..
cd tools/fluidaudio-bridge && swift build -c release && cd ../..
```

### Python Package Dependencies

Based on `uv pip list` output (last checked: 2025-12-26):

| Package | Installed | Latest Available | Status | PyPI |
|---------|-----------|------------------|---------|------|
| **mlx** | 0.27.1 | **0.30.1** (Dec 18, 2025) | ⚠️ **3 minor versions behind** | [PyPI](https://pypi.org/project/mlx/) |
| **mlx-whisper** | 0.4.2 | **0.4.3** (Aug 29, 2025) | ⚠️ **1 patch behind** | [PyPI](https://pypi.org/project/mlx-whisper/) |
| **faster-whisper** | 1.1.1 | **1.2.1** (Oct 31, 2025) | ⚠️ **1 minor + 1 patch behind** | [PyPI](https://pypi.org/project/faster-whisper/) |
| **parakeet-mlx** | 0.3.5 | **0.4.1** (Nov 20, 2025) | ⚠️ **1 minor + 1 patch behind** | [PyPI](https://pypi.org/project/parakeet-mlx/) |
| **whisper-mps** | 0.0.7 | **0.0.9** (Aug 15, 2025) | ⚠️ **2 patches behind** | [PyPI](https://pypi.org/project/whisper-mps/) |
| **insanely-fast-whisper** | 0.0.15 | 0.0.15 | ✅ **Up to date** | [PyPI](https://pypi.org/project/insanely-fast-whisper/) |
| **lightning-whisper-mlx** | 0.0.10 | 0.0.10 | ✅ **Up to date** | [PyPI](https://pypi.org/project/lightning-whisper-mlx/) |
| **pywhispercpp** | 1.3.1.dev38 | git main | 🔄 **Tracking git main** | [GitHub fork](https://github.com/absadiki/pywhispercpp) |

**Related packages:**
- `mlx-metal` 0.27.1 (bundled with mlx)
- `coremltools` ≥8.0.0
- `huggingface-hub` ≥0.20.0

**Location:** `pyproject.toml` (dependencies) and `uv.lock` (locked versions)

### uv sync Results

Last run: 2025-12-26

**Output:**
```
Resolved 160 packages in 18ms
Audited 139 packages in 33ms
```

**Changes reported:** ✅ **None** - No packages were updated

**Why no automatic updates?**

The project uses **minimum version constraints** in `pyproject.toml`:
```toml
dependencies = [
    "mlx>=0.5.0",           # Allows 0.5.0 or newer
    "mlx-whisper>=0.4.2",   # Allows 0.4.2 or newer
    "faster-whisper>=1.1.1" # Allows 1.1.1 or newer
    # etc...
]
```

However, `uv sync` **respects the lock file** (`uv.lock`) and doesn't auto-upgrade to newer versions that satisfy the constraints. This ensures reproducible builds.

**To update dependencies:**

```bash
# Option 1: Update all packages to latest compatible versions
uv lock --upgrade

# Option 2: Update specific package
uv lock --upgrade-package mlx

# Option 3: Manually update version constraints in pyproject.toml
# Then run: uv sync
```

### Summary and Impact Assessment

**Most outdated packages:**

1. 🔴 **mlx**: 0.27.1 → 0.30.1 (3 minor versions, ~3 months behind)
   - **Impact:** HIGH - Core framework for mlx-whisper, lightning-whisper-mlx, and parakeet-mlx
   - **Risk:** Performance improvements, bug fixes, potential API changes
   - **Affected implementations:** 3 (mlx-whisper, lightning-whisper-mlx, parakeet-mlx)

2. 🟡 **faster-whisper**: 1.1.1 → 1.2.1 (1 minor version behind)
   - **Impact:** MEDIUM - May include performance optimizations
   - **Risk:** Bug fixes, potential accuracy improvements
   - **Affected implementations:** 1 (faster-whisper)

3. 🟡 **parakeet-mlx**: 0.3.5 → 0.4.1 (1 minor version behind)
   - **Impact:** MEDIUM - Model updates or API changes
   - **Risk:** Potential performance/quality improvements
   - **Affected implementations:** 1 (parakeet-mlx)

4. 🟡 **WhisperKit** (Swift): 0.13.1 → 0.15.0 (2 versions behind)
   - **Impact:** MEDIUM - Native Swift framework updates
   - **Risk:** CoreML optimizations, bug fixes
   - **Affected implementations:** 1 (WhisperKit)

5. 🟡 **FluidAudio** (Swift): 0.7.12 → 0.8.0 (1 version behind, released Dec 18)
   - **Impact:** MEDIUM - Very recent release (8 days old)
   - **Risk:** Latest optimizations, may have breaking changes
   - **Affected implementations:** 1 (FluidAudio-CoreML)

**Minor updates:**
- whisper-mps: 2 patches behind
- mlx-whisper: 1 patch behind

### Recommendations

**Priority 1 - Update MLX framework:**
```bash
# MLX is 3 versions behind and affects 3 implementations
uv lock --upgrade-package mlx
uv lock --upgrade-package mlx-metal
uv sync
```

**Priority 2 - Update Swift bridges:**
```bash
# Edit tools/whisperkit-bridge/Package.swift
# Change: from: "0.13.1" → from: "0.15.0"

# Edit tools/fluidaudio-bridge/Package.swift
# Change: from: "0.7.12" → from: "0.8.0"

# Rebuild
cd tools/whisperkit-bridge && swift build -c release && cd ../..
cd tools/fluidaudio-bridge && swift build -c release && cd ../..
```

**Priority 3 - Update remaining packages:**
```bash
# Update all packages at once
uv lock --upgrade

# Or selectively:
uv lock --upgrade-package faster-whisper
uv lock --upgrade-package parakeet-mlx
uv lock --upgrade-package whisper-mps
uv lock --upgrade-package mlx-whisper
```

**Testing after updates:**
1. Run quick test: `.venv/bin/mac-whisper-speedtest --model small --num-runs 1`
2. Verify all implementations still work
3. Check for performance regressions
4. Update benchmark results if significant changes

**Risks:**
- ⚠️ **FluidAudio 0.8.0** is very recent - may have breaking changes
- ⚠️ **MLX 0.30.1** is 3 versions ahead - test thoroughly after update
- ⚠️ **WhisperKit 0.15.0** may require iOS/macOS version updates

**Benefits:**
- ✅ Performance improvements from newer MLX framework
- ✅ Bug fixes across all implementations
- ✅ Latest model support and optimizations
- ✅ Security patches in dependencies

### Checking for Updates

To check for the latest versions manually:

```bash
# Python packages
uv pip list --outdated

# Or check individual packages
pip index versions mlx
pip index versions faster-whisper

# Swift packages
# Check GitHub releases pages:
# - https://github.com/argmaxinc/WhisperKit/releases
# - https://github.com/FluidInference/FluidAudio/releases
```

## Related Documentation

- **README.md** - High-level feature comparison
- **APPLE_SILICON_OPTIMIZATIONS.md** - Performance tuning details
- **MODEL_CACHING.md** - Model management and download strategies
- **CLAUDE.md** - Development patterns and implementation guide

---

**Last Updated:** 2025-12-26
**Implementations Covered:** 9 (FluidAudio, WhisperKit, mlx-whisper, whisper.cpp, parakeet-mlx, insanely-fast-whisper, lightning-whisper-mlx, faster-whisper, whisper-mps)
**Version Audit:** Included (last checked: 2025-12-26)
