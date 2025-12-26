# Benchmark Results

**Note:** This benchmark includes 9 working implementations. FluidAudio CoreML now works with a manual model fix and is the **fastest implementation**.

## Test Configuration

- **Model**: small
- **Audio Sample**: JFK speech ("And so my fellow Americans, ask not what your country can do for you...")
- **Audio Duration**: ~11 seconds (176,000 samples at 16kHz)
- **Platform**: macOS with Apple Silicon
- **Test Date**: 2025-12-25

## Performance Rankings

| Rank | Implementation | Avg Time (s) | Speed | Status |
|------|---------------|--------------|-------|--------|
| 1 | **FluidAudio-CoreML** | **0.08** | **⚡⚡⚡⚡ Ultra Fast** | ✅ Working (with fix) |
| 2 | WhisperKit | 0.43 | ⚡⚡⚡ Fastest | ✅ Working |
| 3 | MLX-Whisper | 0.71 | ⚡⚡ Very Fast | ✅ Working |
| 4 | Whisper.cpp (CoreML) | 0.86 | ⚡⚡ Very Fast | ✅ Working |
| 5 | Parakeet-MLX | 0.91 | ⚡ Fast | ✅ Working |
| 6 | Insanely-Fast-Whisper | 1.24 | → Moderate | ✅ Working |
| 7 | Lightning-Whisper-MLX | 1.40 | → Moderate | ✅ Working |
| 8 | Faster-Whisper | 2.11 | ↓ Slow | ✅ Working |
| 9 | Whisper-MPS | 6.90 | ↓↓ Very Slow | ✅ Working |

## Detailed Results

### 1. FluidAudio-CoreML (0.08s) ⭐⭐⭐ WINNER - FASTEST IMPLEMENTATION!
**Parameters:**
- Model: parakeet-tdt-0.6b-v3-coreml
- Backend: FluidAudio Swift Bridge
- Platform: Apple Silicon (CoreML + Neural Engine)

**Notes:**
- **81% faster than WhisperKit** (0.08s vs 0.43s)
- **138x real-time factor** (processes 11s audio in 0.08s)
- Uses NVIDIA Parakeet model optimized for CoreML
- Requires manual model fix: `./tools/fluidaudio-bridge/fix_models.sh`
- See `docs/MODEL_CACHING.md` for setup instructions
- Swift bridge with minimal overhead
- Internal timing excludes subprocess overhead

**Setup Required:**
```bash
cd tools/fluidaudio-bridge
swift build -c release
./fix_models.sh  # Copies models from HuggingFace cache to Application Support
```

### 2. WhisperKit (0.43s) ⭐⭐ 2nd Fastest
**Parameters:**
- Model: small
- Backend: WhisperKit Swift Bridge
- Platform: Apple Silicon (CoreML + Neural Engine)

**Notes:**
- Uses native Apple Neural Engine acceleration
- Swift bridge with minimal overhead
- Internal timing excludes subprocess overhead
- Excellent performance without any setup required

### 3. MLX-Whisper (0.71s)
**Parameters:**
- Model: mlx-community/whisper-small-mlx-4bit
- Quantization: 4-bit

**Notes:**
- Apple MLX framework optimization
- 4-bit quantization for memory efficiency
- Excellent performance/quality balance

### 4. Whisper.cpp CoreML (0.86s)
**Parameters:**
- Model: small
- CoreML: Enabled
- Threads: 4

**Notes:**
- Uses CoreML for acceleration
- C++ implementation with Python bindings
- Good CPU/CoreML hybrid performance

### 5. Parakeet-MLX (0.91s)
**Parameters:**
- Model: mlx-community/parakeet-tdt-0.6b-v2
- Implementation: parakeet-mlx
- Platform: Apple Silicon (MLX)

**Notes:**
- NVIDIA Parakeet model adapted for MLX
- Slightly different transcription style (adds commas)
- Very competitive performance
- Alternative to FluidAudio for Parakeet model testing

### 6. Insanely-Fast-Whisper (1.24s)
**Parameters:**
- Model: openai/whisper-small
- Device: MPS
- Batch Size: 16 (adaptive for Apple Silicon)
- Compute Type: float16
- Quantization: 4bit (not available on macOS)

**Notes:**
- Uses Apple MPS (Metal Performance Shaders)
- Adaptive batch sizing based on available memory
- SDPA attention optimized for MPS
- bitsandbytes quantization not supported on macOS

### 7. Lightning-Whisper-MLX (1.40s)
**Parameters:**
- Model: small
- Batch Size: 12
- Quantization: none

**Notes:**
- MLX-based implementation
- No quantization applied in this test
- Reasonable performance

### 8. Faster-Whisper (2.11s)
**Parameters:**
- Model: small
- Device: CPU
- Compute Type: int8
- Beam Size: 1
- CPU Threads: 6 (auto-detected)

**Notes:**
- CPU-only implementation
- int8 quantization
- Dynamic thread detection for performance/efficiency cores
- Good accuracy but slower than GPU-accelerated options

### 9. Whisper-MPS (6.90s)
**Parameters:**
- Model: small
- Backend: whisper-mps
- Device: MPS
- Language: auto-detect

**Notes:**
- Significantly slower than other implementations
- May benefit from optimization or configuration tuning
- MPS acceleration not as efficient as other frameworks

## Transcription Quality

All working implementations produced correct transcriptions of the JFK speech:

> "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

Minor variations:
- **Parakeet-MLX** adds more punctuation: "And so, my fellow Americans, ask not..."
- All others produce nearly identical output

## Key Findings

### Performance Insights

1. **FluidAudio is the fastest**: At 0.08s, FluidAudio-CoreML is **81% faster than WhisperKit** and achieves **138x real-time factor**
2. **Native Apple frameworks dominate**: CoreML + Neural Engine implementations (FluidAudio, WhisperKit) are significantly faster than alternatives
3. **MLX is highly competitive**: MLX-based implementations (ranks 3, 5, 7) show excellent performance
4. **Quantization helps**: 4-bit quantization in MLX-Whisper provides good speed without sacrificing quality
5. **CPU-only is viable**: Faster-Whisper at 2.11s is still reasonable for many use cases
6. **MPS performance varies**: Insanely-Fast-Whisper (1.24s) vs Whisper-MPS (6.90s) shows significant variance
7. **Parakeet models excel**: Both FluidAudio (Parakeet v3) and ParakeetMLX (v2) show excellent performance

### Apple Silicon Optimizations Applied

- **FluidAudio-CoreML**: Native CoreML + Apple Neural Engine, Parakeet model optimized for real-time streaming
- **WhisperKit**: Native CoreML + Apple Neural Engine utilization
- **Faster-Whisper**: Dynamic CPU thread detection (6 threads for M4 Pro)
- **Insanely-Fast-Whisper**: Adaptive batch sizing (16 for high memory), SDPA attention
- **Lightning-Whisper-MLX**: Batch size optimized for unified memory architecture
- **MLX implementations**: Leveraging Apple's MLX framework optimizations

## Recommendations

### For Maximum Speed
**Use FluidAudio-CoreML** - Fastest implementation at 0.08s (138x RTF), 81% faster than WhisperKit
- Requires one-time setup: `./tools/fluidaudio-bridge/fix_models.sh`
- See `docs/MODEL_CACHING.md` for details

### For Speed + Ease of Use
**Use WhisperKit** - 2nd fastest at 0.43s, no setup required, excellent quality, native Apple integration

### For Balance (Speed + Flexibility)
**Use MLX-Whisper** - 3rd fastest at 0.71s, 4-bit quantization, good ecosystem support

### For CPU-Only Deployment
**Use Faster-Whisper** - Best CPU performance, good accuracy, reliable

### For Real-Time Streaming
**Use FluidAudio-CoreML** - Achieves 138x real-time factor, ideal for streaming applications

## Known Issues

### FluidAudio CoreML - Manual Setup Required ✅ RESOLVED
**Problem:** Framework's automatic model copy from HuggingFace cache to Application Support fails
**Root Cause:** FluidAudio downloads complete models to `~/.cache/huggingface/hub/` but fails to copy them to `~/Library/Application Support/FluidAudio/Models/`
**Solution:** Manual copy using provided script: `./tools/fluidaudio-bridge/fix_models.sh`
**Status:** ✅ Working perfectly after one-time setup - **FASTEST IMPLEMENTATION** at 0.08s
**Documentation:** See `docs/MODEL_CACHING.md` for complete setup instructions and `docs/FLUIDAUDIO_FINAL_STATUS.md` for investigation details

### Insanely-Fast-Whisper Quantization
**Problem:** bitsandbytes library not supported on macOS
**Impact:** 4-bit quantization configuration ignored
**Workaround:** Falls back to float16, still performs adequately
**Documentation:** Warning message added to inform users

## Testing Methodology

- Each implementation tested with identical audio input
- Model loading time excluded from measurements
- For Swift bridges (WhisperKit, FluidAudio): internal timing used to exclude subprocess overhead
- Single run per implementation for this test (production should use 3+ runs for statistical accuracy)
- FluidAudio requires one-time setup using `fix_models.sh` script

## Next Steps

1. ✅ **FluidAudio resolved** - Working with manual model fix, now the fastest implementation
2. **Report FluidAudio bug** - Submit issue to FluidInference about model copy failure
3. **Investigate Whisper-MPS performance** - 6.90s seems unusually slow
4. **Run extended benchmark** with 3+ runs for statistical confidence
5. **Test with larger models** (medium, large) to see if rankings change
6. **Add quality metrics** beyond transcription accuracy (WER, CER)
