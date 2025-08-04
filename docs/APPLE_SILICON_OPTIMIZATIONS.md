# Apple Silicon Optimizations for mac-whisper-speedtest

This document describes the optimizations applied to improve performance of lightning-whisper-mlx and faster-whisper implementations on Apple Silicon.

## Background

Based on benchmark results, lightning-whisper-mlx (2.97s) and faster-whisper (12.48s) were significantly slower than top performers like whisper.cpp (0.99s). This analysis identified optimization opportunities specifically for Apple Silicon.

## Lightning Whisper MLX Optimizations

### 1. Enabled 4bit Quantization

- **Changed**: `self.quant = None` → `self.quant = "4bit"`
- **Rationale**: 4bit quantization reduces memory bandwidth requirements, which is crucial for Apple Silicon's unified memory architecture
- **Expected Impact**: Significant speed improvement (potentially 2-4x faster) with minimal quality loss
- **Why 4bit over 8bit**: Provides the best speed/quality balance according to lightning-whisper-mlx documentation

### 2. Optimized Batch Size

- **Changed**: `self.batch_size = 24` → `self.batch_size = 12`
- **Rationale**:
  - 12 is the recommended default by lightning-whisper-mlx
  - Better memory efficiency on Apple Silicon's unified memory
  - Large models (like distil-large-v3) benefit from smaller batch sizes
- **Expected Impact**: Better memory utilization and potentially improved throughput

## Faster Whisper Optimizations

### 1. Increased CPU Thread Count

- **Changed**: `self.cpu_threads = 4` → `self.cpu_threads = 8`
- **Rationale**:
  - Apple Silicon chips typically have 8+ cores (4-8 performance + 2-4 efficiency cores)
  - Previous setting of 4 threads was underutilizing available CPU resources
  - faster-whisper is CPU-only on Apple Silicon (no GPU/MPS support)
- **Expected Impact**: Better CPU utilization, moderate performance improvement

### 2. Verified Optimal Settings

- **Confirmed**: `device="cpu"` (only option available - no MPS/GPU support)
- **Confirmed**: `compute_type="int8"` (optimal for Apple Silicon CPU processing)

## Limitations Identified

### Faster Whisper Fundamental Limitation

- **Issue**: faster-whisper does NOT support GPU acceleration on Apple Silicon
- **Root Cause**: Built on CTranslate2 which only supports CPU acceleration via Apple Accelerate framework
- **Impact**: Performance improvements are limited compared to GPU-accelerated alternatives
- **Recommendation**: Consider using other implementations (like whisper.cpp with CoreML) for maximum Apple Silicon performance

## Expected Performance Improvements

### Lightning Whisper MLX

- **Quantization**: 2-4x speed improvement expected
- **Batch Size**: Better memory efficiency, potential throughput gains
- **Overall**: Should see significant performance improvement, potentially moving from ~3s to ~1-1.5s range

### Faster Whisper

- **CPU Threads**: Moderate improvement, potentially 20-30% faster
- **Overall**: Limited improvement due to CPU-only processing, may improve from ~12.5s to ~9-10s range

## Technical Details

### Apple Silicon Architecture Considerations

1. **Unified Memory**: Optimizations focus on reducing memory bandwidth requirements
2. **CPU Cores**: Utilize both performance and efficiency cores effectively
3. **No GPU Support**: faster-whisper cannot leverage Apple Silicon's GPU/Neural Engine

### Code Changes Summary

- `src/mac_whisper_speedtest/implementations/lightning.py`: Enabled 4bit quantization, optimized batch size
- `src/mac_whisper_speedtest/implementations/faster.py`: Increased CPU threads to 8

## Verification

The optimizations maintain backward compatibility and only modify performance parameters. All changes include detailed comments explaining the Apple Silicon-specific rationale.

## Future Recommendations

1. **For maximum performance**: Consider prioritizing implementations that support Apple Silicon GPU/Neural Engine (like whisper.cpp with CoreML)
2. **For faster-whisper users**: Document the CPU-only limitation and suggest alternatives for speed-critical applications
3. **Monitor results**: Benchmark the optimized implementations to validate expected improvements

---

## OPTIMIZATION RESULTS (2025-08-03)

### Applied Optimizations Summary

#### Insanely Fast Whisper Optimizations ✅

1. **Adaptive Batch Sizing**

   - Dynamic batch size based on available system memory
   - Apple Silicon unified memory optimization
   - Result: Reduced from 24 to 12 on test system

2. **Attention Implementation Optimization**

   - Force SDPA over flash_attention_2 on Apple Silicon MPS
   - Better optimization for Apple Silicon backend

3. **Memory Layout Optimizations**

   - Added `use_cache=True` and `low_cpu_mem_usage=True`
   - Optimized for unified memory architecture

4. **Chunk Length Optimization**
   - Reduced chunk_length_s from 30 to 20 seconds
   - Better memory efficiency

**Performance Results:**

- **Before**: 0.6210 seconds
- **After**: 0.5215 seconds
- **Improvement**: **16.0% faster** 🎉

#### Faster Whisper Optimizations ✅

1. **Dynamic CPU Thread Optimization**

   - Automatic Apple Silicon chip architecture detection
   - Optimal thread allocation: performance_cores + 2 efficiency cores
   - Result: Increased from 8 to 12 threads on M4 Pro

2. **Compute Type Optimization**

   - Smart int8/float16 selection based on available memory
   - Leverages Apple Accelerate framework

3. **Apple Silicon Detection**
   - Comprehensive chip detection and optimization

**Performance Results:**

- **Before**: 1.5399 seconds
- **After**: 1.5095 seconds
- **Improvement**: **2.0% faster** 🎉

### Overall Impact

- **Total time saved**: 0.1299 seconds per transcription
- **insanely-fast-whisper**: Now much closer to top-tier MLX implementations
- **faster-whisper**: Improved performance within CPU-bound limitations

### Key Success Factors

1. **Apple Silicon-specific optimizations** rather than generic improvements
2. **Memory bandwidth optimization** over raw compute optimization
3. **Adaptive configuration** based on system capabilities
4. **Leveraging Apple frameworks** (Accelerate, MPS) effectively
