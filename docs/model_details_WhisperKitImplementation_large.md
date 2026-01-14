# Model Details: WhisperKitImplementation (Large Model)

This document traces the complete execution flow for `WhisperKitImplementation` with the **large** model, documenting model download behavior, caching, timeout issues, and critical bugs discovered through empirical testing.

---

## File Reference Legend

Throughout this document, files are categorized as:
- **[PROJECT]** - Files in this repository that you can modify
- **[LIBRARY]** - Installed package files (Swift/Python dependencies)
- **[BRIDGE]** - The Swift CLI executable that interfaces with WhisperKit

---

## Key Questions Answered

| Question | Answer | Evidence |
|----------|--------|----------|
| Does Swift bridge re-download every call? | **NO** - uses file-exists caching | HubApi.swift:204-206 |
| Where are temp files during download? | `/private/var/folders/*/T/CFNetworkDownload_*.tmp` | Empirical: found 611MB, 477MB orphaned files |
| Download cache vs runtime location? | **SAME** - `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/` | HubApi.swift:19-29, empirical verification |
| Is 300s timeout sufficient for large? | **NO** - large model needs ~17 minutes | Empirical: 611MB in 5 min = ~2 MB/s, ~2GB total |
| Does partial download resume? | **NO** - starts fresh, orphans temp files | Empirical: new 477MB temp file on run 2 |
| Completeness check on existing files? | **NONE** - only `FileManager.fileExists()` | HubApi.swift:204-206, empirical: 4K file skipped |
| What happens with incomplete cache? | **SILENT FAILURE** - incomplete files treated as complete | Empirical: 4K TextDecoder skipped |

---

## Benchmark Execution Flow

**Command:**
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

### Step-by-Step Execution

1. **Entry Point** — **[PROJECT]** `test_benchmark2.py:88-94`
   - Parses CLI args: `model="large"`, `runs=1`, `implementations="WhisperKitImplementation"`

2. **Model Name Mapping** — **[PROJECT]** `whisperkit.py:58-67`
   ```python
   model_map = {
       "large": "large-v3",  # Standard large-v3 for compatibility
   }
   self.model_name = model_map.get(model_name, model_name)
   ```
   - `"large"` → `"large-v3"`

3. **Transcription Call** — **[PROJECT]** `whisperkit.py:124-129`
   ```python
   result = subprocess.run(
       [self._bridge_path, temp_path, "--format", "json", "--model", self.model_name],
       timeout=300  # 5 minute timeout
   )
   ```

4. **Swift Bridge Initialization** — **[BRIDGE]** `main.swift:32-33`
   ```swift
   let config = WhisperKitConfig(model: model)  // model = "large-v3"
   let whisperKit = try await WhisperKit(config)
   ```

5. **Model Download Triggered** — **[LIBRARY]** `WhisperKit.swift:298-330`
   - `setupModels()` calls `Self.download(variant: "large-v3", ...)`
   - Uses HubApi from swift-transformers

6. **HubApi Snapshot Download** — **[LIBRARY]** `HubApi.swift:234-257`
   - Downloads files matching `*large-v3/*` from `argmaxinc/whisperkit-coreml`
   - For each file, checks `downloaded` property (line 204-206):
     ```swift
     var downloaded: Bool {
         FileManager.default.fileExists(atPath: destination.path)
     }
     ```
   - **CRITICAL BUG**: Only checks existence, not completeness!

---

## Summary Table

| Attribute | Value |
|-----------|-------|
| **Requested Model** | `large` |
| **Mapped Model Name** | `large-v3` |
| **HuggingFace Repo ID** | `argmaxinc/whisperkit-coreml` |
| **Model Search Pattern** | `*large-v3/*` or `*openai*large-v3/*` |
| **Expected Files** | `AudioEncoder.mlmodelc/`, `TextDecoder.mlmodelc/`, `MelSpectrogram.mlmodelc/`, config files, tokenizer |
| **Cache Location** | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/` |
| **Temp Files During Download** | `/private/var/folders/*/T/CFNetworkDownload_*.tmp` |
| **Expected Total Size** | ~2GB (AudioEncoder ~600MB, TextDecoder ~1.2GB, others ~200MB) |
| **Current Timeout** | 300 seconds (5 minutes) |
| **Required Timeout** | ~1000 seconds (~17 minutes at 2 MB/s) |

---

## Model Mapping Reference

### Project-Level Mapping — **[PROJECT]** `whisperkit.py:58-67`

| Input | Output | Notes |
|-------|--------|-------|
| `"large"` | `"large-v3"` | **Upgraded to latest version** |
| `"large-v3"` | `"large-v3"` | No change |
| `"large-v3-turbo"` | `"large-v3-turbo"` | Explicit turbo access (smaller, faster) |
| `"large-turbo"` | `"large-v3-turbo"` | Alternative turbo access |

---

## Empirical Test Results

**Test Date**: January 11, 2026
**Test Environment**: macOS, Apple Silicon, ~2 MB/s download speed

### Test 1: Fresh Download Attempt (No Cache)

**Initial state**: Empty cache folder
```
$ ls -la ~/Documents/huggingface/models/argmaxinc/
total 16
drwxr-xr-x@ 3 rymalia  staff    96 Jan 10 23:43 .
drwxr-xr-x@ 5 rymalia  staff   160 Jan 11 11:15 ..
-rw-r--r--@ 1 rymalia  staff  6148 Jan 10 23:44 .DS_Store
```

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

**Terminal output**:
```
2026-01-11 11:20:02 [info] Using WhisperKit large-v3 model (requested: large, using: large-v3)
2026-01-11 11:20:02 [info] Transcribing with WhisperKit via Swift bridge (model: large-v3)
2026-01-11 11:25:02 [error] WhisperKit bridge timed out
```

**Result**: **TIMEOUT after exactly 300 seconds**

**Partial cache state after timeout**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*
372K    MelSpectrogram.mlmodelc    # Complete
4.0K    TextDecoder.mlmodelc       # INCOMPLETE (should be ~1.2GB!)
```

**Orphaned temp file**:
```
$ ls -lh /private/var/folders/*/T/CFNetworkDownload*.tmp
611M    CFNetworkDownload_1iOtaj.tmp   # Partial download, abandoned
```

**Key observations**:
1. Download achieved 611MB in 5 minutes (~2 MB/s)
2. MelSpectrogram completed (small file)
3. TextDecoder folder created but INCOMPLETE (only analytics subfolder, 4K)
4. AudioEncoder not even started
5. 611MB temp file orphaned (wasted bandwidth)

### Test 2: Retry with Partial Cache

**Initial state**: Partial cache from Test 1
- MelSpectrogram.mlmodelc: 372K (complete)
- TextDecoder.mlmodelc: 4K (INCOMPLETE)
- AudioEncoder.mlmodelc: Missing

**Command run** (same as Test 1):
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

**Result**: **TIMEOUT after 300 seconds**

**Cache state after second attempt**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*
400K    AudioEncoder.mlmodelc      # NEW but INCOMPLETE (empty weights folder!)
372K    MelSpectrogram.mlmodelc    # Unchanged (skipped - already existed)
4.0K    TextDecoder.mlmodelc       # Unchanged (SKIPPED despite being incomplete!)
```

**New orphaned temp file**:
```
477M    CFNetworkDownload_vx8Yul.tmp   # NEW partial download from run 2
```

**CRITICAL FINDINGS**:

| File | Run 1 | Run 2 | Expected | Bug? |
|------|-------|-------|----------|------|
| MelSpectrogram | 372K | 372K (skipped) | ~372K | No - correctly skipped |
| TextDecoder | 4K | 4K (skipped!) | ~1.2GB | **YES - incomplete file treated as complete!** |
| AudioEncoder | Missing | 400K (incomplete) | ~600MB | Partial - weights folder empty |

### Orphaned Temp Files Summary

| File | Size | Date | Source |
|------|------|------|--------|
| CFNetworkDownload_1iOtaj.tmp | 611MB | Jan 11 11:25 | Test 1 - partial download |
| CFNetworkDownload_vx8Yul.tmp | 477MB | Jan 11 11:32 | Test 2 - partial download |
| CFNetworkDownload_O2M1Za.tmp | 362MB | Jan 11 00:20 | Previous session |
| CFNetworkDownload_YeAiwA.tmp | 318MB | Jan 11 00:57 | Previous session |

**Total wasted disk space**: ~1.8GB of orphaned temp files

---

### Test 3: Direct Swift Bridge Download (No Python Timeout)

After cleaning up the corrupt cache, ran the Swift bridge directly without Python's 300s timeout.

**Cleanup performed**:
```bash
rm -rf ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/
rm -f /private/var/folders/*/T/CFNetworkDownload_*.tmp
```

**Command run**:
```bash
tools/whisperkit-bridge/.build/release/whisperkit-bridge tests/jfk.wav --model large-v3 --format json
```

**First attempt** (failed after 23:50):
- Downloaded AudioEncoder completely (1.2GB)
- Downloaded MelSpectrogram completely (392K)
- Started TextDecoder but CDN timed out
- Error: `NSURLErrorDomain Code=-1001 "The request timed out"`

**Second attempt** (succeeded after 27:31):
- Skipped AudioEncoder (already complete)
- Skipped MelSpectrogram (already complete)
- Completed TextDecoder download (1.7GB)
- Downloaded tokenizer files (2.6MB)
- **Success!** Transcription returned correctly

**Total download time**: ~51 minutes across two attempts (network conditions vary)

**Final cache state**:
```
$ du -sh ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/*
1.2G    AudioEncoder.mlmodelc
4.0K    config.json
4.0K    generation_config.json
392K    MelSpectrogram.mlmodelc
2.6M    models (tokenizer)
1.7G    TextDecoder.mlmodelc

Total: 2.9GB
```

---

### Test 4: Cached Model Performance

With the model fully cached, ran the Python benchmark to verify cached behavior.

**Command run**:
```bash
.venv/bin/python3 test_benchmark2.py large 1 WhisperKitImplementation
```

**Terminal output**:
```
2026-01-11 13:59:31 [info] Using WhisperKit large-v3 model (requested: large, using: large-v3)
2026-01-11 13:59:31 [info] Transcribing with WhisperKit via Swift bridge (model: large-v3)
2026-01-11 13:59:38 [info] WhisperKit transcription time: 3.0519s
2026-01-11 13:59:38 [info] Using internal transcription time: 3.0519s (total with overhead: 6.9720s)

=== Benchmark Summary for 'large' model ===
Implementation         Avg Time (s)    Parameters
--------------------------------------------------------------------------------
whisperkit             3.0519          model=large-v3, backend=WhisperKit Swift Bridge
    "And so my fellow Americans, ask not what your country can do for you..."
```

**Key metrics**:

| Metric | Value |
|--------|-------|
| Total time (with overhead) | **6.97 seconds** |
| Internal transcription time | **3.05 seconds** |
| Subprocess overhead | ~3.9 seconds |
| Download required | **NO** |

**Comparison with small model** (from previous session):

| Model | Total (cached) | Internal | Subprocess Overhead | Model Size |
|-------|----------------|----------|---------------------|------------|
| small | 1.37s | 0.44s | ~0.93s | 487MB |
| large-v3 | 6.97s | 3.05s | ~3.9s | 2.9GB |

**Observations**:
1. Large model subprocess overhead is ~4x higher than small (likely due to CoreML model loading time)
2. Internal transcription is ~7x slower than small (expected for 6x larger model)
3. Cached behavior works correctly - no re-download attempted

---

## Known Issues / Bugs Discovered

### P0 (Critical) - Timeout Insufficient for Large Model

**Problem**: The 300-second (5 minute) subprocess timeout is insufficient for downloading the ~2GB large-v3 model.

**Impact**: Large model download **always fails** on first run. Users cannot use large models without manual intervention.

**Evidence**:
- Empirical: 611MB downloaded in 300 seconds (~2 MB/s)
- At 2 MB/s, ~2GB requires ~1000 seconds (~17 minutes)
- 300s timeout is **3x too short**

**Location**: **[PROJECT]** `src/mac_whisper_speedtest/implementations/whisperkit.py:128`
```python
timeout=300  # 5 minute timeout for model download on first run
```

**Recommended Fix**:
```python
timeout=1200  # 20 minute timeout for large model download
```

**Effort**: 1 line change
**Priority**: P0 - Blocks large model usage entirely

---

### P0 (Critical) - No Completeness Check on Cached Files

**Problem**: The Hub library only checks `FileManager.default.fileExists()` - it does NOT verify file size, checksum, or completeness. Partially downloaded files are treated as complete.

**Impact**:
- Incomplete downloads create corrupt cache entries
- Subsequent runs skip re-downloading incomplete files
- **Permanent failure state** - must manually delete cache to recover

**Evidence**:
- 4K TextDecoder.mlmodelc folder (should be ~1.2GB) was SKIPPED on run 2
- The folder exists (contains only `analytics/` subfolder), so `fileExists()` returns true

**Location**: **[LIBRARY]** `swift-transformers/Sources/Hub/HubApi.swift:204-206`
```swift
var downloaded: Bool {
    FileManager.default.fileExists(atPath: destination.path)
}
```

**Recommended Fix** (in HubApi.swift):
```swift
var downloaded: Bool {
    guard FileManager.default.fileExists(atPath: destination.path) else { return false }
    // For directories (like .mlmodelc), check for key files
    if destination.pathExtension == "mlmodelc" {
        let weightPath = destination.appendingPathComponent("weights/weight.bin")
        return FileManager.default.fileExists(atPath: weightPath.path)
    }
    // For regular files, could add size check via HEAD request
    return true
}
```

**Effort**: ~20 lines in library code
**Priority**: P0 - Causes permanent failure state

---

### P1 (High) - No Download Resume Capability

**Problem**: Downloads use URLSession which writes to temp files. If interrupted, temp files are orphaned and download restarts from zero on next attempt.

**Impact**:
- Large model downloads waste bandwidth on retry
- ~1.8GB of orphaned temp files observed
- Users on slow connections may never complete large model download

**Evidence**:
- Run 1: Created 611MB temp file, timed out
- Run 2: Created NEW 477MB temp file (didn't resume 611MB file)

**Location**: **[LIBRARY]** `swift-transformers/Sources/Hub/Downloader.swift:78`
```swift
self.urlSession?.downloadTask(with: request).resume()
```

**Recommended Fix**: Implement Range header support for resumable downloads, or use background URLSession with proper state persistence.

**Effort**: Large (significant library changes)
**Priority**: P1 - Wastes bandwidth, poor UX

---

### P2 (Medium) - Orphaned Temp Files Not Cleaned Up

**Problem**: When downloads are interrupted (timeout, crash, Ctrl+C), temp files in `/private/var/folders/*/T/CFNetworkDownload_*.tmp` are not cleaned up.

**Impact**:
- Disk space waste (observed 1.8GB)
- No automatic cleanup mechanism
- Users unaware of accumulated temp files

**Location**: System-level URLSession behavior

**Recommended Fix**: Add cleanup logic to delete stale CFNetworkDownload temp files on startup, or document manual cleanup procedure.

**Effort**: ~20 lines
**Priority**: P2 - Disk space waste

---

### P2 (Medium) - No Progress Feedback During Download

**Problem**: Model download happens inside subprocess with no progress output. Users see the process hang for minutes with no feedback.

**Impact**: Poor UX - users may think process is frozen and kill it (causing partial download)

**Location**: **[BRIDGE]** `main.swift` - no progress callback implemented

**Recommended Fix**: Add progress output to stderr from Swift bridge:
```swift
let modelFolder = try await Self.download(variant: modelVariant, ...) { progress in
    fputs("Download progress: \(Int(progress.fractionCompleted * 100))%\n", stderr)
}
```

**Effort**: ~10 lines
**Priority**: P2 - Poor UX

---

## Recommended Improvements

### Priority Summary

| Priority | Issue | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| P0 | Timeout insufficient (300s) | 1 line | Blocks large model | Not Fixed |
| P0 | No completeness check | ~20 lines (library) | Permanent failure state | Not Fixed |
| P1 | No download resume | Large | Bandwidth waste | Not Fixed |
| P2 | Orphaned temp files | ~20 lines | Disk waste | Not Fixed |
| P2 | No progress feedback | ~10 lines | Poor UX | Not Fixed |

### Implementation Order Recommendation

**Phase 1: Immediate Fixes (Project-level)**
- [ ] Increase timeout from 300s to 1200s in `whisperkit.py:128`
- [ ] Add documentation warning about first-run download time for large models
- [ ] Add cleanup script for orphaned temp files

**Phase 2: Bridge Improvements**
- [ ] Add progress output to stderr during download
- [ ] Add `--timeout` argument to bridge for configurable timeout

**Phase 3: Library Fixes (Requires PR to swift-transformers)**
- [ ] Add completeness check for .mlmodelc directories
- [ ] Implement resumable downloads with Range headers

---

## Workarounds

### Manual Large Model Download

If the timeout prevents automatic download, manually run the bridge with no timeout:

```bash
# Run bridge directly (will complete download)
cd tools/whisperkit-bridge
.build/release/whisperkit-bridge /path/to/any/audio.wav --model large-v3

# This will download the model (may take 15-20 minutes)
# Once complete, subsequent benchmark runs will use cached model
```

### Clean Corrupt Cache

If cache is in a corrupt state (incomplete files treated as complete):

```bash
# Delete the specific model folder
rm -rf ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/

# Clean orphaned temp files
rm -f /private/var/folders/*/T/CFNetworkDownload_*.tmp
```

### Use Turbo Model Instead

The `large-v3-turbo` model is significantly smaller (~500MB vs ~2GB) and may complete within the timeout:

```bash
.venv/bin/python3 test_benchmark2.py large-v3-turbo 1 WhisperKitImplementation
```

---

## Key Source Files

### Project Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `whisperkit.py` | Implementation wrapper | 128 (timeout), 58-67 (model mapping) |
| `test_benchmark2.py` | Test entry point | 88-94 |

### Bridge Files

| File | Purpose |
|------|---------|
| `tools/whisperkit-bridge/Sources/whisperkit-bridge/main.swift` | Swift CLI, model init |

### Library Files (Read-Only)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `WhisperKit/Core/WhisperKit.swift` | Model setup, download trigger | 298-330 |
| `swift-transformers/Hub/HubApi.swift` | Download logic, caching | 204-206 (exists check), 234-257 (snapshot) |
| `swift-transformers/Hub/Downloader.swift` | URLSession download | 78, 112-119 |

---

## Notes

### Cache Location Difference

WhisperKit uses `~/Documents/huggingface/` (Swift Hub default) while Python libraries use `~/.cache/huggingface/hub/`. Models are **NOT shared** between them.

### Model Sizes (Approximate)

| Model | AudioEncoder | TextDecoder | Total |
|-------|--------------|-------------|-------|
| tiny | ~40MB | ~40MB | ~80MB |
| small | ~180MB | ~305MB | ~487MB |
| large-v3 | ~600MB | ~1.2GB | ~2GB |
| large-v3-turbo | ~300MB | ~200MB | ~500MB |

### Internal Timing

The benchmark uses WhisperKit's internal `fullPipeline` timing which excludes:
- Download time
- Subprocess startup
- Temp file I/O

This means first-run times (with download) show internal timing only, not the actual wall-clock time experienced.
