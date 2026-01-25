# WhisperKit Upgrade Plan: v0.13.1 → v0.15.0

**Created:** 2026-01-14
**Purpose:** Phased upgrade strategy for WhisperKit Swift dependency with discrete commits

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Version History & Breaking Changes](#version-history--breaking-changes)
4. [Phased Upgrade Strategy](#phased-upgrade-strategy)
5. [Implementation Details](#implementation-details)
6. [Coordinated Dependencies](#coordinated-dependencies)
7. [Git Workflow & Commit Strategy](#git-workflow--commit-strategy)
8. [Risk Mitigation](#risk-mitigation)
9. [Testing Checklist](#testing-checklist)

---

## Executive Summary

### The Gap

| Current | Target | Gap | Risk Level |
|---------|--------|-----|------------|
| v0.13.1 | v0.15.0 | 2 minor versions | **MEDIUM** |

### Key Finding: Lower Risk Than FluidAudio

Unlike FluidAudio (9 versions, major API changes), WhisperKit's upgrade is **relatively straightforward**:
- Only 2 minor versions
- One breaking change (`TranscriptionResult` struct→class)
- **Our bridge code pattern is unaffected** by the breaking change

### Recommended Approach: 2-Phase Incremental Upgrade

| Phase | Target Version | Key Changes | Bridge Code Impact |
|-------|---------------|-------------|-------------------|
| **Phase 1** | v0.14.1 | swift-transformers 1.0+, Swift 6 prep | None expected |
| **Phase 2** | v0.15.0 | TranscriptionResult class, swift-transformers 1.1.2 | None expected |

**Why still do 2 phases?**
1. The swift-transformers major version jump (0.1.15 → 1.0+) is significant
2. Phase 1 validates the HubApi download behavior still works
3. Each phase has a stable checkpoint for rollback

---

## Current State Analysis

### Bridge Code Structure

```
tools/whisperkit-bridge/
├── Package.swift          # Declares WhisperKit 0.13.1
├── Package.resolved       # Locks exact versions
└── Sources/whisperkit-bridge/
    └── main.swift         # ~86 lines, simple CLI
```

### Current Resolved Dependencies

```
whisperkit: 0.13.1
swift-transformers: 0.1.15  ← Will jump to 1.1.2+
swift-argument-parser: 1.6.1
swift-collections: 1.2.1
jinja: 1.2.1
```

### Current API Usage (v0.13.1)

From `main.swift:30-44`:
```swift
// Initialize WhisperKit
let config = WhisperKitConfig(model: model)
let whisperKit = try await WhisperKit(config)

// Transcribe
let results = try await whisperKit.transcribe(audioPath: inputFile)

// Access results (TranscriptionResult is currently a struct)
let firstResult = results.first
let transcriptionTime = firstResult?.timings.fullPipeline ?? 0.0
let combinedText = results.map { $0.text }.joined(separator: " ")

// Access segments
result.segments.map { segment in
    ["start": segment.start, "end": segment.end, "text": segment.text]
}
```

### Why Our Code Is Unaffected by the Breaking Change

The `TranscriptionResult` struct→class change matters when:
1. **Copying for isolation** — We don't copy results
2. **Subclassing** — We don't subclass
3. **Concurrent access** — We access sequentially in one task
4. **Array mutations** — We only read, never mutate

Our usage pattern (read-only property access) works identically for both struct and class.

---

## Version History & Breaking Changes

### Complete Version Timeline

| Version | Date | Category | Key Changes |
|---------|------|----------|-------------|
| **v0.13.1** | Jul 31 | Fixes | Tokenizer respects `downloadBase`, offline loading, logit filter fix |
| **v0.14.0** | Sep 20 | Feature | **WhisperKit Local Server**, EnergyVAD public, tokenizer improvements |
| **v0.14.1** | Oct 17 | Compat | **swift-transformers 1.0+**, Swift 6 concurrency, Sendable conformance |
| **v0.15.0** | Nov 7 | **Breaking** | **TranscriptionResult: struct→class**, swift-transformers 1.1.2 |

### Breaking Change Analysis

#### v0.15.0: TranscriptionResult Struct → Class

**What changed:**
```swift
// Before (v0.13.1 - v0.14.1)
public struct TranscriptionResult { ... }

// After (v0.15.0)
open class TranscriptionResult: Codable, @unchecked Sendable { ... }
```

**Impact:**
| Scenario | Old (Struct) | New (Class) | Our Code |
|----------|--------------|-------------|----------|
| Copy semantics | Value copy | Reference copy | ✅ Not affected (read-only) |
| Array of results | Independent values | Shared references | ✅ Not affected (no mutation) |
| Subclassing | Not possible | Supported | ✅ Not affected (don't subclass) |
| Thread safety | N/A (value type) | NSLock-protected | ✅ Not affected (single-threaded) |

**Conclusion:** No code changes required in bridge.

#### v0.14.1: swift-transformers Major Version

**What changed:**
```
swift-transformers: 0.1.15 → 1.0.0+
```

**Potential Impact Areas:**
- `HubApi` for model downloads
- Tokenizer loading
- Cache directory handling

**From model_details documentation:** The critical download behavior (file existence check, cache location) is in `swift-transformers/Sources/Hub/HubApi.swift`. We need to verify this still works.

---

## Phased Upgrade Strategy

### Phase 1: swift-transformers 1.0+ (v0.13.1 → v0.14.1)

**Goal:** Get swift-transformers major version upgrade safely

**What changes:**
- swift-transformers 0.1.15 → 1.0.x
- Swift 6 concurrency compatibility
- Sendable conformance to public structs
- TranscribeTask subclassing hooks (we don't use)

**Bridge code changes required:**
- **None expected**

**Why stop at v0.14.1?**
- Tests swift-transformers 1.0+ compatibility
- Isolates any HubApi changes
- Stable checkpoint before class conversion

**Verification focus:**
- Model download still works
- Cache location unchanged (`~/Documents/huggingface/`)
- Tokenizer loading works

**Commits:**
1. `chore(deps): bump WhisperKit 0.13.1 → 0.14.1`
2. `test(bridge): verify model download and transcription`

---

### Phase 2: TranscriptionResult Class (v0.14.1 → v0.15.0)

**Goal:** Get latest WhisperKit with TranscriptionResult as class

**What changes:**
- `TranscriptionResult` promoted from struct to open class
- swift-transformers 1.0.x → 1.1.2
- Thread-safe property access via `TranscriptionPropertyLock`

**Bridge code changes required:**
- **None expected** (read-only usage pattern)

**Why this is the final phase:**
- Latest stable version
- All breaking changes absorbed
- Ready for future updates

**Verification focus:**
- `results.first?.timings.fullPipeline` still works
- `results.map { $0.text }` still works
- Segment access unchanged

**Commits:**
1. `chore(deps): bump WhisperKit 0.14.1 → 0.15.0`
2. `test(bridge): verify TranscriptionResult class compatibility`

---

## Implementation Details

### Phase 1 Step-by-Step

#### Step 1.1: Create Upgrade Branch
```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_MAIN
git checkout main
git pull origin main
git checkout -b chore/whisperkit-upgrade-phase1
```

#### Step 1.2: Update Package.swift
```swift
// tools/whisperkit-bridge/Package.swift
// Change:
.package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.13.1"),
// To:
.package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.14.1"),
```

#### Step 1.3: Resolve & Update
```bash
cd tools/whisperkit-bridge
swift package resolve
swift package update

# Check what versions were resolved
cat Package.resolved | grep -A3 '"identity"'
```

**Expected Package.resolved changes:**
```
whisperkit: 0.13.1 → 0.14.1
swift-transformers: 0.1.15 → 1.0.x (major jump!)
```

#### Step 1.4: Build
```bash
swift build -c release
```

**If compilation errors occur:**
- Check for API changes in swift-transformers
- May need to update `Hub` or `HubApi` usage patterns

#### Step 1.5: Test Model Download
```bash
# Optionally clear cache to test fresh download
# mv ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-small \
#    ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-sma__OFF__ll

# Test with existing cache first
./tools/whisperkit-bridge/.build/release/whisperkit-bridge tests/jfk.wav --model small --format json
```

#### Step 1.6: Test Python Integration
```bash
.venv/bin/mac-whisper-speedtest -b -m small -n 2 -i WhisperKitImplementation
```

#### Step 1.7: Commit
```bash
git add tools/whisperkit-bridge/Package.swift
git add tools/whisperkit-bridge/Package.resolved

git commit -m "$(cat <<'EOF'
chore(deps): bump WhisperKit 0.13.1 → 0.14.1

Phase 1 of WhisperKit upgrade (v0.13.1 → v0.15.0)

Changes:
- Update Package.swift dependency to 0.14.1
- swift-transformers upgraded from 0.1.15 to 1.0.x
- Swift 6 concurrency compatibility improvements
- Sendable conformance added to public structs

Tested:
- Swift bridge builds successfully
- Model download works correctly
- Transcription produces correct output
- Python benchmark runs without errors

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Phase 2 Step-by-Step

#### Step 2.1: Update Package.swift
```swift
// Change:
.package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.14.1"),
// To:
.package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.15.0"),
```

#### Step 2.2: Resolve & Build
```bash
cd tools/whisperkit-bridge
swift package resolve
swift package update
swift build -c release
```

#### Step 2.3: Verify TranscriptionResult Access
```bash
# Test that result access still works
./tools/whisperkit-bridge/.build/release/whisperkit-bridge tests/jfk.wav --model small --format json
```

**Expected output should include:**
- `"text"`: transcription text
- `"transcription_time"`: timing value
- `"segments"`: array of segments

#### Step 2.4: Full Test Suite
```bash
# Small model
.venv/bin/mac-whisper-speedtest -b -m small -n 2 -i WhisperKitImplementation

# If large model is cached, test that too
.venv/bin/mac-whisper-speedtest -b -m large -n 1 -i WhisperKitImplementation
```

#### Step 2.5: Commit
```bash
git add tools/whisperkit-bridge/Package.swift
git add tools/whisperkit-bridge/Package.resolved

git commit -m "$(cat <<'EOF'
chore(deps): bump WhisperKit 0.14.1 → 0.15.0

Phase 2 (final) of WhisperKit upgrade (v0.13.1 → v0.15.0)

Changes:
- Update Package.swift dependency to 0.15.0
- TranscriptionResult converted from struct to open class
- swift-transformers upgraded to 1.1.2

Note: TranscriptionResult class change has no impact on our
bridge code as we only perform read-only property access.

Tested:
- Swift bridge builds successfully
- result.timings.fullPipeline access works
- result.text and result.segments access works
- Python benchmark runs without errors

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Coordinated Dependencies

### Relationship with swift-transformers

WhisperKit depends on `swift-transformers` for:
1. **HubApi** — Model downloads from HuggingFace
2. **Tokenizer** — Text tokenization
3. **Hub caching** — Local model storage

The swift-transformers upgrade from 0.1.15 → 1.1.2 is **the real upgrade** happening under the hood.

### Potential swift-transformers Changes to Watch

| Area | What to Check |
|------|--------------|
| Cache location | Still `~/Documents/huggingface/`? |
| Download API | `Hub.snapshot()` signature unchanged? |
| File existence check | Still `FileManager.default.fileExists()`? |
| Temp file handling | Same CFNetworkDownload behavior? |

**From v0.13.1 documentation:** The P0 bug (no completeness check) is in swift-transformers `HubApi.swift:204-206`. Verify this hasn't changed (for better or worse).

### When to Upgrade swift-transformers Independently

If you ever need to force a specific swift-transformers version:

```swift
// Package.swift - NOT recommended unless necessary
dependencies: [
    .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.15.0"),
    // Override WhisperKit's swift-transformers version
    .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.1.2"),
]
```

**Warning:** This can cause version conflicts. Only do this if you encounter specific issues.

---

## Git Workflow & Commit Strategy

### Branch Naming

```
chore/whisperkit-upgrade-phase1   # Phase 1: v0.13.1 → v0.14.1
chore/whisperkit-upgrade-phase2   # Phase 2: v0.14.1 → v0.15.0
```

**Or single branch:**
```
chore/whisperkit-upgrade-v0.15
```

### Commit Message Format

```
chore(deps): bump WhisperKit X.Y.Z → A.B.C

Phase N of WhisperKit upgrade (v0.13.1 → v0.15.0)

Changes:
- List key changes from release notes

Tested:
- List what was tested

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Coordinating with FluidAudio Upgrade

If doing both upgrades:

**Option A: Sequential (Recommended for first time)**
```
1. Complete FluidAudio Phase 1, merge
2. Complete FluidAudio Phase 2, merge
3. Complete FluidAudio Phase 3, merge
4. Complete WhisperKit Phase 1, merge
5. Complete WhisperKit Phase 2, merge
```

**Option B: Parallel phases**
```
1. FluidAudio Phase 1 + WhisperKit Phase 1 in parallel branches
2. Merge both
3. Continue with Phase 2 for each
```

**Recommendation:** Sequential is safer for learning and debugging.

---

## Risk Mitigation

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| swift-transformers API change | Low | Medium | Test download flow in Phase 1 |
| TranscriptionResult access breaks | Very Low | Low | Our usage pattern is safe |
| Build errors from Swift 6 | Low | Medium | Fix Sendable warnings |
| Cache location change | Very Low | High | Verify in Phase 1 |

### Rollback Procedures

**If Phase 1 fails:**
```bash
git checkout main -- tools/whisperkit-bridge/
cd tools/whisperkit-bridge
swift package resolve
swift build -c release
```

**If Phase 2 fails (but Phase 1 worked):**
```bash
# Revert to Phase 1 state
# In Package.swift, change back to from: "0.14.1"
swift package resolve
swift build -c release
```

### Known Issues from Current Version

From `model_details_WhisperKitImplementation.md`:

| Issue | Status | Affected by Upgrade? |
|-------|--------|---------------------|
| P0: Timeout insufficient (300s) | Not fixed | No |
| P0: No completeness check | Not fixed | **Maybe** — check swift-transformers |
| P1: No download resume | Not fixed | No |
| P2: Orphaned temp files | Not fixed | No |

**Action:** After Phase 1, verify the completeness check behavior is unchanged (still only `fileExists()`).

---

## Testing Checklist

### Pre-Upgrade Baseline
- [ ] Record current benchmark: `.venv/bin/mac-whisper-speedtest -b -m small -n 3 -i WhisperKitImplementation`
- [ ] Note exact transcription output for JFK audio
- [ ] Verify cache location: `ls ~/Documents/huggingface/models/argmaxinc/`

### Phase 1 Verification
- [ ] Swift build completes without errors
- [ ] No new compiler warnings (or document them)
- [ ] Bridge --help works
- [ ] JFK transcription matches baseline exactly
- [ ] `transcription_time` field still present in JSON output
- [ ] `segments` array still present in JSON output
- [ ] Cache location unchanged
- [ ] Python benchmark completes successfully

### Phase 2 Verification
- [ ] All Phase 1 checks pass
- [ ] `results.first?.timings.fullPipeline` still works
- [ ] `results.map { $0.text }` still works
- [ ] Segment timestamps still accurate
- [ ] No reference/mutation issues (should be N/A for our code)

### Post-Upgrade Validation
- [ ] Compare benchmark times to baseline (should be similar)
- [ ] Compare transcription text (should be identical)
- [ ] Test with longer audio (ted_60.wav if available)
- [ ] Document any behavior changes

---

## Summary

### Quick Reference: Upgrade Commands

```bash
# === PHASE 1 ===
git checkout -b chore/whisperkit-upgrade-phase1
# Edit Package.swift: from: "0.14.1"
cd tools/whisperkit-bridge && swift package update && swift build -c release
.venv/bin/mac-whisper-speedtest -b -m small -n 2 -i WhisperKitImplementation
git add . && git commit -m "chore(deps): WhisperKit 0.13.1 → 0.14.1"

# === PHASE 2 ===
# Edit Package.swift: from: "0.15.0"
cd tools/whisperkit-bridge && swift package update && swift build -c release
.venv/bin/mac-whisper-speedtest -b -m small -n 2 -i WhisperKitImplementation
git add . && git commit -m "chore(deps): WhisperKit 0.14.1 → 0.15.0"
```

### Key Takeaways

1. **Lower risk than FluidAudio** — Only 2 versions, our code pattern unaffected
2. **Real upgrade is swift-transformers** — 0.1.15 → 1.1.2 is the major change
3. **TranscriptionResult change is safe for us** — Read-only access works for both struct and class
4. **Verify download behavior** — The HubApi is in swift-transformers, confirm it still works
5. **Keep phases discrete** — Easy rollback if issues arise

### Comparison: WhisperKit vs FluidAudio Upgrades

| Aspect | FluidAudio | WhisperKit |
|--------|------------|------------|
| Version gap | 9 minor versions | 2 minor versions |
| Breaking changes | API enum (critical) | Struct→Class (non-issue for us) |
| Bridge code changes | Required (add `.v2`) | None expected |
| Risk level | High | Medium |
| Effort | Medium-High | Low |
| Recommended phases | 3 | 2 |

---

## Sources

- [WhisperKit Releases](https://github.com/argmaxinc/WhisperKit/releases)
- [WhisperKit v0.15.0 Documentation](https://swiftpackageindex.com/argmaxinc/WhisperKit/v0.15.0/documentation/whisperkit)
- [swift-transformers Repository](https://github.com/huggingface/swift-transformers)
- [TranscriptionResult Source](https://github.com/argmaxinc/WhisperKit/blob/main/Sources/WhisperKit/Core/Models.swift)
