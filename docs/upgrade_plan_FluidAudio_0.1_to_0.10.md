# FluidAudio Upgrade Plan: v0.1.0 → v0.10.0

**Created:** 2026-01-14
**Updated:** 2026-01-25 (added automated test suite, fixed documentation)
**Purpose:** Phased upgrade strategy for FluidAudio Swift dependency with discrete commits

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Automated Test Suite](#automated-test-suite)
3. [Current State Analysis](#current-state-analysis)
4. [Version History & Breaking Changes](#version-history--breaking-changes)
5. [Phased Upgrade Strategy](#phased-upgrade-strategy)
6. [Implementation Details](#implementation-details)
7. [Best Practices for Coordinated Updates](#best-practices-for-coordinated-updates)
8. [Git Workflow & Commit Strategy](#git-workflow--commit-strategy)
9. [Risk Mitigation](#risk-mitigation)
10. [Testing Checklist](#testing-checklist)

---

## Executive Summary

### The Gap
| Current | Target | Gap | Risk Level |
|---------|--------|-----|------------|
| v0.1.0 | v0.10.0 | 9 minor versions | HIGH |

### Recommended Approach: 3-Phase Incremental Upgrade

Rather than jumping 9 versions at once, we'll upgrade in 3 phases at natural architectural breakpoints:

| Phase | Target Version | Key Changes | Effort |
|-------|---------------|-------------|--------|
| **Phase 1** | v0.4.1 | Model version enum, stability fixes | Low |
| **Phase 2** | v0.8.2 | Streaming ASR, AudioConverter, features | Medium |
| **Phase 3** | v0.10.0 | Swift 6, Sortformer, final polish | Medium |

**Why not jump directly to v0.10.0?**
1. Easier to debug if something breaks (narrower diff)
2. Each phase has a stable checkpoint
3. API changes compound—incremental upgrades let you adapt gradually
4. Git history shows clear progression

> **Note on SPM Behavior:** Swift Package Manager's `from: "0.4.1"` means "0.4.1 or higher",
> so it will resolve to the latest compatible version. During the WhisperKit upgrade, this
> caused both phases to complete in one step. To force specific intermediate versions,
> use `exact: "0.4.1"` constraint instead.

---

## Automated Test Suite

### Test Files

Run these tests **before** and **after** each upgrade phase:

| Test File | Purpose | Run Time |
|-----------|---------|----------|
| `tests/test_fluidaudio_health.py` | Bridge existence, JSON schema, version checks | ~5s |
| `tests/test_fluidaudio_transcription.py` | Actual transcription smoke tests | ~60s |
| `scripts/verify_fluidaudio_upgrade.py` | Quick CLI verification script | ~10s |

### Quick Commands

```bash
# Run all FluidAudio tests (before/after each phase)
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v

# Health checks only (fast)
pytest tests/test_fluidaudio_health.py -v

# Transcription smoke tests only
pytest tests/test_fluidaudio_transcription.py -v

# Quick verification script (alternative to pytest)
python scripts/verify_fluidaudio_upgrade.py
python scripts/verify_fluidaudio_upgrade.py --verbose
python scripts/verify_fluidaudio_upgrade.py --skip-transcription  # Fast mode
```

### What the Tests Verify

**Health Checks (`test_fluidaudio_health.py`):**
- Bridge executable exists and runs
- JSON output has required fields (text, transcription_time, processing_time, language)
- Package.resolved versions are in expected ranges (0.1.x → 0.4.x → 0.8.x → 0.10.x)
- Python implementation imports and instantiates correctly

**Transcription Tests (`test_fluidaudio_transcription.py`):**
- Bridge CLI produces non-empty transcription
- Transcription contains expected JFK speech words ("ask", "country", "fellow")
- JSON schema consistency across upgrades
- Python implementation wrapper works correctly

### JSON Output Schema

FluidAudio's JSON schema differs from WhisperKit:

```json
{
    "text": "transcription text...",
    "transcription_time": 0.5,
    "processing_time": 0.3,
    "language": "en"
}
```

**Note:** FluidAudio does **NOT** provide a `segments` array (unlike WhisperKit).

### Baseline Recording

Before starting the upgrade, save baseline outputs for comparison:

```bash
# Save JSON baseline
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav \
    --format json > docs/fluidaudio_baseline_0.1.0.json

# Save Package.resolved snapshot
cp tools/fluidaudio-bridge/Package.resolved docs/fluidaudio_package_resolved_0.1.0.json
```

---

## Current State Analysis

### Bridge Code Structure

```
tools/fluidaudio-bridge/
├── Package.swift          # Declares FluidAudio dependency (from: "0.0.3")
├── Package.resolved       # Locks to v0.1.0
└── Sources/fluidaudio-bridge/
    └── main.swift         # ~110 lines, simple CLI
```

> **Note:** Package.swift declares `from: "0.0.3"` but Package.resolved locks to v0.1.0.
> This is normal - SPM resolved to the latest compatible version at build time.

### Current API Usage (v0.1.0)

From `main.swift:28-51`:
```swift
// Configuration
let asrConfig = ASRConfig(
    maxSymbolsPerFrame: 3,
    realtimeMode: false,
    chunkSizeMs: 1500,
    tdtConfig: TdtConfig(
        durations: [0, 1, 2, 3, 4],
        maxSymbolsPerStep: 3
    )
)

// Model loading (hardcoded to parakeet-tdt-0.6b-v2)
let asrManager = AsrManager(config: asrConfig)
let models = try await AsrModels.downloadAndLoad()  // No version param!
try await asrManager.initialize(models: models)

// Transcription
let result = try await asrManager.transcribe(audioData)
```

### What Will Change

| API | v0.1.0 | v0.10.0 | Impact |
|-----|--------|---------|--------|
| `downloadAndLoad()` | No params | `version: AsrModelVersion = .v3` | **BREAKING** - defaults to v3 now |
| `transcribe()` | `[Float]` only | `[Float]`, `URL`, `AVAudioPCMBuffer` | Additive |
| Swift version | 5.9 | 6.0 required | Build system |
| Model version | v2 only | v2 or v3 | **Must specify `.v2`** |

---

## Version History & Breaking Changes

### Complete Version Timeline

| Version | Date | Category | Key Changes |
|---------|------|----------|-------------|
| **v0.1.0** | Jul 26 | Initial | VAD, Parakeet TDT ASR, modular architecture |
| v0.2.0 | Aug 8 | Performance | Metal optimization, Sendable compliance, Int8 quantization |
| v0.3.0 | Aug 25 | Feature | Streaming diarization, proxy support, vocabulary improvements |
| **v0.4.0** | Sep 4 | **API** | **`AsrModelVersion` enum (.v2/.v3)**, Silero VAD v2 |
| v0.4.1 | Sep 8 | Stability | Confidence/token timestamps fixes |
| v0.5.0 | Sep 14 | Performance | 10x faster VAD, AudioConverter cleanup, CocoaPods |
| v0.5.2 | Sep 20 | Performance | Streaming VAD, parakeet-tdt-v3 models, ~50% faster |
| v0.6.0 | Sep 26 | Architecture | Model compilation logging, MelEncoder refactor |
| v0.6.1 | Sep 28 | Fix | Restored nvidia/parakeet-tdt-0.6b-v2, token timing fix |
| v0.7.0 | Oct 14 | Feature | Kokoro TTS, VAD probability streaming |
| **v0.7.2** | Oct 19 | **Platform** | **Minimum macOS 14, iOS 17** |
| v0.7.8 | Nov 4 | Performance | Removed shared buffers, 10% fewer missing words |
| v0.8.0 | Dec 17 | Feature | **Parakeet EOU streaming ASR** |
| v0.8.1 | Dec 26 | Feature | Transcription progress, speaker count constraints |
| v0.8.2 | Dec 30 | Feature | SSML support for TTS |
| **v0.9.0** | Dec 31 | **Breaking** | **Swift 6 full compatibility** |
| v0.9.1 | Jan 3 | Fix | Swift 6 Sendable compliance fixes |
| **v0.10.0** | Jan 12 | Feature | **Sortformer real-time speaker diarization** |

### Breaking Change Breakpoints

| Version | Breaking Change | Migration Required |
|---------|----------------|-------------------|
| **v0.4.0** | `AsrModelVersion` enum added | Must add `.v2` to maintain current behavior |
| **v0.7.2** | macOS 14 minimum | Already targeting macOS 14 ✅ |
| **v0.9.0** | Swift 6 required | Already have Swift 6.2.3 ✅ |

---

## Phased Upgrade Strategy

### Phase 1: Foundation (v0.1.0 → v0.4.1)

**Goal:** Get the model version API with minimal risk

**What changes:**
- `AsrModelVersion` enum added
- `downloadAndLoad(version:)` parameter
- Confidence/token timestamp improvements
- Silero VAD v2

**Bridge code changes required:**
```swift
// BEFORE (v0.1.0)
let models = try await AsrModels.downloadAndLoad()

// AFTER (v0.4.1)
let models = try await AsrModels.downloadAndLoad(version: .v2)  // Explicit!
```

**Why stop at v0.4.1?**
- Stable checkpoint before major performance changes
- Establishes the version enum pattern
- Low-risk first step

**Commits:**
1. `chore(deps): bump FluidAudio 0.1.0 → 0.4.1`
2. `fix(bridge): specify AsrModelVersion.v2 explicitly`
3. `test(bridge): verify transcription still works`

---

### Phase 2: Features & Performance (v0.4.1 → v0.8.2)

**Goal:** Get streaming capabilities and performance improvements

**What changes:**
- 10x faster VAD processing
- AudioConverter cleanup
- Parakeet v3 models available (multilingual)
- Streaming ASR with EOU detection
- Transcription progress reporting
- macOS 14 minimum (already targeting)

**Bridge code changes required:**

```swift
// Optional: Add progress reporting
let models = try await AsrModels.downloadAndLoad(version: .v2)

// Optional: Use URL-based transcription
let result = try await asrManager.transcribe(inputURL, source: .system)
```

**Why stop at v0.8.2?**
- Last stable version before Swift 6 requirement
- Full feature set available
- Good testing checkpoint

**Commits:**
1. `chore(deps): bump FluidAudio 0.4.1 → 0.8.2`
2. `feat(bridge): add URL-based transcription (optional)`
3. `test(bridge): verify with longer audio files`

---

### Phase 3: Swift 6 & Polish (v0.8.2 → v0.10.0)

**Goal:** Swift 6 compliance and latest features

**What changes:**
- Swift 6 full compatibility
- Sendable compliance throughout
- Sortformer real-time speaker diarization
- CLI renamed to `fluidaudiocli`

**Bridge code changes required:**

```swift
// Sendable compliance may require:
// - Adding @Sendable to closures
// - Ensuring thread-safe access patterns
// - Using actors or structured concurrency
```

**Why this is the final phase:**
- Swift 6 changes are pervasive
- Sortformer is additive (new capability)
- Clean stopping point

**Commits:**
1. `chore(deps): bump FluidAudio 0.8.2 → 0.10.0`
2. `fix(bridge): Swift 6 Sendable compliance`
3. `refactor(bridge): update for new API patterns`
4. `test(bridge): full regression test`

---

## Implementation Details

### Phase 1 Step-by-Step

#### Step 1.1: Create Upgrade Branch
```bash
cd /Users/rymalia/projects/mac-whisper-speedtest
git checkout main
git pull origin main
git checkout -b chore/fluidaudio-upgrade-phase1
```

#### Step 1.2: Update Package.swift
```swift
// tools/fluidaudio-bridge/Package.swift
// Change:
.package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.0.3"),
// To:
.package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.4.1"),

// NOTE: If you want to force exact version 0.4.1 (not latest), use:
// .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.4.1"),
```

#### Step 1.3: Resolve & Update
```bash
cd tools/fluidaudio-bridge
swift package resolve
swift package update
```

#### Step 1.4: Check for API Changes
```bash
# Try to build - expect compiler errors
swift build -c release 2>&1 | head -50
```

#### Step 1.5: Fix Bridge Code
```swift
// In main.swift, line 46:
// Change:
let models = try await AsrModels.downloadAndLoad()
// To:
let models = try await AsrModels.downloadAndLoad(version: .v2)
```

#### Step 1.6: Rebuild
```bash
swift build -c release
```

#### Step 1.7: Test
```bash
cd ../..
# Clear existing models to test fresh download (optional)
# rm -rf ~/Library/Application\ Support/FluidAudio/Models/

# Test with existing models
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav --format json

# Run Python benchmark
.venv/bin/mac-whisper-speedtest -b -m small -n 2 -i FluidAudioCoreMLImplementation
```

#### Step 1.8: Commit
```bash
git add tools/fluidaudio-bridge/Package.swift
git add tools/fluidaudio-bridge/Package.resolved
git add tools/fluidaudio-bridge/Sources/fluidaudio-bridge/main.swift

git commit -m "$(cat <<'EOF'
chore(deps): bump FluidAudio 0.1.0 → 0.4.1

Phase 1 of FluidAudio upgrade (v0.1.0 → v0.10.0)

Changes:
- Update Package.swift dependency to 0.4.1
- Add explicit .v2 model version to maintain behavior
- AsrModelVersion enum now available for future v3 support

Tested:
- Swift bridge builds successfully
- Transcription produces correct output
- Benchmark runs without errors

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Best Practices for Coordinated Updates

### 1. Dependency Update Order

When updating interconnected dependencies, follow this order:

```
1. Core/Foundation packages first (MLX, Swift runtime)
   └── These are lowest-level, affect everything above

2. Framework packages next (FluidAudio, WhisperKit)
   └── These depend on core packages

3. Wrapper/Integration packages last (Python bridges)
   └── These depend on frameworks
```

**For this project specifically:**
```
1. MLX (if updating)
2. FluidAudio Swift package
3. Python timeout fix in fluidaudio_coreml.py
```

### 2. Atomic Updates Within a Phase

**DO:** Update related packages together
```bash
# Good: Update both mlx and mlx-metal together
uv lock --upgrade-package mlx --upgrade-package mlx-metal
```

**DON'T:** Split tightly-coupled updates
```bash
# Bad: Update only one of a pair
uv lock --upgrade-package mlx
# Later...
uv lock --upgrade-package mlx-metal  # May cause version conflicts!
```

### 3. Lock File Hygiene

**Always commit lock files:**
```bash
# Swift
git add Package.resolved

# Python
git add uv.lock
```

**Why?** Lock files ensure:
- Reproducible builds across machines
- CI/CD gets exact same versions
- Rollback is reliable

### 4. Version Constraint Philosophy

**For stable dependencies (low churn):**
```swift
.package(url: "...", from: "0.4.1"),  // Allow patch updates
```

**For volatile dependencies (high churn):**
```swift
.package(url: "...", exact: "0.4.1"),  // Pin exactly
```

**Recommendation for FluidAudio:** Use `from:` during upgrade phases, then consider `exact:` after stabilizing at v0.10.0 if you want predictability.

### 5. Testing Strategy

**Minimum viable testing:**
```bash
# 1. Build passes
swift build -c release

# 2. Smoke test works
./bridge tests/jfk.wav --format json

# 3. Integration test passes
.venv/bin/mac-whisper-speedtest -b -m small -n 1 -i FluidAudioCoreMLImplementation
```

**Recommended testing:**
```bash
# All of the above, plus:

# 4. Multiple runs for consistency
.venv/bin/mac-whisper-speedtest -b -m small -n 5 -i FluidAudioCoreMLImplementation

# 5. Different audio lengths
./bridge tests/ted_60.wav --format json

# 6. Fresh download test (clear cache first)
rm -rf ~/Library/Application\ Support/FluidAudio/Models/
./bridge tests/jfk.wav --format json
```

---

## Git Workflow & Commit Strategy

### Branch Naming Convention

```
chore/fluidaudio-upgrade-phase1   # Phase 1: v0.1.0 → v0.4.1
chore/fluidaudio-upgrade-phase2   # Phase 2: v0.4.1 → v0.8.2
chore/fluidaudio-upgrade-phase3   # Phase 3: v0.8.2 → v0.10.0
```

**Alternative (single branch with tagged commits):**
```
chore/fluidaudio-upgrade-v0.10
  ├── Commit 1: Phase 1 (v0.4.1)
  ├── Commit 2: Phase 1 tests
  ├── Commit 3: Phase 2 (v0.8.2)
  ├── Commit 4: Phase 2 tests
  └── Commit 5: Phase 3 (v0.10.0)
```

### Commit Message Format

```
<type>(scope): brief description

<body explaining what and why>

<optional breaking change notes>

Tested:
- List what was tested

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Types:**
- `chore(deps)`: Dependency updates
- `fix(bridge)`: Bug fixes in bridge code
- `feat(bridge)`: New features in bridge
- `refactor(bridge)`: Code changes without behavior change
- `test(bridge)`: Test-only changes

### Discrete Commit Strategy

**Goal:** Each commit should be independently revertable

**Phase 1 commits:**
```
1. chore(deps): bump FluidAudio 0.1.0 → 0.4.1
   - Only Package.swift and Package.resolved
   - Build will FAIL at this point (expected)

2. fix(bridge): add explicit AsrModelVersion.v2
   - Only main.swift changes
   - Build passes after this commit

3. test(bridge): verify Phase 1 upgrade works
   - No code changes, just document test results
   - Could be squashed into commit 2
```

**Why discrete commits?**
- If v0.4.1 causes issues, you can revert just commit 1
- History shows exactly what changed for each step
- Easier to cherry-pick specific fixes

### Pull Request Strategy

**Option A: One PR per phase**
```
PR #1: FluidAudio Phase 1 (v0.1.0 → v0.4.1)
PR #2: FluidAudio Phase 2 (v0.4.1 → v0.8.2)  [after PR #1 merged]
PR #3: FluidAudio Phase 3 (v0.8.2 → v0.10.0) [after PR #2 merged]
```

**Option B: Single PR with phased commits**
```
PR: FluidAudio upgrade v0.1.0 → v0.10.0
  - Commit 1-3: Phase 1
  - Commit 4-6: Phase 2
  - Commit 7-9: Phase 3
```

**Recommendation:** Option A for high-risk upgrades (like this one), Option B for routine updates.

---

## Risk Mitigation

### Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model version change breaks output | Medium | High | Explicitly specify `.v2` |
| Swift 6 Sendable errors | High | Medium | Fix one at a time, compiler guides you |
| Download behavior changes | Low | Medium | Test fresh download path |
| Performance regression | Low | Low | Benchmark before/after |

### Rollback Procedures

**If Phase 1 fails:**
```bash
git checkout main -- tools/fluidaudio-bridge/
cd tools/fluidaudio-bridge
swift package resolve
swift build -c release
```

**If Phase 2 fails (but Phase 1 worked):**
```bash
# Revert to Phase 1 state
git log --oneline  # Find the Phase 1 final commit
git revert HEAD~N..HEAD  # Revert Phase 2 commits
# Or more surgically:
git checkout <phase1-commit> -- tools/fluidaudio-bridge/
```

**Nuclear option:**
```bash
git checkout main
git branch -D chore/fluidaudio-upgrade-phase1
# Start fresh
```

### Backup Current State

Before starting:
```bash
# Create a backup tag
git tag backup/pre-fluidaudio-upgrade

# Or create a branch
git branch backup/fluidaudio-v0.1.0
```

---

## Testing Checklist

### Pre-Upgrade Baseline

```bash
# 1. Run automated health checks (establishes baseline)
pytest tests/test_fluidaudio_health.py -v

# 2. Run transcription smoke tests
pytest tests/test_fluidaudio_transcription.py -v

# 3. Save JSON baseline for schema comparison
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav \
    --format json > docs/fluidaudio_baseline_0.1.0.json

# 4. Record Package.resolved versions
cat tools/fluidaudio-bridge/Package.resolved | grep -E '"identity"|"version"'

# 5. Record benchmark timing (save output for comparison)
.venv/bin/mac-whisper-speedtest -b -m small -n 3 -i FluidAudioCoreMLImplementation
```

- [ ] All pytest tests pass
- [ ] Baseline JSON saved
- [ ] Package.resolved versions recorded (FluidAudio 0.1.0)
- [ ] Benchmark timing recorded

### Phase 1 Verification

```bash
# 1. Run automated tests
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v

# 2. Verify Package.resolved versions changed correctly
cat tools/fluidaudio-bridge/Package.resolved | grep -E '"identity"|"version"'
# Expected: fluidaudio 0.4.x or higher (may resolve to 0.10.x due to SPM behavior)
```

- [ ] `pytest tests/test_fluidaudio_health.py -v` — all tests pass
- [ ] `pytest tests/test_fluidaudio_transcription.py -v` — all tests pass
- [ ] FluidAudio version in Package.resolved is 0.4.x or higher
- [ ] Swift build completes without errors
- [ ] No new compiler warnings (or document them)
- [ ] Bridge --help works
- [ ] **Code change applied:** `AsrModels.downloadAndLoad(version: .v2)` added
- [ ] JFK transcription matches baseline (±20% timing variance acceptable)

### Phase 2 Verification

```bash
# 1. Run automated tests
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v

# 2. Verify Package.resolved versions
cat tools/fluidaudio-bridge/Package.resolved | grep -E '"identity"|"version"'
# Expected: fluidaudio 0.8.x or higher
```

- [ ] All Phase 1 checks pass
- [ ] FluidAudio version in Package.resolved is 0.8.x or higher
- [ ] URL-based transcription works (if using new API)
- [ ] Longer audio (ted_60.wav) works correctly
- [ ] Progress reporting visible (if implemented)

### Phase 3 Verification

```bash
# 1. Run automated tests
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v

# 2. Verify Package.resolved versions
cat tools/fluidaudio-bridge/Package.resolved | grep -E '"identity"|"version"'
# Expected: fluidaudio 0.10.x
```

- [ ] All Phase 1-2 checks pass
- [ ] FluidAudio version in Package.resolved is 0.10.x
- [ ] No Swift 6 warnings (or Sendable compliance issues resolved)
- [ ] Fresh download completes successfully
- [ ] Model loading time acceptable
- [ ] Full benchmark suite passes

### Post-Upgrade Validation

```bash
# 1. Generate post-upgrade JSON for comparison
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav \
    --format json > docs/fluidaudio_post_upgrade_0.10.0.json

# 2. Compare JSON structure (keys should match)
diff <(jq 'keys' docs/fluidaudio_baseline_0.1.0.json) \
     <(jq 'keys' docs/fluidaudio_post_upgrade_0.10.0.json)

# 3. Run benchmark and compare timing
.venv/bin/mac-whisper-speedtest -b -m small -n 3 -i FluidAudioCoreMLImplementation
```

- [ ] JSON schema unchanged (same keys in output)
- [ ] Benchmark timing within ±20% of baseline (acceptable variance)
- [ ] Transcription text quality similar (may have minor differences with v2 model)
- [ ] Test with longer audio if available: `tests/ted_60.wav`
- [ ] Test timeout behavior (Python 300s timeout still an issue?)
- [ ] Document any behavior changes in commit message

---

## Summary

### Quick Reference: Upgrade Commands

```bash
# === PRE-UPGRADE BASELINE ===
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav \
    --format json > docs/fluidaudio_baseline_0.1.0.json

# === PHASE 1 ===
# Edit Package.swift: from: "0.4.1"
cd tools/fluidaudio-bridge && swift package update
# Edit main.swift: add version: .v2
swift build -c release
cd ../..
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v
# Verify: FluidAudio should be 0.4.x or higher
cat tools/fluidaudio-bridge/Package.resolved | grep fluidaudio -A2
# User commits after verification

# === PHASE 2 ===
# Edit Package.swift: from: "0.8.2"
cd tools/fluidaudio-bridge && swift package update && swift build -c release
cd ../..
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v
# Verify: FluidAudio should be 0.8.x or higher
cat tools/fluidaudio-bridge/Package.resolved | grep fluidaudio -A2
# User commits after verification

# === PHASE 3 ===
# Edit Package.swift: from: "0.10.0"
cd tools/fluidaudio-bridge && swift package update && swift build -c release
cd ../..
pytest tests/test_fluidaudio_health.py tests/test_fluidaudio_transcription.py -v
# Verify: FluidAudio should be 0.10.x
cat tools/fluidaudio-bridge/Package.resolved | grep fluidaudio -A2
# User commits after verification

# === POST-UPGRADE VALIDATION ===
./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge tests/jfk.wav \
    --format json > docs/fluidaudio_post_upgrade_0.10.0.json
diff <(jq 'keys' docs/fluidaudio_baseline_0.1.0.json) \
     <(jq 'keys' docs/fluidaudio_post_upgrade_0.10.0.json)
```

### Key Takeaways

1. **Run tests before AND after each phase** — Catches regressions automatically
2. **Don't jump 9 versions at once** — use natural breakpoints
3. **Explicit is better than implicit** — specify `.v2` model version
4. **SPM `from:` may skip phases** — Use `exact:` constraint to force specific versions
5. **Test at each phase** — catch issues early
6. **Commit atomically** — one logical change per commit
7. **Keep lock files** — reproducibility matters
8. **Timing variance of ±20% is acceptable** — Don't fail upgrade for minor timing differences
9. **Plan for rollback** — always have an escape route

### Comparison: FluidAudio vs WhisperKit Upgrades

| Aspect | FluidAudio | WhisperKit |
|--------|------------|------------|
| Version gap | 9 minor versions | 2 minor versions |
| Breaking changes | API enum (critical) | Struct→Class (non-issue for us) |
| Bridge code changes | **Required** (add `.v2`) | None |
| Risk level | HIGH | MEDIUM |
| Effort | Medium-High | Low |
| Recommended phases | 3 | 2 |
| JSON output | No segments | Has segments |

---

## Next Steps

1. [ ] Run baseline benchmark and save results
2. [ ] Create Phase 1 branch
3. [ ] Execute Phase 1 upgrade steps
4. [ ] Verify and commit
5. [ ] Repeat for Phases 2 and 3
6. [ ] Merge to main
7. [ ] Update `feature_plan_version_audit.md` status

---

## Sources

- [FluidAudio Releases](https://github.com/FluidInference/FluidAudio/releases)
- [FluidAudio API Documentation](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/API.md)
- [Swift Package Manager Documentation](https://www.swift.org/package-manager/)
