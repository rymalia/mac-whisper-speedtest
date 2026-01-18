# Session Summary: Hidden Dependencies Documentation

**Date:** 2026-01-14
**Focus:** Documenting truly hidden native dependencies in version audit

---

## Summary

Identified and documented a critical gap in the version audit: **bundled native code** that is compiled INTO Python packages and invisible to pip/uv tooling. This is distinct from transitive pip dependencies (which ARE visible in `uv.lock`).

---

## Key Discovery

### The Problem

When reviewing pywhispercpp releases, we noticed updates to the whisper.cpp submodule that weren't captured in our version audit. Investigation revealed:

```
pywhispercpp v1.4.1 (pip package)
      │
      └── Contains: libwhisper.1.8.2.dylib
                    ↑
                    whisper.cpp v1.8.2 (C++ library)
                    - NOT a pip package
                    - NOT in uv.lock
                    - Version only in dylib filename
```

### Truly Hidden vs. Visible Dependencies

| Type | Examples | Visibility |
|------|----------|------------|
| **Truly Hidden** | whisper.cpp in pywhispercpp | NOT in pip/uv, only in `.dylib` files |
| **Visible Transitive** | ctranslate2, torch, mlx | IN `uv.lock`, `pip list` |

**Key insight:** ctranslate2 is NOT hidden—it's a pip package that appears in `uv.lock`. Only bundled native code like whisper.cpp qualifies as "truly hidden."

### Version Mapping Discovered

| pywhispercpp | whisper.cpp |
|--------------|-------------|
| v1.3.0 | v1.7.0 |
| v1.4.0 | v1.8.2 |

Upgrading pywhispercpp 1.3→1.4 implicitly upgrades whisper.cpp 1.7→1.8, which may have breaking changes not mentioned in pywhispercpp's changelog.

---

## Changes Made

### File: `docs/feature_plan_version_audit.md`

**Change 1: Hidden Native Code Warning** (Executive Summary)
- Added callout box after "Related Documents" section
- Includes table with pywhispercpp → whisper.cpp mapping
- Instructions for checking native version via dylib inspection

**Change 2: WhisperCppCoreMLImplementation Section Update**
- Added `whisper.cpp (bundled)` row to dependency table
- Added version mapping note with pywhispercpp → whisper.cpp versions
- Link to whisper.cpp releases for breaking change review

**Change 3: Hidden Dependency References** (Sources)
- Added new subsection with links to:
  - whisper.cpp releases
  - pywhispercpp releases

---

## How to Verify Hidden Dependencies

```bash
# Check whisper.cpp version bundled in pywhispercpp
ls .venv/lib/python3.12/site-packages/pywhispercpp/.dylibs/
# Output: libwhisper.1.8.2.dylib
```

The version number (1.8.2) is encoded in the dylib filename.

---

## Related Documents

- `docs/feature_plan_version_audit.md` — Updated with hidden dependency documentation
- `/Users/rymalia/.claude/plans/tingly-prancing-stroustrup.md` — Implementation plan for this work

---

## References

- [whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases)
- [pywhispercpp Releases](https://github.com/absadiki/pywhispercpp/releases)
