# Session Summary: Git Folder Cleanup

**Date:** 2026-01-24

---

## Goal

Consolidate two git worktree folders into a single, cleanly-named folder on the `main` branch.

---

## Starting State

```
mac-whisper-speedtest_DEV/   ← Main repo (.git/ lives here), on 'dev' branch
mac-whisper-speedtest_MAIN/  ← Worktree, on 'main' branch
```

## Ending State

```
mac-whisper-speedtest/       ← Main repo (.git/ lives here), on 'main' branch
```

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep remote `dev` branch as-is | User wants to potentially revisit that code later; renaming to `experiments` deferred |
| Discard local `dev` files | User confirmed no local dev code was needed; remote backup sufficient |
| Preserve uncommitted files from `_MAIN` | 10 files (2 modified, 8 untracked) needed to be kept |
| Manual folder rename | User performed the `mv` command to avoid shell working directory issues |

---

## Steps Performed

1. **Backed up uncommitted files** to `/tmp/mac-whisper-backup/`
   - CLAUDE.md, test_benchmark.py (modified)
   - .vscode/, feat_model-handling-issues.md, git-log-all.txt, git-log.txt, scratch_next-deep-dive.md, tests/ted_60.m4a, uv-backup.lock, uv-pip-list.txt (untracked)

2. **User manually copied files** from `_MAIN` to `_DEV` (preserving `.git/` in `_DEV`)

3. **Worktree cleanup** — The `_MAIN` worktree reference was automatically cleaned up when the `.git` file was removed

4. **Switched `_DEV` to `main` branch** using `git switch main --force`

5. **Restored uncommitted files** from backup (the branch switch overwrote modified files)

6. **User renamed folder** `_DEV` → `mac-whisper-speedtest`

7. **User deleted orphan `_MAIN` folder** and temp backup

---

## Files Modified

| File | Change |
|------|--------|
| (none committed) | All changes were git operations, no file edits |

---

## Verification Performed

- Confirmed on `main` branch
- Confirmed all 10 uncommitted files preserved with correct status
- Confirmed all branches available: `main`, `dev`, 5 feature branches, `planning`
- Confirmed remote tracking intact: `origin/main`, `origin/dev`

---

## Pending / Future Work

- [ ] **Rename `dev` branch to `experiments`** (local and remote) — deferred by user choice
- [ ] Review which changes from `dev`/`experiments` should be merged into `main`
- [ ] Clean up old feature branches if no longer needed

---

## Key Insights Shared

1. **Git worktrees** are lightweight "windows" into different branches — the `.git/` folder contains all history for all branches
2. **Local vs remote branches** are independent — you can delete/rename local branches without affecting remote until you push
3. **Folder names don't matter to git** — only what's inside `.git/` matters; renaming folders is safe (except for worktree cross-references)

---

## Commands Reference

```bash
# Switch branches (replaces working files)
git switch main
git switch dev

# Force switch (discards uncommitted changes)
git switch main --force

# List worktrees
git worktree list

# Remove worktree
git worktree remove /path/to/worktree

# Prune stale worktree references
git worktree prune
```
