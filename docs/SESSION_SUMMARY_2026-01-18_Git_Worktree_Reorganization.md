# Session Summary: Git Worktree Discovery and Folder Reorganization

**Date:** 2026-01-18
**Goal:** Reorganize project folders for cleaner feature branch workflow

---

## User's Original Idea (And Why It's Wrong)

The user initially proposed this folder structure:

```
mac-whisper-speedtest/
├── main/
├── feature/check-models/
├── feature/cli-batch-mode/
├── feature/model-cache-manager/
├── feature/templating-add-implementation/
├── feature/update-integration-versions/
├── experiments/
├── planning/
```

**Why this is a bad idea:**

| Problem | Why It Hurts |
|---------|--------------|
| **Disk space** | Each folder is a complete copy of the entire repo |
| **Sync nightmares** | Changes in one folder don't appear in others |
| **Merge conflicts** | Hard to merge when branches live in different folders |
| **No single source of truth** | Which folder is "real"? |
| **Defeats git's purpose** | You're manually doing what git automates |

**The correct approach:** One folder, use `git switch branch-name` to change context. Git rewrites the files in place.

---

## Key Discovery: You Were Already Using Git Worktrees

What appeared to be a "bad practice" of copying folders for different branches was actually an advanced git feature called **worktrees** - a legitimate (though advanced) way to have multiple branches checked out simultaneously.

### Folder State at Start of Session

```
mac-whisper-speedtest/          ← Main repo (.git/ lives here), on 'dev' branch
mac-whisper-speedtest_MAIN/     ← Worktree (lightweight), on 'main' branch
```

### Folder State After User's Rename Attempt (Mid-Session)

```
mac-whisper-speedtest_DEV/      ← Main repo (.git/ lives here), on 'dev' branch
mac-whisper-speedtest_MAIN/     ← Worktree (BROKEN - reference pointed to old path)
```

### How Git Worktrees Work

```
mac-whisper-speedtest_DEV/
├── .git/                      ← THE repository (all history, all branches)
│   ├── objects/               ← All commits, blobs, trees
│   ├── refs/                  ← All branch pointers
│   └── worktrees/
│       └── mac-whisper-speedtest_MAIN/   ← Metadata for the worktree
│
└── (code files on 'dev' branch)

mac-whisper-speedtest_MAIN/
├── .git                       ← TEXT FILE pointing back to DEV's .git
└── (code files on 'main' branch)
```

**Key insight:** All git history lives in ONE `.git/` folder. Both folders share that same history. The MAIN folder is essentially a "view" into a specific branch.

---

## Problem Encountered: Broken Worktree Reference

### What Happened

1. User renamed `mac-whisper-speedtest` → `mac-whisper-speedtest_DEV`
2. User renamed `mac-whisper-speedtest_MAIN` → `mac-whisper-speedtest`
3. VS Code was open during rename, causing confusion
4. The `.git` file in MAIN folder still pointed to the OLD path

### Symptom

```bash
cd .git/
# Error: cd: not a directory: .git/
```

The `.git` in MAIN is a FILE (not a folder) - this is normal for worktrees. But the path inside was broken.

### The Fix

Updated `/Users/rymalia/projects/mac-whisper-speedtest_MAIN/.git`:

**Before:**
```
gitdir: /Users/rymalia/projects/mac-whisper-speedtest/.git/worktrees/mac-whisper-speedtest_MAIN
```

**After:**
```
gitdir: /Users/rymalia/projects/mac-whisper-speedtest_DEV/.git/worktrees/mac-whisper-speedtest_MAIN
```

---

## Git Best Practices Learned

### Folder Copies vs. Git Branches

| Approach | Verdict |
|----------|---------|
| One folder per branch (manual copies) | Bad - defeats git's purpose |
| Single folder + `git switch` | Standard - recommended for most workflows |
| Git worktrees | Advanced - legitimate for simultaneous branch work |

### When to Use Worktrees

- Comparing branches side-by-side
- Running tests on one branch while editing another
- Long-running processes on one branch

### Standard Branch Workflow (Single Folder)

```bash
git switch main                    # Switch to main branch
git switch dev                     # Switch to dev branch
git switch -c feature/new-thing    # Create AND switch to new branch
git stash                          # Temporarily save uncommitted work
git stash pop                      # Restore stashed work
```

---

## Renaming Folders is Safe for Git

Git stores everything in `.git/` and tracks file contents, not folder names. You can safely rename a project folder without affecting git.

**Exception:** Worktrees have cross-references that break if you rename folders (as we discovered).

---

## Next Steps (Pending)

User wants to:
1. Have the main `.git/` repository live in a folder representing `main` branch (not `dev`)
2. Rename `dev` branch to `experiments`
3. **Review changes in `experiments` and decide which to merge into `main`**
4. Consolidate to a cleaner structure

**Planned approach:**
- Use `git worktree remove` to cleanly disconnect MAIN folder
- Rename DEV folder to become the primary project folder
- Switch to `main` branch
- Review diff between `experiments` and `main` to decide on merges
- Optionally create new worktree for experiments

**Note on feature branches:** The existing feature branches (`feature/check-models`, `feature/cli-batch-mode`, `feature/model-cache-manager`, etc.) are safe - they live in the git history and will remain accessible via `git switch feature/branch-name` after reorganization.

---

## Commands Reference

```bash
# Check worktree status
git worktree list

# Remove a worktree
git worktree remove /path/to/worktree

# Create a worktree
git worktree add /path/to/folder branch-name

# Rename a branch
git branch -m old-name new-name

# Switch branches
git switch branch-name

# Create and switch to new branch
git switch -c new-branch-name
```
