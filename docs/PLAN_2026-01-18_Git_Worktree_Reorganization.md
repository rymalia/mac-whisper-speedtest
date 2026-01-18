# Plan: Git Worktree Reorganization

**Date:** 2026-01-18

## Goal

Reorganize the git repository so that:
1. The main `.git/` repository lives in a folder that will be on the `main` branch
2. Rename the `dev` branch to `experiments`
3. Review changes in `experiments` and decide which to merge into `main`
4. End up with a clean, understandable folder structure

## Current State

```
mac-whisper-speedtest_DEV/      ← Main repo (.git/ lives here), on 'dev' branch
mac-whisper-speedtest_MAIN/     ← Worktree, on 'main' branch
```

**Existing branches (all will be preserved):**
- `main`
- `dev` (will be renamed to `experiments`)
- `feature/check-models`
- `feature/cli-batch-mode`
- `feature/model-cache-manager`
- `feature/templating-add-implementation`
- `feature/update-integration-versions`
- `planning`

## Target State

```
mac-whisper-speedtest/          ← Main repo (.git/ lives here), on 'main' branch
```

With the option to later create a worktree for `experiments` branch if needed.

**All feature branches remain accessible** via `git switch feature/branch-name`.

---

## Pre-Flight Checks (CRITICAL)

### Step 0.0: Close Your IDE

**IMPORTANT:** Close VS Code (or any IDE) that has either project folder open. We learned during this session that VS Code can interfere with folder renames and cause confusion.

You can reopen the project after Step 3.2 (folder rename) is complete.

Before making any changes, we MUST verify and preserve uncommitted work.

### Step 0.1: List All Worktrees

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git worktree list
```

**Expected output:** Shows both folders and their branches.

### Step 0.2: Check Uncommitted Changes in DEV Folder

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git status
```

**Action if changes exist:** Commit them or stash them before proceeding.

```bash
# Option A: Commit (if work is complete)
git add -A
git commit -m "WIP: saving work before reorganization"

# Option B: Stash (if work is incomplete)
git stash push -m "WIP before reorganization"
```

### Step 0.3: Check Uncommitted Changes in MAIN Folder

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_MAIN
git status
```

**Action if changes exist:** Commit them or stash them.

```bash
# Option A: Commit
git add -A
git commit -m "WIP: saving work before reorganization"

# Option B: Stash
git stash push -m "WIP before reorganization on main"
```

### Step 0.4: Verify Remote is Up to Date (Safety Backup)

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git push origin dev
git push origin main
```

This ensures your work is backed up to GitHub before any local changes.

---

## Phase 1: Rename the Branch

Before removing worktrees, rename `dev` to `experiments`.

### Step 1.1: Switch to the dev Branch (in DEV folder)

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git switch dev
```

### Step 1.2: Rename the Local Branch

```bash
git branch -m dev experiments
```

This renames the current branch from `dev` to `experiments`.

### Step 1.3: Update Remote Tracking (Optional)

If you want to rename the remote branch too:

```bash
# Push the new branch name to remote
git push origin experiments

# Delete the old remote branch (only if you're sure)
git push origin --delete dev

# Update tracking
git branch --set-upstream-to=origin/experiments experiments
```

**Note:** If others use this repo, coordinate before deleting remote branches.

---

## Phase 2: Remove the Worktree

### Step 2.1: Remove the Worktree Cleanly

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git worktree remove /Users/rymalia/projects/mac-whisper-speedtest_MAIN
```

**If there are uncommitted changes**, git will refuse. You must commit or stash first (see Step 0.3).

**If git refuses due to untracked files**, use force:

```bash
git worktree remove --force /Users/rymalia/projects/mac-whisper-speedtest_MAIN
```

### Step 2.2: Verify Worktree is Removed

```bash
git worktree list
```

**Expected output:** Only one entry (the DEV folder).

### Step 2.3: Delete the MAIN Folder (if not auto-deleted)

```bash
rm -rf /Users/rymalia/projects/mac-whisper-speedtest_MAIN
```

---

## Phase 3: Reorganize the Folder

### Step 3.1: Switch to Main Branch

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git switch main
```

### Step 3.2: Rename the Folder

```bash
mv /Users/rymalia/projects/mac-whisper-speedtest_DEV /Users/rymalia/projects/mac-whisper-speedtest
```

### Step 3.3: Verify Everything Works

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest
git status
git branch -a
git log --oneline -5
```

**Expected:**
- You're on `main` branch
- All branches visible (`main`, `experiments`, feature branches)
- History intact

---

## Phase 4: Review and Merge Changes from Experiments

Now that you're on `main` with a clean folder, decide which changes from `experiments` to incorporate.

### Step 4.1: See What's Different

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest

# See commit differences (what experiments has that main doesn't)
git log main..experiments --oneline

# See file differences
git diff main..experiments --stat

# See detailed diff for specific files
git diff main..experiments -- path/to/file.py
```

### Step 4.2: Choose Your Merge Strategy

**Option A: Merge everything from experiments**
```bash
git merge experiments -m "Merge experiments branch into main"
```

**Option B: Cherry-pick specific commits**
```bash
# List commits to choose from
git log experiments --oneline

# Pick specific commits by their hash
git cherry-pick abc1234
git cherry-pick def5678
```

**Option C: Merge but review interactively**
```bash
git merge experiments --no-commit --no-ff
# Review staged changes
git status
git diff --staged
# Then commit if happy, or reset if not
git commit -m "Merge selected experiments"
# OR: git reset --hard HEAD  (to abort)
```

**Option D: Keep branches separate for now**
Skip merging - you can always do it later. The `experiments` branch will remain available.

### Step 4.3: Handle Merge Conflicts (If Any)

If git reports conflicts:
```bash
# See which files conflict
git status

# Edit conflicted files (look for <<<<<<< markers)
# Then mark as resolved
git add path/to/resolved/file.py

# Complete the merge
git commit
```

---

## Phase 5: (Optional) Create Worktree for Experiments

If you want to keep two folders for simultaneous work on different branches:

```bash
cd /Users/rymalia/projects/mac-whisper-speedtest
git worktree add ../mac-whisper-speedtest_experiments experiments
```

This creates:
```
mac-whisper-speedtest/              ← main branch (primary repo)
mac-whisper-speedtest_experiments/  ← experiments branch (worktree)
```

**Note:** Most developers don't need worktrees. The standard workflow is:
```bash
git switch experiments    # Work on experiments
git switch main           # Switch back to main
git stash                 # Temporarily save uncommitted work when switching
```

---

## Recovery: If Something Goes Wrong

### Scenario: Lost uncommitted work

Check the stash:
```bash
git stash list
git stash pop  # Restore most recent stash
```

### Scenario: Accidentally deleted something

Your pushes to origin (Step 0.4) serve as backup:
```bash
git fetch origin
git reset --hard origin/main  # or origin/experiments
```

### Scenario: Worktree remove failed

Manually clean up:
```bash
# Remove worktree metadata
rm -rf /Users/rymalia/projects/mac-whisper-speedtest_DEV/.git/worktrees/mac-whisper-speedtest_MAIN

# Remove the folder
rm -rf /Users/rymalia/projects/mac-whisper-speedtest_MAIN
```

---

## Final Verification Checklist

After completing all steps, verify:

- [ ] `git status` shows clean working tree (or expected uncommitted files)
- [ ] `git branch -a` shows `main`, `experiments`, and all feature branches
- [ ] `git log --oneline -10` shows your commit history
- [ ] `git remote -v` shows origin pointing to GitHub
- [ ] Project runs: `uv sync && .venv/bin/mac-whisper-speedtest --help`
- [ ] You can switch branches: `git switch experiments && git switch main`
- [ ] VS Code opens the new folder location correctly

---

## Summary of Commands (Quick Reference)

```bash
# 0. Pre-flight (CLOSE VS CODE FIRST!)
cd /Users/rymalia/projects/mac-whisper-speedtest_DEV
git worktree list
git status
# (commit or stash any changes in both folders)
git push origin dev
git push origin main

# 1. Rename branch
git switch dev
git branch -m dev experiments

# 2. Remove worktree
git worktree remove /Users/rymalia/projects/mac-whisper-speedtest_MAIN
rm -rf /Users/rymalia/projects/mac-whisper-speedtest_MAIN  # if needed

# 3. Reorganize
git switch main
mv /Users/rymalia/projects/mac-whisper-speedtest_DEV /Users/rymalia/projects/mac-whisper-speedtest

# 4. Review & merge (choose one)
cd /Users/rymalia/projects/mac-whisper-speedtest
git log main..experiments --oneline          # See what's different
git diff main..experiments --stat            # See changed files
git merge experiments                        # Merge everything (Option A)
# OR: git cherry-pick <commit-hash>          # Pick specific commits (Option B)
# OR: skip for now                           # Keep separate (Option D)

# 5. Verify
git status
git branch -a
git log --oneline -10
```
