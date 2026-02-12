# How to Apply the Fix to PR #32

This guide explains how to fix the pending check issue in PR #32.

## Quick Fix (One-line change)

### Option 1: Via GitHub Web Interface

1. Go to PR #32: https://github.com/SINTEF/shipdesignlab/pull/32
2. Navigate to the Files changed tab
3. Find the file `.github/workflows/tests.yml`
4. Click the "..." menu and select "Edit file"
5. Change line 1 from:
   ```yaml
   name: Run tests and formatting checks
   ```
   to:
   ```yaml
   name: Run nbdev_test on all projects
   ```
6. Commit the change directly to the `clean_up_nbdev_github` branch

### Option 2: Via Command Line

If you have the PR branch checked out locally:

```bash
cd /path/to/shipdesignlab
git checkout clean_up_nbdev_github
git pull origin clean_up_nbdev_github

# Apply the patch
git apply fix-pr32-workflow-name.patch

# Or edit manually
sed -i '1s/Run tests and formatting checks/Run nbdev_test on all projects/' .github/workflows/tests.yml

# Commit and push
git add .github/workflows/tests.yml
git commit -m "Fix pending check by restoring workflow name"
git push origin clean_up_nbdev_github
```

### Option 3: Using the Patch File

A patch file is provided in this repository: `fix-pr32-workflow-name.patch`

```bash
cd /path/to/shipdesignlab
git checkout clean_up_nbdev_github
git apply fix-pr32-workflow-name.patch
git add .github/workflows/tests.yml
git commit -m "Fix pending check by restoring workflow name"
git push origin clean_up_nbdev_github
```

## What This Fix Does

- **Restores the workflow name** to match the required status check in branch protection
- **Keeps all other changes** from PR #32 (pytest, updated actions versions, etc.)
- **Allows the PR to become mergeable** once the workflow runs successfully

## Verification

After applying the fix:

1. A new workflow run will be triggered on PR #32
2. The workflow will run with the name "Run nbdev_test on all projects"
3. GitHub will recognize this as the required status check
4. The PR status should change from "pending" to showing the actual check results
5. If the workflow passes, the PR will become mergeable

## Why This Works

GitHub Actions workflows are identified by their name for status checks. When PR #32 changed the workflow name, it created a new workflow check, but GitHub branch protection was still waiting for the old workflow name. By restoring the original name, we ensure continuity of the status check without requiring changes to branch protection rules.

The workflow name is just a display name and doesn't affect what the workflow actually does. The workflow will still run pytest and all the updated logic from PR #32.
