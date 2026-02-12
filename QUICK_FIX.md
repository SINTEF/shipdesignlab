# Quick Fix Reference for PR #32

## TL;DR
PR #32 is stuck because the workflow name changed. Change it back to unblock the PR.

## The Fix (Choose One Method)

### Method 1: GitHub Web (Easiest)
1. Go to: https://github.com/SINTEF/shipdesignlab/pull/32/files
2. Find: `.github/workflows/tests.yml`
3. Click: "..." → "Edit file"
4. Line 1: Change `Run tests and formatting checks` to `Run nbdev_test on all projects`
5. Commit to `clean_up_nbdev_github` branch

### Method 2: Command Line
```bash
git checkout clean_up_nbdev_github
git pull
sed -i '1s/Run tests and formatting checks/Run nbdev_test on all projects/' .github/workflows/tests.yml
git add .github/workflows/tests.yml
git commit -m "Fix: Restore workflow name for status check compatibility"
git push
```

### Method 3: Apply Patch
```bash
git checkout clean_up_nbdev_github
git pull
git apply fix-pr32-workflow-name.patch
git add .github/workflows/tests.yml
git commit -m "Fix: Restore workflow name for status check compatibility"
git push
```

## What This Does
- ✅ Unblocks PR #32
- ✅ Keeps all the good changes (pytest, updated actions, etc.)
- ✅ No admin access needed
- ✅ No branch protection changes needed

## Result
After pushing, the workflow will run with the correct name and PR #32 becomes mergeable.

## Need More Info?
- **Technical details**: See `PR32_PENDING_CHECK_FIX.md`
- **Step-by-step**: See `HOW_TO_FIX_PR32.md`
- **Executive summary**: See `SUMMARY.md`
