# Branch Protection Update Guide - Solution #2

This guide provides step-by-step instructions for repository administrators to update branch protection rules and resolve the pending check issue.

## Problem Summary

**Issue**: PR #32 shows "pending" status despite workflow completing successfully.

**Root Cause**: GitHub is tracking a stale check from an old workflow name that will never complete. When the workflow name changed from `Run nbdev_test on all projects` → `test (ship_model_lib)` → back to `Run nbdev_test on all projects`, the old check remained in pending state.

**Current State**:
- ✅ Workflow name: `Run nbdev_test on all projects` (correct)
- ✅ Latest run: Completed successfully 
- ❌ Status: Shows "pending" due to stale check

## Current Workflows

Based on the repository analysis, the following workflows are active:

1. **`Run nbdev_test on all projects`** - Main testing workflow (tests.yml)
2. **`release-please`** - Release automation workflow
3. **`Copilot coding agent`** - GitHub Copilot integration
4. **`Dependabot Updates`** - Dependency updates

## Solution: Update Branch Protection Rules

### Prerequisites
- Repository admin or owner access
- Access to repository Settings

### Step-by-Step Instructions

#### Step 1: Access Branch Protection Settings

1. Navigate to the repository: https://github.com/SINTEF/shipdesignlab
2. Click **Settings** (top navigation bar)
3. Click **Branches** (left sidebar under "Code and automation")
4. Find the branch protection rule for `main` branch
5. Click **Edit** button

#### Step 2: Identify Required Status Checks

In the "Require status checks to pass before merging" section, you'll see a list of required checks.

**Current configuration likely includes**:
- Old workflow names like:
  - `test (ship_model_lib)` ⚠️ (stale - needs removal)
  - `Tests and Linting` ⚠️ (stale - needs removal)
  - Other variations from workflow name changes

**Should be configured with**:
- `Run nbdev_test on all projects` ✅ (current correct name)
- Any other legitimately required checks

#### Step 3: Update Required Checks

1. **Search for the status check**: In the "Search for status checks" box, type the workflow name
2. **Remove stale checks**:
   - Uncheck any old workflow names like:
     - `test (ship_model_lib)`
     - `Tests and Linting`
     - Any other old names that appear
3. **Add current check** (if not already present):
   - Type: `Run nbdev_test on all projects`
   - Check the box to require this check
4. **Optional checks**: Decide if you want to require:
   - `release-please` (typically not required for PRs)
   - `Copilot coding agent` (typically not required)

#### Step 4: Save Changes

1. Scroll to the bottom of the page
2. Click **Save changes**

#### Step 5: Verify Fix

After updating the rules:

1. **Wait 30-60 seconds** for GitHub to propagate the changes
2. Navigate to PR #32: https://github.com/SINTEF/shipdesignlab/pull/32
3. **Refresh the page**
4. Check the status:
   - Should now show: "All checks have passed" ✅
   - Merge button should be enabled
   - No more "pending" status

### Troubleshooting

#### If PR Still Shows Pending:

**Option A: Push Empty Commit**
```bash
cd /path/to/shipdesignlab
git checkout clean_up_nbdev_github
git commit --allow-empty -m "chore: trigger status refresh"
git push
```

**Option B: Re-run Checks**
1. Go to the PR page
2. Click "Details" next to any check
3. Click "Re-run jobs"

**Option C: Contact GitHub Support**
If the above don't work, GitHub Support can manually clear stale checks.

### Alternative: Disable Branch Protection Temporarily

If you need to merge urgently and can't clear the stale check:

1. Go to Settings → Branches
2. Temporarily **disable** the branch protection rule
3. Merge the PR
4. **Re-enable** the branch protection rule with correct settings

⚠️ **Warning**: Only use this as a last resort as it temporarily reduces repository security.

## Prevention

To prevent this issue in the future:

1. **Don't rename workflows** that are used in branch protection
2. If you must rename a workflow:
   - Update branch protection rules FIRST
   - Then rename the workflow
   - Verify all PRs update correctly
3. Use workflow file names that match the workflow name for clarity

## Verification Script

A helper script is provided in `scripts/verify-workflows.sh` to check current workflow names and status.

## Additional Resources

- [GitHub Docs: Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)
- [GitHub Docs: Status Checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)

## Questions?

If you encounter issues following this guide, please:
1. Check that you have admin access to the repository
2. Verify the workflow names match exactly (case-sensitive)
3. Contact GitHub Support for help with persistent stale checks

---

**Last Updated**: 2026-02-12  
**Applies To**: PR #32 pending check issue
