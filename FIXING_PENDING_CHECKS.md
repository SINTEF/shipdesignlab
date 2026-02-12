# Fixing Pending Check Issues

## Quick Start

If you're seeing a **pending check** that never completes on a PR, follow these steps:

### For Repository Admins

**Option 1: Update Branch Protection Rules (Recommended)**
1. Run the verification script:
   ```bash
   ./scripts/verify-workflows.sh
   ```
2. Follow the detailed guide: [`docs/BRANCH_PROTECTION_UPDATE_GUIDE.md`](docs/BRANCH_PROTECTION_UPDATE_GUIDE.md)

**Option 2: Push Empty Commit (Quick Fix)**
```bash
git checkout <branch-name>
git commit --allow-empty -m "chore: trigger status refresh"
git push
```

### For Contributors

If your PR shows pending checks:
1. Notify a repository admin about the issue
2. Share this document: `docs/BRANCH_PROTECTION_UPDATE_GUIDE.md`
3. The admin can update branch protection rules to clear stale checks

## Root Cause

This issue occurs when:
1. A workflow name changes in `.github/workflows/*.yml`
2. Branch protection rules still reference the old name
3. GitHub tracks both old (pending forever) and new (passes) checks
4. PR remains blocked despite new checks passing

## Files in This Fix

- **`docs/BRANCH_PROTECTION_UPDATE_GUIDE.md`** - Complete admin guide with screenshots
- **`scripts/verify-workflows.sh`** - Helper script to identify current workflow names
- **`PENDING_CHECK_FIX.md`** - Technical explanation and context

## Need Help?

See the full guide: [`docs/BRANCH_PROTECTION_UPDATE_GUIDE.md`](docs/BRANCH_PROTECTION_UPDATE_GUIDE.md)
