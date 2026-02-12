# Screenshot Guide: Updating Branch Protection Rules

This guide provides a visual reference of what you'll see when updating branch protection rules.

## Navigation Path

```
GitHub Repository (SINTEF/shipdesignlab)
    ↓
Settings (top navigation bar)
    ↓
Branches (left sidebar under "Code and automation")
    ↓
Branch protection rules
    ↓
main (click "Edit" button)
    ↓
Require status checks to pass before merging section
```

## What You'll See

### 1. Branch Protection Rules Page

```
┌─────────────────────────────────────────────────────────────┐
│ Branch protection rules                                      │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Branch name pattern: main                      [Edit] │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ⚡ Protected branches restrict who can push commits        │
│    and require status checks                                │
└─────────────────────────────────────────────────────────────┘
```

### 2. Edit Rule Page - Status Checks Section

```
┌───────────────────────────────────────────────────────────────┐
│ Require status checks to pass before merging         ☑       │
│                                                               │
│ Choose which status checks must pass before branches can be  │
│ merged into a branch that matches this rule.                 │
│                                                               │
│ ☑ Require branches to be up to date before merging          │
│                                                               │
│ Status checks found in the last week for this repository     │
│ ┌─────────────────────────────────────────────────────┐     │
│ │ Search for status checks in the last week...        │     │
│ └─────────────────────────────────────────────────────┘     │
│                                                               │
│ ☑ Run nbdev_test on all projects                    ✅      │
│ ☑ test (ship_model_lib)                             ⚠️ OLD   │
│ ☑ Tests and Linting                                  ⚠️ OLD   │
│ ☐ release-please                                             │
│ ☐ Copilot coding agent                                       │
└───────────────────────────────────────────────────────────────┘
```

### 3. What to Change

**BEFORE (Incorrect - causes pending):**
```
☑ Run nbdev_test on all projects    ← Keep this ✅
☑ test (ship_model_lib)              ← UNCHECK THIS ❌
☑ Tests and Linting                  ← UNCHECK THIS ❌
```

**AFTER (Correct - allows merge):**
```
☑ Run nbdev_test on all projects    ← Only this should be checked ✅
☐ test (ship_model_lib)              ← Unchecked
☐ Tests and Linting                  ← Unchecked
```

### 4. Save Changes Button

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│                    [ Save changes ]                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Finding Status Checks

If you can't find a check in the list:

1. **Use the search box**: Type the workflow name
2. **Check recently**: GitHub only shows checks from the last week
3. **Trigger a new run**: Push a commit to see the check appear

### Search Box Example

```
┌─────────────────────────────────────────────────────────┐
│ Status checks found in the last week for this repository│
│ ┌───────────────────────────────────────────────────┐   │
│ │ Run nbdev_test                    [Search icon]   │   │
│ └───────────────────────────────────────────────────┘   │
│                                                          │
│ Results:                                                 │
│ ☐ Run nbdev_test on all projects                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Common Issues and Solutions

### Issue 1: Check Not in List

**Problem**: The workflow name you want doesn't appear in the list.

**Solution**: 
- Run `./scripts/verify-workflows.sh` to confirm the exact name
- Push a commit to trigger the workflow
- Wait a few minutes and refresh the page

### Issue 2: Multiple Similar Names

**Problem**: You see multiple variations like:
- `Run nbdev_test on all projects`
- `test (ship_model_lib)`
- `Tests and Linting`

**Solution**: 
- Only keep the one that matches your current workflow file
- Run `./scripts/verify-workflows.sh` to verify
- Uncheck all others

### Issue 3: Can't Find Edit Button

**Problem**: You don't see an "Edit" button.

**Solution**: 
- You need admin or owner access
- Contact a repository owner to grant you access
- Or ask them to make the change following this guide

## Verification After Changes

### In the PR (after saving and waiting 60 seconds):

**BEFORE:**
```
┌──────────────────────────────────────────────────────┐
│ Some checks haven't completed yet                    │
│                                                       │
│ ⏳ test (ship_model_lib)           Pending           │
│ ✅ Run nbdev_test on all projects  Passed            │
│                                                       │
│ Merge button is: DISABLED ❌                          │
└──────────────────────────────────────────────────────┘
```

**AFTER:**
```
┌──────────────────────────────────────────────────────┐
│ All checks have passed                                │
│                                                       │
│ ✅ Run nbdev_test on all projects  Passed            │
│                                                       │
│ Merge button is: ENABLED ✅                           │
└──────────────────────────────────────────────────────┘
```

## Quick Checklist

Use this checklist when making changes:

- [ ] Navigate to Settings → Branches
- [ ] Click "Edit" on the main branch protection rule
- [ ] Scroll to "Require status checks to pass before merging"
- [ ] Run `./scripts/verify-workflows.sh` to see current names
- [ ] Uncheck any old/stale workflow names
- [ ] Ensure only current workflow names are checked
- [ ] Click "Save changes" at the bottom
- [ ] Wait 30-60 seconds for GitHub to sync
- [ ] Check the PR - should show "All checks have passed"
- [ ] Merge button should be enabled

## Need Help?

If you're still having issues:

1. Check you have admin access to the repository
2. Verify the workflow name matches exactly (case-sensitive)
3. Try the empty commit solution: `git commit --allow-empty && git push`
4. Contact GitHub Support for persistent issues

---

**For more details**: See `docs/BRANCH_PROTECTION_UPDATE_GUIDE.md`
