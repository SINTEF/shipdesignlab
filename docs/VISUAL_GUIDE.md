# Visual Guide: Branch Protection Update Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: Pending Check Issue                    │
│                                                                   │
│  Symptom: PR shows "pending" despite workflow passing            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Run Verification │
                    │     Script        │
                    └─────────┬─────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │  scripts/verify-workflows.sh           │
         │                                        │
         │  Output:                               │
         │  ✓ Run nbdev_test on all projects     │
         │  ✓ release-please                      │
         └──────────────┬─────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │ Access Repository Settings      │
        │                                 │
        │ Settings → Branches             │
        │    → Edit rule for "main"       │
        └────────────┬───────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────────┐
   │ Review Required Status Checks               │
   │                                             │
   │ Currently Required:                         │
   │ ☑ Run nbdev_test on all projects          │
   │ ☑ test (ship_model_lib)          ← STALE! │
   │ ☑ Tests and Linting               ← STALE! │
   └─────────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │ Update Required Checks          │
        │                                 │
        │ Remove:                         │
        │   ☐ test (ship_model_lib)      │
        │   ☐ Tests and Linting           │
        │                                 │
        │ Keep:                           │
        │   ☑ Run nbdev_test on...       │
        └────────────┬───────────────────┘
                     │
                     ▼
             ┌──────────────┐
             │ Save Changes  │
             └──────┬───────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Wait 30-60 seconds   │
         │ for GitHub to sync   │
         └──────┬───────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │ Check PR Status                │
    │                                │
    │ Should now show:               │
    │ ✅ All checks have passed      │
    └──────┬────────────────────────┘
           │
           ▼
    ┌─────────────┐      ┌──────────────────────┐
    │   SUCCESS   │──No──│ Still Pending?       │
    │             │      │                      │
    │ PR can be   │      │ Try:                 │
    │ merged!     │      │ 1. Push empty commit │
    └─────────────┘      │ 2. Re-run checks     │
                         │ 3. Contact support   │
                         └──────────────────────┘
```

## Key Decision Points

### Should a check be required?

```
                    ┌─────────────────┐
                    │ Is it a test    │
                    │ workflow?       │
                    └────┬────────────┘
                         │
                    Yes  │  No
                    ┌────▼────┐
                    │         │
                    ▼         ▼
            ┌──────────┐  ┌───────────┐
            │ REQUIRE  │  │ OPTIONAL  │
            │ IT       │  │           │
            └──────────┘  └───────────┘
            
Examples:
✅ REQUIRE: "Run nbdev_test on all projects"
❌ OPTIONAL: "release-please" (runs on main, not PRs)
❌ OPTIONAL: "Copilot coding agent" (bot workflow)
❌ OPTIONAL: "Dependabot Updates" (automatic)
```

## Timeline

```
Time        Event
────────────────────────────────────────────────────────
T=0         Workflow name changed
            Old check: "test (ship_model_lib)"
            New check: "Run nbdev_test on all projects"

T+1min      New workflow runs and passes ✅
            Old check: still pending ⏳

T+5min      PR shows "pending" despite new check passing
            Branch protection still requires old name

T+NOW       Admin updates branch protection rules
            Removes old check name

T+1min      GitHub syncs changes
            Status updates to "passed" ✅

T+2min      PR can be merged 🎉
```

## Before vs After

### BEFORE (Incorrect Configuration)

```
Branch Protection for "main":
├─ Require status checks to pass: ✓
│  ├─ Run nbdev_test on all projects ✅ (passing)
│  ├─ test (ship_model_lib) ⏳ (pending forever - STALE!)
│  └─ Tests and Linting ⏳ (pending forever - STALE!)
│
└─ Result: PR BLOCKED ❌
```

### AFTER (Correct Configuration)

```
Branch Protection for "main":
├─ Require status checks to pass: ✓
│  └─ Run nbdev_test on all projects ✅ (passing)
│
└─ Result: PR CAN MERGE ✅
```

## Quick Reference

| Action | Command/Location |
|--------|-----------------|
| Verify current workflows | `./scripts/verify-workflows.sh` |
| Access branch protection | Settings → Branches → Edit rule |
| Force status refresh | `git commit --allow-empty && git push` |
| View workflow runs | Actions tab → Select workflow |
| See detailed guide | `docs/BRANCH_PROTECTION_UPDATE_GUIDE.md` |

---

**Pro Tip**: Bookmark this page for quick reference when dealing with pending check issues!
