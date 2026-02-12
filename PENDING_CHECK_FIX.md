# Fix for Pending Check Issue

This PR addresses the build system compatibility issues that would cause workflow failures with modern pip (v24+).

## Changes Made

1. **Added `ship_model_lib/pyproject.toml`** - PEP 517 build configuration required by modern pip
2. **Updated `ship_model_lib/setup.py`** - Replaced deprecated `pkg_resources` with `packaging.version`  
3. **Updated `ship_model_lib/requirements.txt`** - Added `packaging>=21.0` dependency

## Testing

- ✅ Tests pass locally (`nbdev_test`)
- ✅ Formatting passes (`black --check`)
- ✅ Package installs with modern pip
- ✅ CodeQL security scan: 0 vulnerabilities

## Status

This PR demonstrates the fixes needed for the build system. The workflow requires manual approval (GitHub security policy for bot PRs).

## Note on PR #32

PR #32's pending check issue is caused by a stale status from the old workflow name. The workflow has been fixed and runs successfully, but GitHub's commit status API still shows the old pending check. This is a known GitHub behavior when workflow names change.

To resolve:
1. Push an empty commit to force status refresh, OR
2. Update branch protection rules to match current workflow name
