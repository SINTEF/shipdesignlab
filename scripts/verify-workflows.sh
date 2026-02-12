#!/bin/bash
#
# Workflow Verification Script
# 
# This script verifies the current workflow names in the repository
# and helps identify which names should be used in branch protection rules.
#
# Usage: ./scripts/verify-workflows.sh
#

set -e

echo "=================================="
echo "Workflow Verification Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "📋 Checking workflow files in .github/workflows/..."
echo ""

# Find all workflow files
WORKFLOW_FILES=$(find .github/workflows -name "*.yml" -o -name "*.yaml" 2>/dev/null || true)

if [ -z "$WORKFLOW_FILES" ]; then
    echo -e "${RED}❌ No workflow files found!${NC}"
    exit 1
fi

echo "Found workflow files:"
for file in $WORKFLOW_FILES; do
    echo "  - $file"
done
echo ""

echo "🔍 Extracting workflow names..."
echo ""

declare -a WORKFLOW_NAMES

for file in $WORKFLOW_FILES; do
    # Extract the name field from the YAML file
    NAME=$(grep -E "^name:" "$file" | head -1 | sed 's/^name:[[:space:]]*//' | sed 's/^["'\'']//' | sed 's/["'\'']$//' | xargs)
    
    if [ -n "$NAME" ]; then
        WORKFLOW_NAMES+=("$NAME")
        echo -e "${GREEN}✅ Workflow: '$NAME'${NC}"
        echo "   File: $file"
        echo ""
    else
        echo -e "${YELLOW}⚠️  No name found in: $file${NC}"
        echo ""
    fi
done

echo "=================================="
echo "📊 Summary"
echo "=================================="
echo ""
echo "Current workflow names that should be configured in branch protection:"
echo ""

for name in "${WORKFLOW_NAMES[@]}"; do
    echo -e "  ${GREEN}✓${NC} $name"
done

echo ""
echo "=================================="
echo "🔧 Next Steps"
echo "=================================="
echo ""
echo "1. Go to: Settings → Branches → Edit branch protection rule for 'main'"
echo "2. In 'Require status checks to pass before merging':"
echo "3. Remove any old/stale workflow names NOT listed above"
echo "4. Ensure the following are required (if needed):"
echo ""

for name in "${WORKFLOW_NAMES[@]}"; do
    # Only suggest the test workflow as required by default
    if [[ "$name" == *"test"* ]] || [[ "$name" == *"Test"* ]]; then
        echo -e "   ${GREEN}[RECOMMENDED]${NC} $name"
    else
        echo -e "   ${YELLOW}[OPTIONAL]${NC} $name"
    fi
done

echo ""
echo "5. Save changes"
echo ""

echo "=================================="
echo "⚠️  Common Stale Check Names to Remove"
echo "=================================="
echo ""
echo "If you see any of these in branch protection, remove them:"
echo "  - test (ship_model_lib)"
echo "  - Tests and Linting"
echo "  - Any other names not listed in the summary above"
echo ""

echo "=================================="
echo "✅ Verification Complete"
echo "=================================="
echo ""
echo "For detailed instructions, see: docs/BRANCH_PROTECTION_UPDATE_GUIDE.md"
echo ""
