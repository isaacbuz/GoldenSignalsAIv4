#!/bin/bash

# Daily Development Report for GoldenSignalsAI
# Provides automated progress tracking and metrics

echo "📊 GoldenSignalsAI Daily Development Report"
echo "==========================================="
echo "📅 Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Git activity summary
echo "🚀 Features Completed Today:"
git log --since="1 day ago" --oneline --grep="feat:" --grep="feature:" --grep="add:" | head -10
if [ $? -ne 0 ] || [ -z "$(git log --since='1 day ago' --oneline --grep='feat:' --grep='feature:' --grep='add:')" ]; then
    echo "   No features completed today"
fi
echo ""

echo "🐛 Bugs Fixed Today:"
git log --since="1 day ago" --oneline --grep="fix:" --grep="bug:" | head -10
if [ $? -ne 0 ] || [ -z "$(git log --since='1 day ago' --oneline --grep='fix:' --grep='bug:')" ]; then
    echo "   No bugs fixed today"
fi
echo ""

echo "🔧 Refactoring & Improvements:"
git log --since="1 day ago" --oneline --grep="refactor:" --grep="improve:" --grep="update:" | head -5
if [ $? -ne 0 ] || [ -z "$(git log --since='1 day ago' --oneline --grep='refactor:' --grep='improve:' --grep='update:')" ]; then
    echo "   No refactoring done today"
fi
echo ""

# Code statistics
echo "📈 Code Changes:"
if git diff --stat HEAD~1 >/dev/null 2>&1; then
    STATS=$(git diff --stat HEAD~1 | tail -1)
    echo "   $STATS"

    FILES_CHANGED=$(git diff --name-only HEAD~1 | wc -l | xargs)
    echo "   Files modified: $FILES_CHANGED"

    # Component analysis
    COMPONENTS_MODIFIED=$(git diff --name-only HEAD~1 | grep -E "\.(tsx|ts)$" | grep -v test | wc -l | xargs)
    echo "   Components modified: $COMPONENTS_MODIFIED"

    TESTS_MODIFIED=$(git diff --name-only HEAD~1 | grep -E "\.test\.(tsx|ts)$" | wc -l | xargs)
    echo "   Tests modified: $TESTS_MODIFIED"
else
    echo "   No changes from yesterday"
fi
echo ""

# Test status
echo "🧪 Test Status:"
if command -v npm >/dev/null 2>&1; then
    if [ -f "package.json" ] && grep -q "test" package.json; then
        echo "   Running test suite..."
        if npm run test --silent >/dev/null 2>&1; then
            echo "   ✅ All tests passing"
        else
            echo "   ❌ Some tests failing - run 'npm test' for details"
        fi
    else
        echo "   ⚠️  No test script found in package.json"
    fi
else
    echo "   ⚠️  npm not available"
fi
echo ""

# Component inventory
echo "📦 Component Inventory:"
if [ -d "src/components" ]; then
    TOTAL_COMPONENTS=$(find src/components -name "*.tsx" -not -name "*.test.tsx" | wc -l | xargs)
    echo "   Total components: $TOTAL_COMPONENTS"

    UNIFIED_COMPONENTS=$(find src/components -name "Unified*.tsx" | wc -l | xargs)
    echo "   Unified components: $UNIFIED_COMPONENTS"

    TEST_FILES=$(find src/components -name "*.test.tsx" | wc -l | xargs)
    echo "   Test files: $TEST_FILES"



else
    echo "   ⚠️  src/components directory not found"
fi
echo ""

# Redux store status
echo "🏪 Redux Store Status:"
if [ -d "src/store" ]; then
    SLICES=$(find src/store -name "*Slice.ts" | wc -l | xargs)
    echo "   Redux slices: $SLICES"

    SELECTORS=$(find src/store -name "selectors.ts" | wc -l | xargs)
    echo "   Selector files: $SELECTORS"

    HOOKS=$(find src/store -name "hooks.ts" | wc -l | xargs)
    echo "   Custom hooks: $HOOKS"
else
    echo "   ⚠️  src/store directory not found"
fi
echo ""

# Bundle analysis (if available)
echo "📦 Bundle Status:"
if [ -d "dist" ]; then
    BUNDLE_SIZE=$(du -sh dist 2>/dev/null | cut -f1)
    echo "   Bundle size: $BUNDLE_SIZE"

    JS_FILES=$(find dist -name "*.js" | wc -l | xargs)
    echo "   JavaScript files: $JS_FILES"
else
    echo "   📝 Run 'npm run build' to analyze bundle size"
fi
echo ""

# Development environment
echo "🛠️  Development Environment:"
echo "   Node version: $(node --version 2>/dev/null || echo 'Not available')"
echo "   npm version: $(npm --version 2>/dev/null || echo 'Not available')"

if [ -f "package.json" ]; then
    DEPENDENCIES=$(grep -c '".*":' package.json 2>/dev/null || echo "0")
    echo "   Total dependencies: $DEPENDENCIES"
fi
echo ""

# Recent activity summary
echo "📊 Activity Summary (Last 7 Days):"
COMMITS_WEEK=$(git log --since="7 days ago" --oneline | wc -l | xargs)
echo "   Commits this week: $COMMITS_WEEK"

CONTRIBUTORS=$(git log --since="7 days ago" --format="%an" | sort | uniq | wc -l | xargs)
echo "   Active contributors: $CONTRIBUTORS"
echo ""

# Recommendations
echo "💡 Recommendations:"
if [ "$TESTS_MODIFIED" -lt "$COMPONENTS_MODIFIED" ] && [ "$COMPONENTS_MODIFIED" -gt 0 ]; then
    echo "   ⚠️  Consider adding more tests for modified components"
fi

if [ "$UNIFIED_COMPONENTS" -lt 4 ]; then
    echo "   📝 Expected 4 unified components (Dashboard, AIChat, SearchBar, Chart)"
fi

if [ "$COMMITS_WEEK" -gt 20 ]; then
    echo "   🚀 High development velocity this week!"
elif [ "$COMMITS_WEEK" -lt 5 ]; then
    echo "   📈 Consider increasing development activity"
fi
echo ""

# Context check reminder
echo "🚨 Remember:"
echo "   📖 Read frontend/DEVELOPMENT_CONTEXT.md before making changes"
echo "   📋 Check frontend/COMPONENT_INVENTORY.md to avoid duplicates"
echo "   🎯 Use existing unified components instead of creating new ones"
echo "   🤖 Use prompt templates from frontend/PROMPT_TEMPLATES.md"
echo ""

echo "Report generated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
