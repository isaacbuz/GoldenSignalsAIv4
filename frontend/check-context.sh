#!/bin/bash
# Quick context check script
echo "🔍 CHECKING EXISTING COMPONENTS..."
echo ""
echo "📁 Unified Components:"
ls -la src/components/*/Unified*.tsx 2>/dev/null || echo "No unified components found"
echo ""
echo "📁 Pages:"
ls -la src/pages/*/*.tsx 2>/dev/null | head -10
echo ""
echo "📁 Redux Store:"
ls -la src/store/ 2>/dev/null
echo ""
echo "📊 Component Count:"
find src/components -name "*.tsx" | wc -l | xargs echo "Total TSX files:"
echo ""
echo "🚨 Remember: READ CONTEXT FILES BEFORE CODING!"
echo "   - frontend/DEVELOPMENT_CONTEXT.md"
echo "   - frontend/COMPONENT_INVENTORY.md"
