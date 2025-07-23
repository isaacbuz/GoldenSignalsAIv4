#!/bin/bash

echo "🧹 Cleaning Vite caches and dependencies..."

# Kill any running dev server on port 3000
echo "Stopping any processes on port 3000..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Remove all cache directories
echo "Removing cache directories..."
rm -rf node_modules/.vite
rm -rf .vite
rm -rf dist
rm -rf .parcel-cache
rm -rf .cache

# Clear npm cache
echo "Clearing npm cache..."
npm cache clean --force

# Optional: Full reinstall (uncomment if needed)
# echo "Removing node_modules..."
# rm -rf node_modules
# echo "Reinstalling dependencies..."
# npm install

echo "✅ Cleanup complete!"
echo ""
echo "To start the dev server, run:"
echo "npm run dev -- --force"
echo ""
echo "Or for a full fresh install:"
echo "rm -rf node_modules && npm install && npm run dev"
