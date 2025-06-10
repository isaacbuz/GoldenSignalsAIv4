#!/bin/bash

# GoldenSignalsAI V3 Frontend Restart Script
echo "🔄 Restarting Frontend Development Server..."
echo "========================================"

# Kill existing processes
echo "🔪 Stopping any running frontend processes..."
pkill -f "vite"
pkill -f "npm run dev"
pkill -f "node.*3000"

# Clear Vite cache
echo "🧹 Clearing Vite cache..."
cd frontend
rm -rf node_modules/.vite
rm -rf dist
rm -rf .vite

# Clear npm cache
echo "🧹 Clearing npm cache..."
npm cache clean --force

# Reinstall dependencies if needed
echo "📦 Checking dependencies..."
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start development server
echo "🚀 Starting frontend development server..."
npm run dev 