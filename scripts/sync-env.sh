#!/bin/bash
# sync-env.sh: Syncs root .env to frontend and backend .env files for GoldenSignalsAI
# Usage: bash scripts/sync-env.sh

set -e

ROOT_ENV="$(dirname "$0")/../.env"
FRONTEND_ENV="$(dirname "$0")/../presentation/frontend/.env"
BACKEND_ENV="$(dirname "$0")/../backend/.env"

if [ ! -f "$ROOT_ENV" ]; then
  echo "Root .env file not found at $ROOT_ENV"
  exit 1
fi

echo "Syncing frontend environment variables..."
grep '^VITE_' "$ROOT_ENV" > "$FRONTEND_ENV"
echo "Frontend .env updated at $FRONTEND_ENV"

# Uncomment and customize the following if you want to sync backend-specific variables
# echo "Syncing backend environment variables..."
# grep -v '^VITE_' "$ROOT_ENV" > "$BACKEND_ENV"
# echo "Backend .env updated at $BACKEND_ENV"

echo "Sync complete."
