#!/bin/bash
# Run all quality checks

echo "🔍 Running quality checks..."

# Python
echo "📐 Python checks:"
black --check src/
isort --check-only src/
flake8 src/
mypy src/

# Frontend
echo "📐 Frontend checks:"
cd frontend
npm run lint
npm run typecheck
npm run test:ci
cd ..

# Security
echo "🔒 Security checks:"
bandit -r src/
safety check

echo "✅ All checks complete!"
