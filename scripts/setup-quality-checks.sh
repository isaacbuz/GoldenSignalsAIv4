#!/bin/bash

# Setup Quality Checks for GoldenSignalsAI
# This script sets up all the quality check tools and pre-commit hooks

set -e

echo "🔧 Setting up quality checks for GoldenSignalsAI..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the project root
if [ ! -f "package.json" ] || [ ! -f "pyproject.toml" ]; then
    print_error "This script must be run from the project root directory"
    exit 1
fi

# Install Python quality tools
echo "📦 Installing Python quality tools..."
pip install -q black isort flake8 mypy pytest pytest-cov bandit safety pre-commit
print_status "Python quality tools installed"

# Install frontend quality tools
echo "📦 Installing frontend quality tools..."
cd frontend
npm install --save-dev \
    @typescript-eslint/parser \
    @typescript-eslint/eslint-plugin \
    eslint-plugin-react \
    eslint-plugin-react-hooks \
    prettier \
    eslint-config-prettier \
    eslint-plugin-prettier
cd ..
print_status "Frontend quality tools installed"

# Initialize pre-commit
echo "🎣 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg
print_status "Pre-commit hooks installed"

# Run initial checks
echo "🔍 Running initial quality checks..."

# Python checks
echo "  Python:"
black --check src/ 2>/dev/null && echo "    ✓ Black formatting" || echo "    ✗ Black formatting (run: black src/)"
isort --check-only src/ 2>/dev/null && echo "    ✓ Import sorting" || echo "    ✗ Import sorting (run: isort src/)"
flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics > /dev/null 2>&1 && echo "    ✓ Flake8 linting" || echo "    ✗ Flake8 linting"

# Frontend checks
echo "  Frontend:"
cd frontend
npm run lint 2>/dev/null && echo "    ✓ ESLint" || echo "    ✗ ESLint (run: npm run lint)"
npm run typecheck 2>/dev/null && echo "    ✓ TypeScript" || echo "    ✗ TypeScript (run: npm run typecheck)"
cd ..

# Create git hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create a commit message template
cat > .gitmessage << 'EOF'
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type should be one of: feat, fix, docs, style, refactor, perf, test, chore, revert, security
# Scope is optional and can be anything specifying the place of the commit change
# Subject is a short description of the change (imperative, present tense)
# Body should include motivation for the change and contrast with previous behavior
# Footer should contain any breaking changes or issues closed
#
# Example:
# feat(api): add new endpoint for agent analysis
#
# Added /api/v1/agents/analyze endpoint that triggers comprehensive
# multi-agent analysis for a given symbol. This endpoint orchestrates
# all 9 specialized agents and returns consensus recommendations.
#
# Closes #123
EOF

# Configure git to use the template
git config commit.template .gitmessage

print_status "Git commit template configured"

# Create quality check script
cat > run-quality-checks.sh << 'EOF'
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
EOF

chmod +x run-quality-checks.sh
print_status "Quality check script created: ./run-quality-checks.sh"

# Summary
echo ""
echo "✨ Quality checks setup complete!"
echo ""
echo "📋 Available commands:"
echo "  • Run all checks: ./run-quality-checks.sh"
echo "  • Run pre-commit: pre-commit run --all-files"
echo "  • Python formatting: black src/"
echo "  • Python import sorting: isort src/"
echo "  • Frontend linting: cd frontend && npm run lint"
echo "  • Frontend type checking: cd frontend && npm run typecheck"
echo ""
echo "🎯 Pre-commit hooks will run automatically on git commit"
echo "📝 Use the commit template for consistent commit messages"
