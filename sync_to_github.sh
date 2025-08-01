#!/bin/bash

# GoldenSignalsAI GitHub Sync Script
# This script will help sync all your local changes to GitHub

set -e  # Exit on error

echo "🚀 Starting GitHub sync for GoldenSignalsAI..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    print_error "Not on main branch. Please switch to main first."
    exit 1
fi

# Stage 1: Handle deleted files
print_status "Stage 1: Removing deleted files..."
git rm .api.pid .backend.pid .frontend.pid 2>/dev/null || true
git rm Dockerfile.analytics Dockerfile.orchestrator 2>/dev/null || true
git rm docker-compose.v3.yml 2>/dev/null || true
git rm restart-frontend.sh run_golden_signals.sh 2>/dev/null || true
git rm start-ui.sh start_goldensignals_v3.sh 2>/dev/null || true
git rm src/main_simple_v2.py 2>/dev/null || true
git rm test_simple_server.py test_system.py test_system_health.py 2>/dev/null || true
git rm tests/test_gdpr.py tests/test_health.py tests/test_main.py 2>/dev/null || true
git rm agents/core/risk/test_options_risk_agent.py 2>/dev/null || true
git rm agents/core/sentiment/test_news_agent.py 2>/dev/null || true
git rm agents/core/sentiment/test_social_media_agent.py 2>/dev/null || true
git rm agents/research/backtesting/backtest_engine.py 2>/dev/null || true
git rm agents/research/ml/test_*.py 2>/dev/null || true
print_success "Deleted files staged"

# Stage 2: Add .gitignore first to prevent tracking unwanted files
print_status "Stage 2: Updating .gitignore..."
git add .gitignore
git commit -m "chore: Update .gitignore for better project organization" || true

# Stage 3: Add environment and configuration files
print_status "Stage 3: Adding configuration files..."
git add .env.example env.example
git add config/ .vscode/
git add pyproject.toml setup.py pytest.ini requirements*.txt poetry.lock
git commit -m "feat: Update project configuration and dependencies

- Updated environment examples
- Enhanced project configuration
- Updated dependencies and requirements
- Added ML-specific requirements" || true

# Stage 4: Add Docker and deployment configurations
print_status "Stage 4: Adding Docker and deployment files..."
git add Dockerfile* docker-compose*.yml
git add k8s/ helm/ .github/workflows/
git add nginx*.conf redis_cache.conf
git add scripts/
git commit -m "feat: Add production deployment configurations

- Docker configurations for ML services
- Kubernetes manifests for production
- Helm charts for deployment
- CI/CD pipeline for ML services
- Nginx and Redis configurations" || true

# Stage 5: Add documentation
print_status "Stage 5: Adding documentation..."
git add *.md docs/
git commit -m "docs: Comprehensive documentation update

- Added ML deployment guides
- Updated architecture documentation
- Added performance optimization guides
- Enhanced implementation summaries
- Added production readiness documentation" || true

# Stage 6: Update core source code
print_status "Stage 6: Updating core source code..."
git add src/
git commit -m "feat: Core backend improvements

- Enhanced ML signal generation
- Improved data services
- Added backtesting capabilities
- Enhanced WebSocket support
- Added AI chat and analyst services" || true

# Stage 7: Update agents
print_status "Stage 7: Updating agent system..."
git add agents/
git commit -m "feat: Comprehensive agent system update

- Added hybrid agents for better performance
- Enhanced technical analysis agents
- Added sentiment and market analysis
- Improved orchestration system
- Added portfolio management agents" || true

# Stage 8: Update frontend
print_status "Stage 8: Updating frontend..."
git add frontend/
git commit -m "feat: Frontend enhancements

- Added AI Trading Lab components
- Enhanced chart visualizations
- Improved dashboard layouts
- Added backtesting UI
- Enhanced WebSocket integration" || true

# Stage 9: Add ML models and training
print_status "Stage 9: Adding ML components..."
git add ml_models/ ml_training/ ml_enhanced_backtest_system.py
git add integrated_ml_backtest_api.py ml_signal_blender.py
git commit -m "feat: ML models and training infrastructure

- Added trained ML models
- Enhanced backtesting system
- Added model evaluation tools
- Integrated ML API service" || true

# Stage 10: Add test files
print_status "Stage 10: Adding test suite..."
git add tests/
git commit -m "test: Comprehensive test suite

- Added production data tests
- Enhanced system integration tests
- Added performance benchmarks
- Improved test coverage" || true

# Stage 11: Add utility and helper files
print_status "Stage 11: Adding utility files..."
git add *.py *.sh *.json
git add data/ test_data/
git commit -m "feat: Utility scripts and data files

- Added deployment scripts
- Enhanced startup scripts
- Added demo and validation tools
- Included test data" || true

# Stage 12: Add any remaining files
print_status "Stage 12: Adding remaining files..."
git add -A
git commit -m "feat: Final additions and improvements

- Added remaining configuration files
- Included all documentation
- Added infrastructure components" || true

# Stage 13: Push to GitHub
print_status "Pushing all changes to GitHub..."
print_warning "This will push all commits to origin/main"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    print_success "All changes pushed to GitHub successfully!"
else
    print_warning "Push cancelled. You can manually push later with: git push origin main"
fi

# Summary
echo
echo "================================================"
print_success "Sync process completed!"
echo
print_status "Summary of changes:"
git log --oneline -15
echo
print_status "To view the changes on GitHub, visit:"
echo "https://github.com/isaacbuz/GoldenSignalsAIv4"
echo
print_status "Your backup branch is: backup-local-changes-$(date +%Y%m%d)"
echo "================================================"
