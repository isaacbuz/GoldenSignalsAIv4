# GitHub Actions Workflows

This directory contains all GitHub Actions workflows for the GoldenSignalsAIv4 project.

## 🚀 Quick Start

1. **Authenticate GitHub CLI**: Run `./scripts/github-auth-setup.sh`
2. **Install Pre-commit Hooks**: Run `./scripts/setup-quality-checks.sh`
3. **Fix Current Issues**: Run linters with auto-fix before pushing
4. **Push Code**: Workflows will run automatically

## Workflows Overview

### Core Workflows

#### 1. **main.yml** - Main CI/CD Pipeline
- **Triggers**: Push to main/develop, Pull requests to main
- **Purpose**: Primary continuous integration pipeline
- **Jobs**:
  - Frontend lint and type checking
  - Frontend build
  - Backend lint and format checking
  - Backend tests with PostgreSQL and Redis
  - Security vulnerability scanning
  - Status notification

#### 2. **quality-checks.yml** - Comprehensive Quality Checks
- **Triggers**: Push to main/develop, Pull requests
- **Purpose**: Ensures code quality across frontend and backend
- **Features**:
  - Multi-version testing (Node 18.x/20.x, Python 3.9/3.10/3.11)
  - Code formatting validation
  - Security scanning with Trivy
  - Docker build testing

#### 3. **pre-commit.yml** - Pre-commit Hook Validation
- **Triggers**: Pull requests
- **Purpose**: Runs pre-commit hooks to catch issues early
- **Includes**: Black, isort, flake8, ESLint, security checks

### Deployment Workflows

#### 4. **deploy-staging.yml** - Staging Deployment
- **Triggers**: Push to develop, Manual trigger
- **Purpose**: Build and deploy to staging environment
- **Features**:
  - Docker image building and pushing to GitHub Container Registry
  - Environment-specific configuration
  - Deployment notifications

### Monitoring Workflows

#### 5. **health-check.yml** - Scheduled Health Checks
- **Triggers**: Every 6 hours, Manual trigger
- **Purpose**: Monitor application health and dependencies
- **Checks**:
  - API endpoint health
  - Frontend performance (Lighthouse)
  - Dependency updates

## Getting Started

### 1. Set up Secrets

Add these secrets to your GitHub repository:

```
STAGING_API_URL     # Your staging API URL
STAGING_URL         # Your staging frontend URL
PRODUCTION_API_URL  # Your production API URL (for future use)
PRODUCTION_URL      # Your production frontend URL (for future use)
```

### 2. Enable GitHub Actions

1. Go to Settings → Actions → General
2. Under "Actions permissions", select "Allow all actions and reusable workflows"
3. Under "Workflow permissions", select "Read and write permissions"

### 3. Install Pre-commit Locally

```bash
pip install pre-commit
pre-commit install
```

### 4. Fix Current Issues

Before the workflows can pass, you'll need to:

1. Fix ESLint errors:
   ```bash
   cd frontend
   npm run lint -- --fix
   ```

2. Format Python code:
   ```bash
   black src/
   isort src/
   ```

3. Run tests locally:
   ```bash
   # Frontend
   cd frontend && npm test

   # Backend
   pytest tests/
   ```

## Workflow Status Badges

Add these badges to your main README.md:

```markdown
![Main CI/CD](https://github.com/YOUR_USERNAME/GoldenSignalsAIv4/workflows/Main%20CI%2FCD%20Pipeline/badge.svg)
![Quality Checks](https://github.com/YOUR_USERNAME/GoldenSignalsAIv4/workflows/Quality%20Checks/badge.svg)
![Health Check](https://github.com/YOUR_USERNAME/GoldenSignalsAIv4/workflows/Health%20Check/badge.svg)
```

## Customization

### Adjusting Linting Rules

- **Python**: Edit `.flake8` for flake8 rules
- **JavaScript/TypeScript**: Edit `frontend/eslint.config.js`
- **Pre-commit**: Edit `.pre-commit-config.yaml`

### Adding New Workflows

1. Create a new `.yml` file in this directory
2. Use existing workflows as templates
3. Test locally using [act](https://github.com/nektos/act)

### Disabling Checks

If you need to temporarily disable certain checks:

1. Add `continue-on-error: true` to specific steps
2. Or comment out jobs in the workflow files
3. Or disable workflows in Settings → Actions

## Troubleshooting

### Common Issues

1. **"Resource not accessible by integration"**
   - Check repository permissions in Settings → Actions

2. **Docker build failures**
   - Ensure Dockerfile exists in the correct location
   - Check for syntax errors in Dockerfile

3. **Test failures**
   - Run tests locally first
   - Check environment variables and secrets

4. **Lint errors blocking deployment**
   - Fix with auto-fix commands when possible
   - Update rules if they're too strict

### Debugging Workflows

1. Enable debug logging:
   - Add secret `ACTIONS_RUNNER_DEBUG` with value `true`
   - Add secret `ACTIONS_STEP_DEBUG` with value `true`

2. Use workflow_dispatch for manual testing:
   ```yaml
   on:
     workflow_dispatch:
   ```

3. Check workflow runs in the Actions tab for detailed logs

## Best Practices

1. **Keep workflows fast**: Use caching for dependencies
2. **Fail fast**: Run quick checks (lint) before slow ones (tests)
3. **Use matrix builds**: Test multiple versions in parallel
4. **Monitor costs**: GitHub Actions has usage limits
5. **Security**: Never commit secrets, use GitHub Secrets

## Future Enhancements

Consider adding:
- [ ] Semantic versioning and auto-release
- [ ] Performance benchmarking
- [ ] E2E testing with Cypress/Playwright
- [ ] Database migration checks
- [ ] Load testing for APIs
- [ ] Automated dependency updates with Dependabot
