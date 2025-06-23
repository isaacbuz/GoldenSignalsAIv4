# GoldenSignalsAI CI/CD Implementation Guide

## Overview

This guide documents the comprehensive CI/CD pipeline implementation for GoldenSignalsAI V2, providing automated testing, security scanning, and deployment workflows.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Development   │────▶│   CI Pipeline   │────▶│   CD Pipeline   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
    Code Push               Test & Build              Deploy
    PR Creation             Security Scan             Staging
                           Quality Check              Production
```

## CI Pipeline (`.github/workflows/ci.yml`)

### Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

### Jobs

#### 1. Backend Tests
- **Environment**: Ubuntu latest with Redis service
- **Steps**:
  - Python setup with dependency caching
  - Linting (Black, Flake8, mypy)
  - Unit tests with coverage
  - Integration tests
  - Coverage reporting to Codecov

#### 2. Frontend Tests
- **Environment**: Ubuntu latest with Node.js
- **Steps**:
  - Node.js setup with npm caching
  - ESLint and TypeScript checking
  - Unit tests with Jest
  - Production build verification

#### 3. Security Scanning
- **Tools**:
  - Trivy for vulnerability scanning
  - Safety for Python dependencies
  - npm audit for frontend dependencies
- **Actions**: Fail on critical/high vulnerabilities

#### 4. Code Quality
- **Tools**:
  - SonarCloud integration
  - Coverage threshold enforcement (60%)
- **Dependencies**: Requires backend and frontend tests

#### 5. Performance Tests
- **Tools**:
  - pytest-benchmark for API benchmarks
  - Locust for load testing
- **Metrics**:
  - Response time benchmarks
  - Load test with 100 concurrent users

#### 6. Docker Build
- **Images**:
  - Backend: `goldensignals-backend`
  - Frontend: `goldensignals-frontend`
- **Features**:
  - Multi-platform builds
  - Layer caching
  - Push to Docker Hub

## CD Pipeline (`.github/workflows/cd.yml`)

### Triggers
- Successful CI pipeline completion
- Manual workflow dispatch

### Deployment Flow

```
CI Success → Staging → E2E Tests → Production (Canary) → Full Production
                ↓                        ↓
           Smoke Tests              Monitor Metrics
                                         ↓
                                    Rollback if Failed
```

### Jobs

#### 1. Deploy to Staging
- **Infrastructure**: AWS EKS
- **Tools**: kubectl, Helm
- **Steps**:
  - Deploy with Helm chart
  - Run smoke tests
  - Slack notifications

#### 2. E2E Tests
- **Tool**: Cypress
- **Target**: Staging environment
- **Coverage**: Critical user flows

#### 3. Deploy to Production
- **Strategy**: Canary deployment (10% traffic)
- **Steps**:
  1. Database backup
  2. Deploy canary version
  3. Monitor metrics for 15 minutes
  4. Promote to full production if healthy
  5. Create GitHub release
  6. Invalidate CDN cache

#### 4. Rollback
- **Trigger**: Automatic on deployment failure
- **Action**: Revert to previous Helm release

## Security Scanning (`.github/workflows/security.yml`)

### Schedule
- On push/PR
- Daily at 2 AM UTC

### Security Checks

#### 1. CodeQL Analysis
- Languages: Python, JavaScript
- Queries: Security and quality

#### 2. Dependency Scanning
- Trivy for filesystem scanning
- Safety for Python packages
- npm audit for Node packages

#### 3. Container Scanning
- Scan Docker images with Trivy
- Report vulnerabilities to GitHub Security

#### 4. Secret Scanning
- TruffleHog for commit history
- GitLeaks for current codebase

#### 5. License Compliance
- Check for approved licenses
- Fail on GPL/proprietary licenses

#### 6. OWASP Check
- Comprehensive dependency analysis
- Check for known vulnerabilities

## Supporting Scripts

### 1. Smoke Tests (`scripts/smoke-tests.sh`)
- Tests all critical API endpoints
- WebSocket connectivity check
- Response time validation
- Retry logic for reliability

### 2. Canary Monitor (`scripts/monitor-canary.sh`)
- Monitors error rate (< 5%)
- Tracks P95 latency (< 1000ms)
- CPU and memory usage
- Automatic failure detection

### 3. Load Testing (`tests/performance/locustfile.py`)
- Three user personas:
  - Regular users
  - Power users
  - Mobile users
- Realistic usage patterns
- Performance goal validation

## Configuration Requirements

### GitHub Secrets

```yaml
# Docker Hub
DOCKER_USERNAME: Your Docker Hub username
DOCKER_PASSWORD: Docker Hub access token

# AWS
AWS_ACCESS_KEY_ID: AWS access key
AWS_SECRET_ACCESS_KEY: AWS secret key
CLOUDFRONT_DISTRIBUTION_ID: CloudFront ID

# Monitoring
SLACK_WEBHOOK: Slack webhook URL
SENTRY_DSN: Sentry project DSN

# Code Quality
SONAR_TOKEN: SonarCloud token
CODECOV_TOKEN: Codecov token

# Testing
CYPRESS_RECORD_KEY: Cypress dashboard key
```

### Environments

1. **Staging**
   - URL: `https://staging.goldensignals.ai`
   - Auto-deploy from `main`
   - Used for integration testing

2. **Production**
   - URL: `https://goldensignals.ai`
   - Manual approval required
   - Canary deployment strategy

## Best Practices

### 1. Branch Protection
```yaml
main:
  - Require PR reviews (2)
  - Require status checks
  - Require up-to-date branches
  - Include administrators
```

### 2. Commit Standards
- Use conventional commits
- Sign commits with GPG
- Reference issue numbers

### 3. Testing Requirements
- Minimum 60% code coverage
- All tests must pass
- No critical security issues

### 4. Deployment Safety
- Always backup before deploy
- Use canary deployments
- Monitor for 15 minutes
- Automatic rollback on failure

## Monitoring & Alerts

### Metrics to Track
1. **Build Success Rate**
   - Target: > 95%
   - Alert on 3 consecutive failures

2. **Deployment Frequency**
   - Target: Daily to staging
   - Weekly to production

3. **Lead Time**
   - Target: < 1 hour from commit to staging
   - < 4 hours to production

4. **MTTR (Mean Time to Recovery)**
   - Target: < 30 minutes
   - Automatic rollback helps achieve this

### Alert Channels
- Slack: Build status, deployment notifications
- Email: Security vulnerabilities
- PagerDuty: Production incidents

## Troubleshooting

### Common Issues

1. **CI Failures**
   ```bash
   # Check specific job logs
   gh run view <run-id> --log
   
   # Re-run failed jobs
   gh run rerun <run-id> --failed
   ```

2. **Docker Build Issues**
   ```bash
   # Clear builder cache
   docker builder prune -a
   
   # Check Dockerfile syntax
   docker build --no-cache .
   ```

3. **Deployment Failures**
   ```bash
   # Check Helm release
   helm list -n production
   
   # View deployment logs
   kubectl logs -n production -l app=goldensignals
   ```

## Future Enhancements

1. **Progressive Delivery**
   - Feature flags integration
   - A/B testing support
   - Gradual rollout controls

2. **Advanced Security**
   - Runtime security scanning
   - Compliance automation
   - Pen testing integration

3. **Performance**
   - Synthetic monitoring
   - Real user monitoring
   - SLO/SLA tracking

4. **GitOps**
   - ArgoCD integration
   - Declarative deployments
   - Automated rollbacks

## Maintenance

### Weekly Tasks
- Review security scan results
- Update dependencies
- Check build times

### Monthly Tasks
- Review and optimize workflows
- Update documentation
- Audit secrets and permissions

### Quarterly Tasks
- Disaster recovery testing
- Performance baseline updates
- Tool version updates

## Resources

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [OWASP DevSecOps](https://owasp.org/www-project-devsecops-guideline/) 