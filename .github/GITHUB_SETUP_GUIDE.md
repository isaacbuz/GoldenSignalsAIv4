# GitHub Setup Guide

## Overview
This guide helps you set up GitHub Actions and Claude integration for your project.

## Authentication Options

### Option 1: GitHub CLI (Recommended)
```bash
# 1. Install GitHub CLI
brew install gh  # macOS
# or
sudo apt install gh  # Ubuntu/Debian

# 2. Run our auth setup script
./scripts/github-auth-setup.sh

# 3. Follow the prompts to authenticate
```

### Option 2: Using Your Test Token
Since you mentioned you have a test token, you can use it directly:

```bash
# Set up with your token
echo "YOUR_GITHUB_TOKEN" | gh auth login --with-token

# Verify it worked
gh auth status
```

### Option 3: Manual Setup (No CLI)
If you prefer not to use GitHub CLI, you can set everything up manually through the GitHub web interface:

1. **Go to your repository on GitHub**
2. **Navigate to Settings → Secrets and variables → Actions**
3. **Add these repository secrets:**
   - Click "New repository secret" for each:
   ```
   STAGING_API_URL = http://your-staging-api.com
   STAGING_URL = http://your-staging-frontend.com
   OPENAI_API_KEY = your-openai-key
   ANTHROPIC_API_KEY = your-anthropic-key
   XAI_API_KEY = your-xai-key
   ```

## Claude Integration

### For Claude + GitHub Integration:
1. **Claude App**: Already installed ✓
2. **No additional token needed** - Claude uses GitHub's built-in authentication
3. **Just use Claude commands** in issues and PRs:
   - `@claude help`
   - `@claude review`
   - `@claude fix-lint`

### GitHub Actions:
1. **No token needed** - GitHub provides `GITHUB_TOKEN` automatically
2. **Workflows use** `${{ secrets.GITHUB_TOKEN }}` which is automatic
3. **External services** need their own tokens (OpenAI, etc.)

## Common Issues & Solutions

### "Bad credentials" Error
```bash
# Solution 1: Re-authenticate
gh auth logout
gh auth login

# Solution 2: Use token directly
export GH_TOKEN="your-token-here"
gh auth status
```

### "Resource not accessible by integration"
This means the GitHub Actions doesn't have proper permissions:
1. Go to Settings → Actions → General
2. Under "Workflow permissions", select "Read and write permissions"
3. Check "Allow GitHub Actions to create and approve pull requests"

### "gh: command not found"
Install GitHub CLI first:
```bash
# macOS
brew install gh

# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

## Quick Setup Commands

### After Authentication:
```bash
# 1. Add a secret (example)
gh secret set OPENAI_API_KEY

# 2. List your secrets
gh secret list

# 3. Run a workflow manually
gh workflow run main.yml

# 4. View workflow runs
gh run list

# 5. Create an issue with Claude
gh issue create --title "[FIX] ESLint errors" --body "@claude fix-lint"
```

## Testing Your Setup

### 1. Test GitHub CLI:
```bash
gh auth status
gh repo view
```

### 2. Test Claude:
Create a test issue:
```bash
gh issue create --title "Test Claude Integration" --body "
@claude help me understand this codebase
"
```

### 3. Test GitHub Actions:
```bash
# Push your changes
git add .
git commit -m "feat: Add GitHub Actions"
git push

# Watch the workflow
gh run watch
```

## What Claude Can Access

Claude for GitHub can:
- ✅ Read your code
- ✅ Comment on issues and PRs
- ✅ Suggest code changes
- ✅ Analyze problems
- ❌ Cannot access your secrets
- ❌ Cannot make direct commits (only suggestions)

## No Token Needed For:
1. **GitHub Actions** - Uses built-in `GITHUB_TOKEN`
2. **Claude Integration** - Uses GitHub App authentication
3. **Basic gh commands** - After you authenticate once

## Tokens Only Needed For:
1. **External APIs** (OpenAI, Anthropic, market data)
2. **Custom integrations**
3. **CI/CD to external services**

## Next Steps
1. Run: `./scripts/github-auth-setup.sh`
2. If it works, run: `./scripts/setup-github-actions.sh`
3. Create an issue for ESLint fixes: `@claude fix-lint`
4. Push your code to trigger workflows
