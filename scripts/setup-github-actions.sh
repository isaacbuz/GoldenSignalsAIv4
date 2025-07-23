#!/bin/bash

# Setup script for GitHub Actions
# This script helps configure your repository for CI/CD

echo "🚀 GitHub Actions Setup Script"
echo "=============================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed.${NC}"
    echo "Please install it first:"
    echo "  macOS: brew install gh"
    echo "  Linux: https://github.com/cli/cli#installation"
    exit 1
fi

echo -e "${GREEN}✓ GitHub CLI found${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Not in a git repository${NC}"
    exit 1
fi

# Get repository info
REPO_OWNER=$(git remote get-url origin | sed -E 's/.*[:/]([^/]+)\/[^/]+\.git/\1/')
REPO_NAME=$(git remote get-url origin | sed -E 's/.*\/([^/]+)\.git/\1/')

echo -e "${GREEN}✓ Repository: $REPO_OWNER/$REPO_NAME${NC}"

# Function to add a secret
add_secret() {
    local secret_name=$1
    local secret_value=$2
    local secret_desc=$3

    echo -e "\n${YELLOW}Adding secret: $secret_name${NC}"
    echo "Description: $secret_desc"

    if [ -z "$secret_value" ]; then
        echo -n "Enter value for $secret_name (press Enter to skip): "
        read -s secret_value
        echo
    fi

    if [ -n "$secret_value" ]; then
        echo "$secret_value" | gh secret set "$secret_name" --repo="$REPO_OWNER/$REPO_NAME"
        echo -e "${GREEN}✓ Secret $secret_name added${NC}"
    else
        echo -e "${YELLOW}⚠ Skipped $secret_name${NC}"
    fi
}

# Setup GitHub authentication if needed
echo -e "\n${YELLOW}Checking GitHub authentication...${NC}"
if ! gh auth status &> /dev/null; then
    echo "You need to authenticate with GitHub CLI first."
    echo "Run: gh auth login"
    exit 1
fi

echo -e "${GREEN}✓ Authenticated with GitHub${NC}"

# Add repository secrets
echo -e "\n${YELLOW}Setting up repository secrets...${NC}"

# API URLs
add_secret "STAGING_API_URL" "" "Staging API URL (e.g., https://api-staging.goldensignalsai.com)"
add_secret "STAGING_URL" "" "Staging Frontend URL (e.g., https://staging.goldensignalsai.com)"
add_secret "PRODUCTION_API_URL" "" "Production API URL (e.g., https://api.goldensignalsai.com)"
add_secret "PRODUCTION_URL" "" "Production Frontend URL (e.g., https://goldensignalsai.com)"

# API Keys (only if you need them for external services)
add_secret "OPENAI_API_KEY" "" "OpenAI API Key for GPT-4"
add_secret "ANTHROPIC_API_KEY" "" "Anthropic API Key for Claude"
add_secret "XAI_API_KEY" "" "xAI API Key for Grok"

# Enable GitHub Actions
echo -e "\n${YELLOW}Enabling GitHub Actions...${NC}"
gh api -X PUT "repos/$REPO_OWNER/$REPO_NAME/actions/permissions" \
  -f enabled=true \
  -f allowed_actions=all

echo -e "${GREEN}✓ GitHub Actions enabled${NC}"

# Set workflow permissions
echo -e "\n${YELLOW}Setting workflow permissions...${NC}"
gh api -X PUT "repos/$REPO_OWNER/$REPO_NAME/actions/permissions/workflow" \
  -f default_workflow_permissions=write \
  -f can_approve_pull_request_reviews=true

echo -e "${GREEN}✓ Workflow permissions configured${NC}"

# Create environments
echo -e "\n${YELLOW}Creating deployment environments...${NC}"

# Create staging environment
gh api -X PUT "repos/$REPO_OWNER/$REPO_NAME/environments/staging" \
  -F "wait_timer=0" \
  -F "reviewers=[]" \
  -F "deployment_branch_policy=null" 2>/dev/null || true

echo -e "${GREEN}✓ Staging environment created${NC}"

# Create production environment with protection rules
gh api -X PUT "repos/$REPO_OWNER/$REPO_NAME/environments/production" \
  -F "wait_timer=0" \
  -F "reviewers=[]" \
  -F "deployment_branch_policy={\"protected_branches\":true,\"custom_branch_policies\":false}" 2>/dev/null || true

echo -e "${GREEN}✓ Production environment created${NC}"

# Install pre-commit hooks
echo -e "\n${YELLOW}Installing pre-commit hooks...${NC}"
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}⚠ pre-commit not installed. Run: pip install pre-commit${NC}"
fi

# Summary
echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
echo -e "\nNext steps:"
echo -e "1. Push your code to trigger the workflows:"
echo -e "   ${YELLOW}git push origin main${NC}"
echo -e "\n2. Check workflow runs at:"
echo -e "   ${YELLOW}https://github.com/$REPO_OWNER/$REPO_NAME/actions${NC}"
echo -e "\n3. Add status badges to your README.md:"
echo -e "   ${YELLOW}![Main CI/CD](https://github.com/$REPO_OWNER/$REPO_NAME/workflows/Main%20CI%2FCD%20Pipeline/badge.svg)${NC}"
echo -e "\n4. Fix any failing checks:"
echo -e "   ${YELLOW}cd frontend && npm run lint -- --fix${NC}"
echo -e "   ${YELLOW}black src/ && isort src/${NC}"

echo -e "\n${GREEN}Happy coding! 🚀${NC}"
