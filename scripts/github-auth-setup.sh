#!/bin/bash

# GitHub Authentication Setup Script
echo "🔧 GitHub Authentication Setup"
echo "=============================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI not installed${NC}"
    echo ""
    echo "Please install GitHub CLI first:"
    echo -e "${BLUE}macOS:${NC} brew install gh"
    echo -e "${BLUE}Ubuntu/Debian:${NC} sudo apt install gh"
    echo -e "${BLUE}Windows:${NC} winget install --id GitHub.cli"
    echo ""
    echo "Or download from: https://cli.github.com/"
    exit 1
fi

echo -e "${GREEN}✓ GitHub CLI found${NC}"

# Check current auth status
echo -e "\n${YELLOW}Checking authentication status...${NC}"
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✓ Already authenticated with GitHub${NC}"
    gh auth status
    echo ""
    read -p "Do you want to re-authenticate? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing authentication."
    else
        gh auth logout
        gh auth login
    fi
else
    echo -e "${YELLOW}Not authenticated. Starting login process...${NC}"
    echo ""
    echo "You have several authentication options:"
    echo ""
    echo -e "${BLUE}Option 1: Interactive Login (Recommended)${NC}"
    echo "  - Opens browser for authentication"
    echo "  - Most secure method"
    echo "  - Works with 2FA"
    echo ""
    echo -e "${BLUE}Option 2: Personal Access Token${NC}"
    echo "  - Use existing token"
    echo "  - Good for CI/CD environments"
    echo ""

    read -p "Choose option (1 or 2): " option

    case $option in
        1)
            echo -e "\n${YELLOW}Starting interactive login...${NC}"
            gh auth login
            ;;
        2)
            echo -e "\n${YELLOW}Personal Access Token Setup${NC}"
            echo "Create a token at: https://github.com/settings/tokens/new"
            echo ""
            echo "Required scopes:"
            echo "  - repo (Full control of private repositories)"
            echo "  - workflow (Update GitHub Action workflows)"
            echo "  - admin:org (optional, for org settings)"
            echo ""
            read -s -p "Enter your Personal Access Token: " token
            echo
            if [ -n "$token" ]; then
                echo "$token" | gh auth login --with-token
            else
                echo -e "${RED}No token provided${NC}"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            exit 1
            ;;
    esac
fi

# Verify authentication worked
echo -e "\n${YELLOW}Verifying authentication...${NC}"
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✅ Successfully authenticated!${NC}"
    echo ""
    gh auth status

    # Test API access
    echo -e "\n${YELLOW}Testing API access...${NC}"
    if gh api user --jq .login &> /dev/null; then
        USER=$(gh api user --jq .login)
        echo -e "${GREEN}✓ API access working for user: $USER${NC}"
    else
        echo -e "${RED}❌ API access failed${NC}"
        exit 1
    fi

    # Get repo info if in a git directory
    if git rev-parse --git-dir > /dev/null 2>&1; then
        REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")
        if [ -n "$REPO" ]; then
            echo -e "${GREEN}✓ Connected to repository: $REPO${NC}"
        fi
    fi

    echo -e "\n${GREEN}🎉 GitHub authentication complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run: ${YELLOW}./scripts/setup-github-actions.sh${NC}"
    echo "2. Or manually add secrets: ${YELLOW}gh secret set SECRET_NAME${NC}"

else
    echo -e "${RED}❌ Authentication failed${NC}"
    echo "Please try again or check: https://cli.github.com/manual/gh_auth_login"
    exit 1
fi
