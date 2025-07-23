#!/bin/bash

# Script to add GitHub secrets from .env file
# This is a more secure approach that reads from your existing .env file

set -e

echo "🔐 Adding GitHub Secrets from .env file"
echo "======================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if gh is authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI is not authenticated!${NC}"
    echo "Please run: gh auth login"
    exit 1
fi

# Check if .env file exists
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please ensure you're in the project root directory"
    exit 1
fi

echo -e "${GREEN}✅ Found .env file${NC}"
echo ""

# Function to add a secret
add_secret() {
    local name=$1
    local value=$2
    echo -n "Adding $name... "
    if echo "$value" | gh secret set "$name" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠️  Failed (may already exist)${NC}"
    fi
}

# Read .env file and add secrets
echo "Processing .env file..."
echo "======================="

while IFS='=' read -r key value; do
    # Skip empty lines and comments
    if [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Remove quotes from value
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

    # Skip if value is empty or placeholder
    if [[ -z "$value" || "$value" == "your-"* || "$value" == "" ]]; then
        echo -e "${YELLOW}⚠️  Skipping $key (empty or placeholder value)${NC}"
        continue
    fi

    # Add the secret
    add_secret "$key" "$value"
done < "$ENV_FILE"

echo ""
echo -e "${GREEN}✅ Secrets processing complete!${NC}"
echo ""
echo "Listing all secrets:"
echo "===================="
gh secret list

echo ""
echo -e "${YELLOW}⚠️  Important Notes:${NC}"
echo "1. Placeholder values (starting with 'your-') were skipped"
echo "2. Empty values were skipped"
echo "3. Update these manually in GitHub Settings → Secrets"
echo "4. Consider different values for production vs development"
