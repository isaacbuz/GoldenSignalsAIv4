#!/bin/bash

# Security Cleanup Script for GoldenSignalsAIv4
# This script helps remove sensitive data from git history and set up secure environment

echo "🔒 GoldenSignalsAI Security Cleanup Script"
echo "========================================="
echo ""
echo "⚠️  WARNING: This script will rewrite git history!"
echo "⚠️  Make sure you have a backup of your repository"
echo ""
read -p "Do you want to continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "📝 Step 1: Creating backup of current .env file..."
if [ -f ".env" ]; then
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backup created"
else
    echo "⚠️  No .env file found"
fi

echo ""
echo "🗑️  Step 2: Removing .env from git history..."
echo "This may take a while for large repositories..."

# Remove .env from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

echo "✅ .env removed from git history"

echo ""
echo "🔄 Step 3: Cleaning up git references..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "✅ Git cleanup complete"

echo ""
echo "📤 Step 4: Force push to remote (if desired)..."
echo "⚠️  WARNING: This will rewrite history on the remote!"
echo "⚠️  All collaborators will need to re-clone the repository"
echo ""
echo "To push these changes to remote, run:"
echo "  git push origin --force --all"
echo "  git push origin --force --tags"

echo ""
echo "🔑 Step 5: Rotate your API keys..."
echo "You MUST rotate the following API keys immediately:"
echo ""
echo "1. OpenAI API Key"
echo "   👉 https://platform.openai.com/api-keys"
echo ""
echo "2. Anthropic API Key"
echo "   👉 https://console.anthropic.com/settings/keys"
echo ""
echo "3. xAI (Grok) API Key"
echo "   👉 https://x.ai/api"
echo ""
echo "4. TwelveData API Key"
echo "   👉 https://twelvedata.com/account/api-keys"
echo ""
echo "5. Finnhub API Key"
echo "   👉 https://finnhub.io/dashboard"
echo ""
echo "6. Alpha Vantage API Key"
echo "   👉 https://www.alphavantage.co/support/#api-key"
echo ""
echo "7. Polygon API Key"
echo "   👉 https://polygon.io/dashboard/api-keys"
echo ""
echo "8. FMP API Key"
echo "   👉 https://site.financialmodelingprep.com/developer/docs"
echo ""

echo "🔐 Step 6: Generate secure keys..."
echo "Here are some secure random keys for your new .env:"
echo ""
echo "SECRET_KEY=$(openssl rand -base64 32)"
echo "JWT_SECRET_KEY=$(openssl rand -base64 32)"
echo "MCP_API_KEY=$(openssl rand -base64 32)"

echo ""
echo "✅ Security cleanup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Rotate ALL API keys listed above"
echo "2. Create a new .env file from .env.example"
echo "3. Fill in the new API keys"
echo "4. Never commit .env to git again!"
echo "5. Consider using a secret management service (AWS Secrets Manager, etc.)"
echo ""
