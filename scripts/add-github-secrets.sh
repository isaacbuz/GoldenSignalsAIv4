#!/bin/bash

# Script to add GitHub secrets to isaacbuz/GoldenSignalsAIv4 repository

echo "Adding GitHub secrets to isaacbuz/GoldenSignalsAIv4..."

# Repository name
REPO="isaacbuz/GoldenSignalsAIv4"

# Function to add a secret
add_secret() {
    local secret_name=$1
    local secret_value=$2

    echo "Adding secret: $secret_name"
    echo "$secret_value" | gh secret set "$secret_name" -R "$REPO"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully added $secret_name"
    else
        echo "✗ Failed to add $secret_name"
    fi
    echo ""
}

# Check if gh is authenticated
echo "Checking GitHub CLI authentication..."
if ! gh auth status &>/dev/null; then
    echo "❌ You are not authenticated with GitHub CLI."
    echo "Please run: gh auth login"
    echo "Then run this script again."
    exit 1
fi

echo "✓ GitHub CLI is authenticated"
echo ""

# Add all secrets
add_secret "ANTHROPIC_API_KEY" "YOUR_ANTHROPIC_API_KEY"
add_secret "XAI_API_KEY" "YOUR_XAI_API_KEY"
add_secret "TWELVEDATA_API_KEY" "91b91adf18634887b02865b314ba79af"
add_secret "FINNHUB_API_KEY" "d0ihu29r01qrfsag9qo0d0ihu29r01qrfsag9qog"
add_secret "DATABASE_URL" "postgresql://goldensignalsai:goldensignals123@localhost:5432/goldensignalsai"
add_secret "SECRET_KEY" "your-secret-key-here-minimum-32-chars"
add_secret "JWT_SECRET_KEY" "your-jwt-secret-key-here"

echo "All secrets have been processed!"
echo ""
echo "You can verify the secrets were added by running:"
echo "gh secret list -R $REPO"
