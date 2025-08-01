#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up Live Transformer Evaluation Environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/transformer
mkdir -p ml_training/metrics
mkdir -p logs

# Set up logging
touch logs/live_evaluation.log

# Check for API keys
if [ -z "$POLYGON_API_KEY" ]; then
    echo "⚠️  Warning: POLYGON_API_KEY not set"
    echo "   You can set it by running:"
    echo "   export POLYGON_API_KEY='your_api_key_here'"
fi

echo "✅ Setup complete!"
echo "To start the live evaluation, run:"
echo "python ml_training/evaluate_live_transformer.py"
