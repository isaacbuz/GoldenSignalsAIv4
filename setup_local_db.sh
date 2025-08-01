#!/bin/bash

# Setup Local PostgreSQL for GoldenSignalsAI
# This script will install and configure PostgreSQL locally on macOS

echo "=================================================="
echo "🚀 GoldenSignalsAI Local Database Setup"
echo "=================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "📦 Installing PostgreSQL..."
    brew install postgresql@15
    brew link postgresql@15 --force
else
    echo "✅ PostgreSQL is already installed"
fi

# Start PostgreSQL service
echo "🔧 Starting PostgreSQL service..."
brew services start postgresql@15

# Wait for PostgreSQL to start
sleep 3

# Create database and user
echo "🗄️ Setting up database..."

# Create the database user
psql postgres -c "CREATE USER goldensignalsai WITH PASSWORD 'goldensignals123';" 2>/dev/null || echo "   User already exists"

# Create the database
psql postgres -c "CREATE DATABASE goldensignalsai OWNER goldensignalsai;" 2>/dev/null || echo "   Database already exists"

# Grant all privileges
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE goldensignalsai TO goldensignalsai;"

# Create .env.local file with local database settings
echo "📝 Creating .env.local file..."
cat > .env.local << EOF
# GoldenSignalsAI V3 - Local Database Configuration

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL Configuration (Local)
DATABASE_URL=postgresql+asyncpg://goldensignalsai:goldensignals123@localhost:5432/goldensignalsai
DB_HOST=localhost
DB_PORT=5432
DB_NAME=goldensignalsai
DB_USER=goldensignalsai
DB_PASSWORD=goldensignals123

# Redis Configuration (Local)
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Environment
ENVIRONMENT=development
DEBUG=true

# Security
SECRET_KEY=your-ultra-secure-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# =============================================================================
# EXTERNAL SERVICES (Optional)
# =============================================================================

# Market Data APIs
POLYGON_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPHA_VANTAGE_KEY=

# AI Services
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Monitoring
SENTRY_DSN=
PROMETHEUS_ENABLED=false

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Enable/disable features
ENABLE_LIVE_DATA=true
ENABLE_AI_CHAT=true
ENABLE_OPTIONS_TRADING=true
ENABLE_BACKTESTING=true

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL=INFO
LOG_FORMAT=json

# =============================================================================
# RATE LIMITING
# =============================================================================

RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
EOF

echo "✅ .env.local file created"

# Test the connection
echo ""
echo "🔍 Testing database connection..."
psql -U goldensignalsai -d goldensignalsai -h localhost -c "SELECT version();" -W

echo ""
echo "=================================================="
echo "✅ Local PostgreSQL setup complete!"
echo "=================================================="
echo ""
echo "📋 Database Details:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: goldensignalsai"
echo "   User: goldensignalsai"
echo "   Password: goldensignals123"
echo ""
echo "🎯 Next Steps:"
echo "1. Copy .env.local to .env: cp .env.local .env"
echo "2. Test the connection: python test_local_db.py"
echo "3. Start the application: python src/main.py"
echo ""
echo "💡 To stop PostgreSQL: brew services stop postgresql@15"
echo "💡 To restart PostgreSQL: brew services restart postgresql@15"
