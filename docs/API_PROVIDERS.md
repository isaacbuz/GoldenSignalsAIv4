# Market Data API Providers Setup

## Overview
The GoldenSignalsAI platform supports multiple market data providers for maximum reliability and data quality. Providers are tried in order of reliability and data quality.

## Provider Priority Order

### 1. **Twelve Data** (Primary - Highly Recommended)
- **Reliability**: ⭐⭐⭐⭐⭐ Excellent
- **Free Tier**: 800 API calls/day
- **Data Quality**: Real-time, high quality
- **Setup**: Get free API key at https://twelvedata.com/
- **Environment Variable**: `TWELVEDATA_API_KEY=your_key_here`

### 2. **Finnhub** (Primary - Excellent)
- **Reliability**: ⭐⭐⭐⭐⭐ Excellent
- **Free Tier**: 60 API calls/minute
- **Data Quality**: Real-time, institutional grade
- **Setup**: Get free API key at https://finnhub.io/
- **Environment Variable**: `FINNHUB_API_KEY=your_key_here`

### 3. **Alpha Vantage** (Backup - Good)
- **Reliability**: ⭐⭐⭐⭐ Good
- **Free Tier**: 500 API calls/day, 5 calls/minute
- **Data Quality**: Good, delayed by 15-20 minutes for free tier
- **Setup**: Get free API key at https://www.alphavantage.co/
- **Environment Variable**: `ALPHA_VANTAGE_API_KEY=your_key_here`

### 4. **Polygon.io** (Professional)
- **Reliability**: ⭐⭐⭐⭐⭐ Excellent
- **Free Tier**: Limited (paid service recommended)
- **Data Quality**: Professional grade, real-time
- **Setup**: Get API key at https://polygon.io/
- **Environment Variable**: `POLYGON_API_KEY=your_key_here`

### 5. **Yahoo Finance** (Fallback)
- **Reliability**: ⭐⭐ Limited (rate limiting issues)
- **Free Tier**: Free but unreliable
- **Data Quality**: Good when working
- **Setup**: No API key required
- **Note**: Used as last resort only

### 6. **Financial Modeling Prep** (Alternative)
- **Reliability**: ⭐⭐⭐ Good
- **Free Tier**: 250 API calls/day
- **Data Quality**: Good historical data
- **Setup**: Get free API key at https://financialmodelingprep.com/
- **Environment Variable**: `FMP_API_KEY=your_key_here`

## Quick Setup (Recommended)

### Option 1: Free Tier Setup (Best Reliability)
```bash
# Set up the top 2 free providers for maximum reliability
export TWELVEDATA_API_KEY="your_twelvedata_key"
export FINNHUB_API_KEY="your_finnhub_key"
export ALPHA_VANTAGE_API_KEY="your_alphavantage_key"
```

### Option 2: Professional Setup
```bash
# Add professional providers for production use
export POLYGON_API_KEY="your_polygon_key"
export FINNHUB_API_KEY="your_finnhub_key"
export TWELVEDATA_API_KEY="your_twelvedata_key"
```

## Environment Variables Setup

### .env File (Recommended)
Create a `.env` file in the project root:
```bash
# Primary providers (recommended)
TWELVEDATA_API_KEY=your_twelvedata_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Backup providers
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here
POLYGON_API_KEY=your_polygon_key_here
FMP_API_KEY=your_fmp_key_here
```

### System Environment
```bash
# Add to ~/.bashrc or ~/.zshrc
export TWELVEDATA_API_KEY="your_key_here"
export FINNHUB_API_KEY="your_key_here"
export ALPHA_VANTAGE_API_KEY="your_key_here"
```

## Testing the Setup

### Check Provider Status
```bash
# Test market data endpoint
curl "http://localhost:8000/api/v1/market-data/AAPL"

# Check backend logs to see which provider succeeded
tail -f backend.log | grep "provider\|✅\|❌"
```

### Expected Log Output
```
INFO:main:🔄 Trying twelvedata for AAPL
INFO:main:✅ twelvedata succeeded for AAPL - 30 data points
```

## API Limits Summary

| Provider | Free Calls/Day | Rate Limit | Real-time |
|----------|----------------|------------|-----------|
| Twelve Data | 800 | 8/minute | Yes (15-min delay) |
| Finnhub | 300-1800 | 60/minute | Yes (paid) |
| Alpha Vantage | 500 | 5/minute | Yes (15-min delay) |
| Polygon | Limited | Varies | Yes (paid) |
| FMP | 250 | 10/minute | No (daily) |
| Yahoo Finance | Unlimited* | Unreliable* | Yes |

*Yahoo Finance is free but has reliability issues and rate limiting

## Troubleshooting

### Common Issues
1. **"No API key" errors**: Ensure environment variables are set correctly
2. **Rate limiting**: The system automatically tries the next provider
3. **Invalid symbols**: Try standard US stock symbols (AAPL, MSFT, etc.)

### Debug Mode
Enable detailed logging to see provider attempts:
```bash
# Check which providers are being tried
tail -f backend.log | grep "Trying\|succeeded\|failed"
```

## Data Quality Comparison

### Real-time Accuracy
1. **Polygon** - Professional/institutional grade
2. **Finnhub** - Excellent real-time data
3. **Twelve Data** - Very good, slight delay
4. **Alpha Vantage** - Good, 15-20 min delay
5. **FMP** - Daily data only
6. **Yahoo Finance** - Good when working

### Recommended Setup for Production
- **Primary**: Twelve Data + Finnhub (covers most use cases)
- **Backup**: Alpha Vantage + Polygon
- **Fallback**: Yahoo Finance (automatic)

This setup provides ~1500+ API calls per day with excellent reliability.
