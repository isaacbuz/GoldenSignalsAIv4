# API Keys Setup Guide for Enhanced Sentiment Analysis

## Overview
The enhanced sentiment analysis service integrates multiple data sources to provide comprehensive market sentiment. This guide will help you obtain and configure the necessary API keys.

## Required API Keys

### 1. X (Twitter) API v2
**Purpose**: Real-time social sentiment from tweets

**How to Get**:
1. Go to https://developer.twitter.com/
2. Sign up for a developer account
3. Create a new project and app
4. Generate Bearer Token
5. Copy the Bearer Token

**Environment Variable**:
```bash
export TWITTER_BEARER_TOKEN="your_bearer_token_here"
```

**Free Tier Limits**:
- 500,000 tweets per month
- 300 requests per 15-minute window

### 2. News API
**Purpose**: Financial news sentiment analysis

**How to Get**:
1. Go to https://newsapi.org/
2. Sign up for free account
3. Get your API key from dashboard

**Environment Variable**:
```bash
export NEWS_API_KEY="your_news_api_key"
```

**Free Tier Limits**:
- 100 requests per day
- 500 requests per month

### 3. Reddit API
**Purpose**: Sentiment from WSB and finance subreddits

**How to Get**:
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - Name: GoldenSignalsAI
   - App type: script
   - Description: Trading sentiment analysis
   - Redirect URI: http://localhost:8000
4. Copy Client ID (under "personal use script")
5. Copy Secret

**Environment Variables**:
```bash
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"
```

**Free Tier Limits**:
- 60 requests per minute
- No daily limit

### 4. Alternative Free Options

#### Alpha Vantage (Already configured)
- Free tier: 5 API requests per minute, 500 per day
- Good for market data, limited sentiment

#### IEX Cloud
- Free tier: 50,000 messages per month
- Includes some sentiment data

## Setting Up Environment Variables

### Option 1: .env File (Recommended for Development)
Create a `.env` file in the project root:

```bash
# Sentiment Analysis APIs
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
NEWS_API_KEY=your_news_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Existing APIs (already configured)
ALPHA_VANTAGE_API_KEY=UBSR12WCJA4COJLC
POLYGON_API_KEY=aAAdnfA4lJ5AAr4cXT9pCmslGEHJ1mVQ
FINNHUB_API_KEY=d0ihu29r01qrfsag9qo0d0ihu29r01qrfsag9qog
```

### Option 2: Export in Terminal
```bash
# Add to ~/.zshrc or ~/.bashrc for persistence
export TWITTER_BEARER_TOKEN="your_token"
export NEWS_API_KEY="your_key"
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
```

### Option 3: Use Existing Keys (For Testing)
The system will work with partial API keys:
- **No keys**: Only mock sentiment data
- **News API only**: Real news sentiment + mock others
- **Any combination**: Real data from configured sources

## Testing Your Configuration

Run the test script:
```bash
python test_enhanced_sentiment.py
```

Expected output with all APIs configured:
```
API Key Configuration:
✅ X/Twitter API: Configured
✅ News API: Configured
✅ Reddit Client ID: Configured
✅ Reddit Client Secret: Configured
```

## API Rate Limit Management

The service implements smart rate limiting:
- 15-minute caching per symbol
- Fallback to cached data when limits exceeded
- Graceful degradation to mock data
- Error recovery with exponential backoff

## Free Alternatives for Development

If you don't want to set up all APIs:

### 1. Use Mock Mode
The system works without API keys using realistic mock data:
```python
# In simple_backend.py, sentiment will use fallbacks
sentiment_score = round(random.uniform(0.6, 0.9), 2)
```

### 2. Use Only Free APIs
- **StockTwits**: No API key required (currently blocked by 403)
- **News API**: Easy free tier signup
- **Yahoo Finance**: Sentiment from news (via yfinance)

### 3. Sample API Response
The mock sentiment provides realistic data:
```json
{
  "overall_score": 0.75,
  "overall_label": "Bullish",
  "sources": {
    "twitter": {"score": 0.8, "volume": 150},
    "news": {"score": 0.7, "volume": 25},
    "reddit": {"score": 0.75, "volume": 45}
  },
  "keywords": ["bullish", "buy", "moon", "calls"]
}
```

## Production Considerations

1. **API Key Security**:
   - Use environment variables
   - Never commit keys to git
   - Use secrets management service

2. **Rate Limit Handling**:
   - Implement request queuing
   - Use multiple API keys
   - Cache aggressively

3. **Cost Management**:
   - Monitor API usage
   - Set up billing alerts
   - Use free tiers effectively

## Troubleshooting

### StockTwits 403 Error
- Their API may require authentication now
- Use other sources as primary

### Empty Sentiment Data
- Check API key configuration
- Verify rate limits not exceeded
- Check network connectivity

### News API No Results
- Some symbols may have limited news
- Try major stocks (AAPL, TSLA, SPY)
- Check date range in query

## Next Steps

1. **Minimum Setup** (5 minutes):
   - Sign up for News API (free)
   - Add to .env file
   - Test with major stocks

2. **Full Setup** (30 minutes):
   - Get all API keys
   - Configure .env file
   - Test all sources

3. **Production Setup**:
   - Paid API tiers
   - Multiple API keys
   - Redis caching
   - Rate limit monitoring

The system is designed to work at any level of API integration, from no keys (mock data) to full production APIs. 