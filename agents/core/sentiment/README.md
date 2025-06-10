# Sentiment Analysis Agents

## Overview
This directory contains agents specialized in market sentiment analysis from various sources including news, social media, and analyst reports.

## Components

### News Analysis
- `FinBERTSentimentAgent`: BERT-based financial news sentiment analysis
- News source adapters for major providers
- Historical news impact analysis

### Social Media Analysis
- `SocialSentimentAgent`: Social media sentiment aggregation
- Platform-specific adapters (Twitter, StockTwits, Reddit)
- Trend analysis and virality detection

### Meta Analysis
- `SentimentAggregator`: Combines multiple sentiment sources
- Cross-source correlation analysis
- Sentiment impact scoring

## Directory Structure
```
sentiment/
├── news/              # News-based sentiment analysis
├── social/            # Social media sentiment
├── meta/             # Meta-analysis and aggregation
└── models/           # Pretrained sentiment models
```

## Usage Examples

```python
from agents.core.sentiment import FinBERTSentimentAgent, SocialSentimentAgent

# Initialize news sentiment agent
news_sentiment = FinBERTSentimentAgent(
    model_path="finbert-sentiment",
    batch_size=32
)

# Initialize social sentiment agent
social_sentiment = SocialSentimentAgent(
    platforms=["twitter", "stocktwits"],
    update_interval=300  # 5 minutes
)

# Combine sentiments
from agents.core.sentiment import SentimentAggregator
aggregator = SentimentAggregator([news_sentiment, social_sentiment])
```

## Best Practices
1. Handle real-time data streams properly
2. Implement proper text preprocessing
3. Consider source credibility weights
4. Handle missing data gracefully
5. Implement rate limiting for API calls
6. Cache frequently accessed data
7. Validate sentiment scores

## Data Sources
- News APIs (Bloomberg, Reuters, etc.)
- Social Media APIs
- Analyst Report Databases
- Alternative Data Sources

## Performance Considerations
- Use async processing for real-time data
- Implement proper rate limiting
- Cache processed sentiments
- Use batch processing where applicable

## Dependencies
Required packages:
- transformers>=4.5.0
- torch>=1.9.0
- nltk>=3.6.0
- tweepy>=4.0.0
- beautifulsoup4>=4.9.3 