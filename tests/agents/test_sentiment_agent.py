"""
Tests for the sentiment analysis agent.
"""
import pytest
from agents.sentiment.sentiment_agent import SentimentAgent

def test_sentiment_initialization():
    """Test sentiment agent initialization"""
    agent = SentimentAgent(name="Sentiment_Test")
    assert agent.name == "Sentiment_Test"
    assert agent.agent_type == "sentiment"
    assert agent.analyzer is not None

def test_text_analysis():
    """Test sentiment analysis of individual texts"""
    agent = SentimentAgent()

    # Test positive sentiment
    positive_text = "The company reported excellent earnings, exceeding all expectations!"
    positive_scores = agent.analyze_text(positive_text)
    assert positive_scores["compound"] > 0
    assert positive_scores["pos"] > positive_scores["neg"]

    # Test negative sentiment
    negative_text = "The company's performance was disappointing, with significant losses."
    negative_scores = agent.analyze_text(negative_text)
    assert negative_scores["compound"] < 0
    assert negative_scores["neg"] > negative_scores["pos"]

    # Test neutral sentiment
    neutral_text = "The company released its quarterly report today."
    neutral_scores = agent.analyze_text(neutral_text)
    assert abs(neutral_scores["compound"]) < 0.2
    assert neutral_scores["neu"] > neutral_scores["pos"]
    assert neutral_scores["neu"] > neutral_scores["neg"]

def test_sentiment_signals(sample_news_data):
    """Test sentiment signal generation"""
    agent = SentimentAgent()

    # Test with sample news data
    result = agent.process({"texts": sample_news_data})
    assert "action" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert result["action"] in ["buy", "sell", "hold"]

    # Test with strongly positive texts
    positive_texts = [
        "Company reports record profits!",
        "Breakthrough innovation announced!",
        "Stock price soars on excellent news!"
    ]
    result = agent.process({"texts": positive_texts})
    assert result["action"] == "buy"
    assert result["confidence"] > 0.5

    # Test with strongly negative texts
    negative_texts = [
        "Company faces major lawsuit",
        "Significant losses reported",
        "Market share declining rapidly"
    ]
    result = agent.process({"texts": negative_texts})
    assert result["action"] == "sell"
    assert result["confidence"] > 0.5

def test_sentiment_error_handling():
    """Test sentiment agent error handling"""
    agent = SentimentAgent()

    # Test missing data
    with pytest.raises(ValueError):
        agent.process({})

    # Test empty text list
    result = agent.process({"texts": []})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert "error" in result["metadata"]

    # Test invalid data type
    with pytest.raises(AttributeError):
        agent.process({"texts": [123]})  # Numbers instead of strings

def test_sentiment_metadata():
    """Test sentiment metadata in results"""
    agent = SentimentAgent()
    texts = ["Positive news about growth", "Some concerns about costs"]

    result = agent.process({"texts": texts})
    metadata = result["metadata"]

    assert "average_sentiment" in metadata
    assert "sentiment_distribution" in metadata
    assert "analyzed_texts" in metadata
    assert metadata["analyzed_texts"] == len(texts)

    distribution = metadata["sentiment_distribution"]
    assert "positive" in distribution
    assert "negative" in distribution
    assert "neutral" in distribution
    assert sum(distribution.values()) == len(texts)

def test_sentiment_aggregation():
    """Test sentiment aggregation with mixed signals"""
    agent = SentimentAgent()
    mixed_texts = [
        "Company reports strong growth",  # Positive
        "Regulatory challenges ahead",    # Negative
        "New product launch next week",   # Neutral
        "Market share increasing",        # Positive
        "Costs rising moderately"        # Slightly negative
    ]

    result = agent.process({"texts": mixed_texts})

    # Verify that confidence reflects mixed signals
    assert 0 <= result["confidence"] <= 1

    # Check distribution
    distribution = result["metadata"]["sentiment_distribution"]
    assert distribution["positive"] > 0
    assert distribution["negative"] > 0

    # Verify average sentiment is between strongest positive and negative
    sentiment = result["metadata"]["average_sentiment"]
    assert -1 <= sentiment <= 1
