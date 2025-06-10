import pytest
from agents.sentiment.news import NewsSentimentAgent

def test_news_agent():
    data = {'news_sentiment': [
        {'sentiment_score': 0.7, 'headline': 'Positive earnings report.'},
        {'sentiment_score': -0.5, 'headline': 'Regulatory risk.'}
    ]}
    agent = NewsSentimentAgent()
    signal = agent.process(data)
    assert signal['action'] in ['buy', 'sell', 'hold']
    assert isinstance(signal, dict)
