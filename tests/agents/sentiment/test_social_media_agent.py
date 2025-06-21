import pytest
from agents.sentiment.social_media import SocialMediaSentimentAgent

def test_social_media_agent():
    data = {'social_media_sentiment': [
        {'score': 0.6, 'text': 'Bullish news!'},
        {'score': -0.4, 'text': 'Bearish tweet.'}
    ]}
    agent = SocialMediaSentimentAgent()
    signal = agent.process(data)
    assert signal['action'] in ['buy', 'sell', 'hold']
    assert isinstance(signal, dict)
