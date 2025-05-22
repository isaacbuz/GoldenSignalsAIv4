from agents.finbert_sentiment_agent import FinBERTSentimentAgent

def test_analyze_texts():
    agent = FinBERTSentimentAgent()
    texts = ["Stocks are up.", "Bad news for the market."]
    result = agent.analyze_texts(texts)
    assert "average_score" in result
    assert "raw_results" in result
