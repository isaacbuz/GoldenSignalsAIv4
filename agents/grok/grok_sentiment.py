import requests

class GrokSentimentAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.x.ai/v1/grok"

    def get_sentiment_score(self, symbol):
        prompt = f"Analyze real-time market sentiment for {symbol} based on recent news and X posts. Provide a sentiment score (0-100) and key trends."
        response = requests.post(self.endpoint, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }, json={"prompt": prompt})
        data = response.json()
        return data.get("sentimentScore", 50)
