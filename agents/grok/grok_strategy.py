import requests

class GrokStrategyAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.x.ai/v1/grok"

    def generate_logic(self, symbol, timeframe="1h"):
        prompt = f"Generate trading logic for {symbol} on a {timeframe} timeframe using 9 EMA and price action. Output a simplified pseudocode strategy."
        response = requests.post(self.endpoint, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }, json={"prompt": prompt})
        data = response.json()
        return data.get("logic", "")
