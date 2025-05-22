import requests

class GrokBacktestCritic:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.x.ai/v1/grok"

    def critique(self, logic, win_rate, avg_return):
        prompt = (
            f"Backtest result: Strategy logic: {logic}. "
            f"Win rate: {win_rate}%. Avg return: {avg_return}%. "
            "Suggest two improvements."
        )
        response = requests.post(self.endpoint, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }, json={"prompt": prompt})
        data = response.json()
        return data.get("suggestions", [])
