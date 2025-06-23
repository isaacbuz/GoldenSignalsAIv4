import openai
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GPTModelCopilot:
    """Copilot for GPT-based feature/model critique with robust error handling and logging."""
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        openai.api_key = api_key
        self.model = model

    def critique_features(self, feature_json: Dict[str, Any]) -> str:
        """Given a feature set for a stock prediction model, suggest improvements, new features, or data cleaning tips."""
        prompt = (
            "Given the following feature set for a stock prediction model, suggest improvements, new features, or data cleaning tips. "
            f"Features: {feature_json}\n"
        )
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            logger.debug(f"GPT response: {response}")
            return response.choices[0].message["content"]
        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            return f"Error: {str(e)}"
