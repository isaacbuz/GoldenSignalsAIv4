import logging
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)

class FinBERTSentimentAgent(BaseAgent):
    """Agent for FinBERT-based sentiment analysis with robust error handling."""
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        super().__init__(name="FinBERT", agent_type="sentiment")
        try:
            self.classifier: Pipeline = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT pipeline: {e}")
            self.classifier = None

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of texts for sentiment analysis."""
        texts = data.get("texts", [])
        if not self.classifier:
            logger.error("FinBERT pipeline not initialized.")
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": "FinBERT pipeline not initialized."}}
        if not texts or not isinstance(texts, list):
            logger.warning("No texts provided or input is not a list.")
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": "No texts provided."}}

        try:
            results = self.classifier(texts)
            sentiment_scores = []
            for res in results:
                label = res.get("label", "neutral").lower()
                if label == "positive":
                    score = 1
                elif label == "negative":
                    score = -1
                else:
                    score = 0
                sentiment_scores.append(score)

            avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

            # Convert to trading signal
            if avg_score > 0.2:
                action = "buy"
                confidence = min(abs(avg_score), 1.0)
            elif avg_score < -0.2:
                action = "sell"
                confidence = min(abs(avg_score), 1.0)
            else:
                action = "hold"
                confidence = 0.0

            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "average_score": avg_score,
                    "raw_results": results,
                    "analyzed_texts": len(texts)
                }
            }
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}
