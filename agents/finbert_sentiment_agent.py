import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)

class FinBERTSentimentAgent:
    """Agent for FinBERT-based sentiment analysis with robust error handling."""
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        try:
            self.classifier: Pipeline = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT pipeline: {e}")
            self.classifier = None

    def analyze_texts(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze a list of texts for sentiment. Returns average score and raw results. Handles batching and errors."""
        if not self.classifier:
            logger.error("FinBERT pipeline not initialized.")
            return {"error": "FinBERT pipeline not initialized."}
        if not texts or not isinstance(texts, list):
            logger.warning("No texts provided or input is not a list.")
            return {"error": "No texts provided."}
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
            return {
                "average_score": avg_score,
                "raw_results": results
            }
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {"error": str(e)}
