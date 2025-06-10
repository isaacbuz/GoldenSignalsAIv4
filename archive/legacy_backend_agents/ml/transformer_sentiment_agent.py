from archive.legacy_backend_agents.base import BaseSignalAgent
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class TransformerSentimentAgent(BaseSignalAgent):
    """
    Uses a transformer (e.g., FinBERT) for fine-grained sentiment analysis of financial text.
    """
    def __init__(self, symbol: str, model_name="yiyanghkust/finbert-tone"):
        super().__init__(symbol)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def run(self, texts: list) -> dict:
        if not texts:
            return {"agent": "TransformerSentimentAgent", "signal": "neutral", "confidence": 0, "explanation": "No texts provided."}
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        avg_probs = np.mean(probs, axis=0)
        labels = ["negative", "neutral", "positive"]
        idx = int(np.argmax(avg_probs))
        signal = labels[idx]
        explanation = f"FinBERT: {dict(zip(labels, avg_probs.round(2)))}"
        return {"agent": "TransformerSentimentAgent", "signal": signal, "confidence": int(100 * avg_probs[idx]), "explanation": explanation}
