from backend.agents.base import BaseSignalAgent
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class BERTEventExtractionAgent(BaseSignalAgent):
    """
    Uses a transformer (e.g., BERT) to extract financial events (e.g., mergers, earnings) from news headlines or tweets.
    """
    def __init__(self, symbol: str, model_name="dslim/bert-base-NER"):
        super().__init__(symbol)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def run(self, texts: list) -> dict:
        if not texts:
            return {"agent": "BERTEventExtractionAgent", "events": [], "confidence": 0, "explanation": "No texts provided."}
        events = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs).logits
            predictions = torch.argmax(outputs, dim=2)[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            for token, pred in zip(tokens, predictions):
                if pred != 0:  # 0 = 'O' (no entity)
                    events.append(token)
        explanation = f"Extracted events: {events}"
        return {"agent": "BERTEventExtractionAgent", "events": events, "confidence": 80 if events else 30, "explanation": explanation}
