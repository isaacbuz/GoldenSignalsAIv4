from archive.legacy_backend_agents.base import BaseSignalAgent
from transformers import CLIPProcessor, CLIPModel
import torch

class MultimodalNewsImageAgent(BaseSignalAgent):
    """
    Uses a multimodal model (e.g., CLIP) to analyze news headlines and associated images for sentiment or event detection.
    """
    def __init__(self, symbol: str, model_name="openai/clip-vit-base-patch16"):
        super().__init__(symbol)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def run(self, texts: list, images: list) -> dict:
        # texts: list of news headlines
        # images: list of PIL.Image objects (must be same length as texts)
        if not texts or not images or len(texts) != len(images):
            return {"agent": "MultimodalNewsImageAgent", "signal": None, "confidence": 0, "explanation": "Texts and images required and must match in length."}
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image.cpu().numpy()
        # Example: use logits to score relevance or sentiment (customize as needed)
        avg_score = float(logits_per_image.mean())
        explanation = f"Average CLIP relevance score: {avg_score:.2f}"
        return {"agent": "MultimodalNewsImageAgent", "score": avg_score, "confidence": 75, "explanation": explanation}
