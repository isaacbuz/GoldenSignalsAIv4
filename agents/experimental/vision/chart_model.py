"""
Chart Vision Model Loader & Inference
Loads a YOLOv8 or compatible model and predicts chart patterns from images.
"""
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


class ChartPatternModel:
    def __init__(self, model_path="models/yolov8_chart.pt"):
        # For demo: fallback to yolov5 if yolov8 is unavailable
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        img_np = np.array(image)
        results = self.model(img_np)
        predictions = results.pandas().xyxy[0]
        patterns = []
        for _, row in predictions.iterrows():
            patterns.append({
                "pattern": row["name"],
                "confidence": float(row["confidence"]),
                "box": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            })
        return patterns
