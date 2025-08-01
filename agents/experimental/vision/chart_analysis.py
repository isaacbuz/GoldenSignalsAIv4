"""
Chart Analysis Module

Accepts chart images (screenshots or camera frames), detects technical patterns, and generates option trade signals.
Uses ChartPatternModel for real pattern detection and overlay drawing.
"""
import base64
import io
from typing import Any, Dict

from fastapi import UploadFile
from PIL import Image, ImageDraw

from .chart_model import ChartPatternModel

model = ChartPatternModel()

def analyze_chart_image(file: UploadFile) -> Dict[str, Any]:
    """
    Analyze an uploaded chart image, detect patterns, and return results + overlay image.
    Args:
        file (UploadFile): Uploaded chart image file.
    Returns:
        dict: patterns, overlay_image (bytes)
    """
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        patterns = model.predict(image)
        # Draw overlays
        overlay = image.convert("RGBA").copy()
        draw = ImageDraw.Draw(overlay)
        for pat in patterns:
            box = pat["box"]
            draw.rectangle(box, outline="yellow", width=3)
            draw.text((box[0], box[1]), pat["pattern"], fill="yellow")
        # Save overlay to bytes
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        overlay_bytes = buf.getvalue()
        return {
            "patterns": patterns,
            "overlay_image": overlay_bytes
        }
    except Exception as e:
        return {"error": str(e)}
