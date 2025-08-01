"""
Chart vision agent for visual and numerical pattern recognition.
"""
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src.base.base_agent import BaseAgent
from src.predictive.patterns.candlestick_pattern_agent import CandlestickPatternAgent

from .chart_model import ChartPatternModel

logger = logging.getLogger(__name__)

class ChartVisionAgent(BaseAgent):
    """Agent that combines visual and numerical pattern recognition for chart analysis."""

    def __init__(
        self,
        name: str = "ChartVision",
        model_path: str = "models/yolov8_chart.pt",
        min_visual_confidence: float = 0.5,
        min_pattern_confidence: float = 0.6
    ):
        """
        Initialize chart vision agent.

        Args:
            name: Agent name
            model_path: Path to YOLOv8 model
            min_visual_confidence: Minimum confidence for visual patterns
            min_pattern_confidence: Minimum confidence for candlestick patterns
        """
        super().__init__(name=name, agent_type="vision")
        self.min_visual_confidence = min_visual_confidence
        self.min_pattern_confidence = min_pattern_confidence

        # Initialize sub-components
        try:
            self.vision_model = ChartPatternModel(model_path)
            self.pattern_agent = CandlestickPatternAgent(
                min_confidence=min_pattern_confidence
            )
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            self.vision_model = None
            self.pattern_agent = None

    def analyze_visual_patterns(
        self,
        image: Image.Image
    ) -> List[Dict[str, Any]]:
        """Analyze chart image for visual patterns."""
        try:
            if self.vision_model is None:
                return []

            # Get predictions from vision model
            predictions = self.vision_model.predict(image)

            # Filter by confidence
            patterns = [
                p for p in predictions
                if p["confidence"] >= self.min_visual_confidence
            ]

            return patterns

        except Exception as e:
            logger.error(f"Visual pattern analysis failed: {str(e)}")
            return []

    def analyze_candlestick_patterns(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market data for candlestick patterns."""
        try:
            if self.pattern_agent is None:
                return {}

            # Process data through candlestick pattern agent
            result = self.pattern_agent.process(market_data)

            return result

        except Exception as e:
            logger.error(f"Candlestick pattern analysis failed: {str(e)}")
            return {}

    def create_overlay_image(
        self,
        image: Image.Image,
        visual_patterns: List[Dict[str, Any]],
        candlestick_patterns: Dict[str, Any]
    ) -> bytes:
        """Create an overlay image with detected patterns."""
        try:
            overlay = image.convert("RGBA").copy()
            draw = ImageDraw.Draw(overlay)

            # Draw visual patterns
            for pattern in visual_patterns:
                box = pattern["box"]
                confidence = pattern["confidence"]
                pattern_name = pattern["pattern"]

                # Draw box
                draw.rectangle(box, outline="yellow", width=3)

                # Draw label with confidence
                label = f"{pattern_name} ({confidence:.2f})"
                draw.text((box[0], box[1] - 20), label, fill="yellow")

            # Draw candlestick patterns if any found
            if candlestick_patterns.get("action") != "hold":
                patterns = candlestick_patterns.get("metadata", {}).get("patterns", {})
                y_pos = 30
                for pattern, value in patterns.items():
                    text = f"{pattern}: {value}"
                    draw.text((10, y_pos), text, fill="white")
                    y_pos += 20

            # Convert to bytes
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Overlay creation failed: {str(e)}")
            return b""

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chart image and market data for pattern recognition."""
        try:
            if "image" not in data:
                raise ValueError("Missing chart image")

            image = data["image"]
            market_data = data.get("market_data", {})

            # Analyze patterns
            visual_patterns = self.analyze_visual_patterns(image)
            candlestick_patterns = self.analyze_candlestick_patterns(market_data)

            # Create overlay
            overlay_image = self.create_overlay_image(
                image,
                visual_patterns,
                candlestick_patterns
            )

            # Combine signals
            visual_signal = any(
                p["confidence"] >= self.min_visual_confidence
                for p in visual_patterns
            )
            candlestick_signal = candlestick_patterns.get("action") != "hold"

            # Generate combined signal
            if visual_signal and candlestick_signal:
                action = candlestick_patterns["action"]
                # Average of visual and candlestick confidences
                confidence = (
                    max(p["confidence"] for p in visual_patterns) +
                    candlestick_patterns["confidence"]
                ) / 2
            elif visual_signal:
                action = "alert"  # Visual pattern detected but no candlestick confirmation
                confidence = max(p["confidence"] for p in visual_patterns)
            else:
                action = candlestick_patterns.get("action", "hold")
                confidence = candlestick_patterns.get("confidence", 0.0)

            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "visual_patterns": visual_patterns,
                    "candlestick_patterns": candlestick_patterns.get("metadata", {}),
                    "overlay_image": overlay_image
                }
            }

        except Exception as e:
            logger.error(f"Chart vision processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
