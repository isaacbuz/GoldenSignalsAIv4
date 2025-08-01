"""
Base class for pretrained model agents.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
from agents.common.base_agent import BaseAgent
from agents.ml.pretrained.model_metadata import PretrainedModelInfo, get_model_info
from keras.models import load_model

logger = logging.getLogger(__name__)

class BasePretrainedAgent(BaseAgent):
    """Base class for agents using pretrained models."""

    def __init__(
        self,
        name: str,
        model_name: str,
        model_path: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pretrained agent.

        Args:
            name: Agent identifier
            model_name: Name of pretrained model
            model_path: Path to model file
            config: Optional configuration
        """
        super().__init__(name, config)
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.model_info = get_model_info(model_name)
        self._load_model()

    def _load_model(self) -> None:
        """Load the pretrained model."""
        try:
            self.model = load_model(self.model_path)
            logger.info(f"Loaded pretrained model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")

    def _validate_input_shape(self, data: np.ndarray) -> bool:
        """
        Validate input data shape matches model requirements.

        Args:
            data: Input data array

        Returns:
            True if valid, False otherwise
        """
        if self.model_info is None:
            return True  # Skip validation if no metadata

        expected_shape = self.model_info.input_shape
        if len(data.shape) != len(expected_shape):
            return False

        for actual, expected in zip(data.shape[1:], expected_shape):
            if actual != expected:
                return False

        return True

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using pretrained model.

        Args:
            data: Input data dictionary

        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.validate_input(data):
            return {"error": "Invalid input data"}

        try:
            # Prepare input data
            X = self._prepare_input(data)
            if not self._validate_input_shape(X):
                return {"error": "Invalid input shape"}

            # Get prediction
            prediction = self.model.predict(X)

            # Format output
            result = self._format_output(prediction, data)

            # Add metadata
            if self.model_info:
                result.update({
                    "model_type": self.model_info.type,
                    "source": self.model_info.source,
                    "trained_on": self.model_info.trained_on
                })

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def _prepare_input(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare input data for model.

        Args:
            data: Input data dictionary

        Returns:
            Numpy array ready for model input
        """
        raise NotImplementedError

    def _format_output(self, prediction: np.ndarray, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format model output.

        Args:
            prediction: Raw model prediction
            data: Original input data

        Returns:
            Formatted prediction dictionary
        """
        raise NotImplementedError
