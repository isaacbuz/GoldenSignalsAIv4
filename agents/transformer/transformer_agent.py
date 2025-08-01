import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from agents.common.base import BaseAgent, registry
from agents.common.models import MarketData, Prediction, Signal
from agents.transformer.transformer_config import TransformerConfig
from agents.transformer.transformer_model import TransformerModel
from agents.transformer.transformer_utils import (
    calculate_attention_weights,
    get_transformer_device,
    prepare_transformer_input,
)

logger = logging.getLogger(__name__)

@registry.register_agent('transformer')
class TransformerAgent(BaseAgent):
    """Transformer-based trading agent implementation."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the transformer agent.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.model = None
        self.device = get_transformer_device()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the transformer model."""
        try:
            config = TransformerConfig(**self.config)
            self.model = TransformerModel(config).to(self.device)
            logger.info(f"Initialized transformer model with config: {config}")
        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> bool:
        """Load a trained transformer model.

        Args:
            model_path: Path to the saved model file

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Loaded transformer model from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load transformer model: {str(e)}")
            return False

    def save_model(self, model_path: str) -> bool:
        """Save the current transformer model.

        Args:
            model_path: Path to save the model file

        Returns:
            bool: True if model saved successfully, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, model_path)
            logger.info(f"Saved transformer model to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save transformer model: {str(e)}")
            return False

    def train(self, market_data: List[MarketData]) -> bool:
        """Train the transformer model on historical market data.

        Args:
            market_data: List of historical market data points

        Returns:
            bool: True if training completed successfully, False otherwise
        """
        try:
            if not market_data:
                logger.error("No training data provided")
                return False

            # Prepare training data
            X, y = prepare_transformer_input(market_data)
            if X is None or y is None:
                return False

            # Convert to tensors
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)

            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))

            # Training loop
            self.model.train()
            for epoch in range(self.config.get('epochs', 100)):
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config.get('epochs', 100)}], Loss: {loss.item():.4f}")

            logger.info("Transformer model training completed")
            return True

        except Exception as e:
            logger.error(f"Failed to train transformer model: {str(e)}")
            return False

    def predict(self, market_data: MarketData) -> Optional[Prediction]:
        """Generate predictions using the transformer model.

        Args:
            market_data: Current market data point

        Returns:
            Optional[Prediction]: Prediction object if successful, None otherwise
        """
        try:
            if self.model is None:
                logger.error("Model not initialized")
                return None

            # Prepare input data
            X = prepare_transformer_input([market_data])
            if X is None:
                return None

            # Convert to tensor
            X = torch.FloatTensor(X).to(self.device)

            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
                attention_weights = calculate_attention_weights(self.model, X)

            # Convert prediction to numpy
            prediction = output.cpu().numpy()[0]

            # Create prediction object
            return Prediction(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                predicted_price=float(prediction[0]),
                confidence=float(prediction[1]),
                metadata={
                    'attention_weights': attention_weights,
                    'model_type': 'transformer'
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate prediction: {str(e)}")
            return None

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate trading signals based on transformer model predictions.

        Args:
            market_data: Current market data point

        Returns:
            Optional[Signal]: Trading signal if generated, None otherwise
        """
        try:
            # Get model prediction
            prediction = self.predict(market_data)
            if prediction is None:
                return None

            # Calculate signal strength based on prediction confidence
            signal_strength = abs(prediction.predicted_price - market_data.close)
            signal_strength = min(signal_strength / market_data.close, 1.0)

            # Determine signal direction
            if prediction.predicted_price > market_data.close * 1.001:  # 0.1% threshold
                direction = 1  # Buy
            elif prediction.predicted_price < market_data.close * 0.999:  # 0.1% threshold
                direction = -1  # Sell
            else:
                direction = 0  # Hold

            # Only generate signal if confidence is high enough
            if prediction.confidence < self.config.get('min_confidence', 0.7):
                return None

            # Create signal object
            return Signal(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                direction=direction,
                strength=signal_strength,
                confidence=prediction.confidence,
                metadata={
                    'predicted_price': prediction.predicted_price,
                    'current_price': market_data.close,
                    'model_type': 'transformer'
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate trading signal: {str(e)}")
            return None
