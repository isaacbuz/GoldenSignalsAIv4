import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class BaseMLModel:
    """Advanced base class for machine learning models with comprehensive capabilities."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        learning_rate: float = 0.001,
    ):
        """
        Initialize base machine learning model.

        Args:
            input_dim (int): Input dimension of the data
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension
            learning_rate (float): Learning rate for optimization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            {
                "message": f"Initializing {self.__class__.__name__}",
                "device": str(self.device),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
            }
        )

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """
        Preprocess and normalize input data.

        Args:
            data (np.ndarray): Input data

        Returns:
            torch.Tensor: Preprocessed tensor
        """
        # Normalize data
        scaled_data = self.scaler.fit_transform(data)
        return torch.FloatTensor(scaled_data).to(self.device)

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Train the model with comprehensive logging and metrics.

        Args:
            train_data (np.ndarray): Training input data
            train_labels (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training

        Returns:
            Dict[str, Any]: Training metrics and performance
        """
        raise NotImplementedError("Subclasses must implement training method")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            data (np.ndarray): Input data for prediction

        Returns:
            np.ndarray: Predicted values
        """
        raise NotImplementedError("Subclasses must implement prediction method")


class LSTMModel(BaseMLModel, nn.Module):
    """Advanced LSTM model for time-series prediction."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2
    ):
        """
        Initialize LSTM model with advanced configuration.

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        BaseMLModel.__init__(self, input_dim, hidden_dim)
        nn.Module.__init__(self)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass through LSTM network."""
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


class TransformerModel(BaseMLModel, nn.Module):
    """Advanced Transformer model for complex sequence prediction."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2
    ):
        """
        Initialize Transformer model with advanced configuration.

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
        """
        BaseMLModel.__init__(self, input_dim, hidden_dim)
        nn.Module.__init__(self)

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass through Transformer network."""
        x = self.embedding(x)
        transformer_out = self.transformer_encoder(x)
        predictions = self.fc(transformer_out.mean(dim=1))
        return predictions
