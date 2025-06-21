"""
Transformer-based trading agent for GoldenSignalsAI.
Implements a transformer model for market prediction and signal generation.
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import math

from src.agents.common.base.base_agent import BaseAgent
from src.agents.common.models.market_data import MarketData
from src.agents.common.models.prediction import Prediction
from src.agents.common.models.signal import Signal, SignalType, SignalStrength
from src.agents.common.utils.torch_utils import get_device
from src.agents.common.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    """Transformer model for market prediction"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerAgent(BaseAgent):
    """
    Transformer-based trading agent that uses attention mechanisms
    for market prediction and signal generation.
    """
    
    def __init__(
        self,
        model_path: str,
        input_dim: int = 10,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        confidence_threshold: float = 0.7
    ):
        super().__init__()
        self.model_path = model_path
        self.device = get_device()
        
        # Initialize model
        self.model = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)
        
        # Load trained model if exists
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
    def preprocess_data(self, data: MarketData) -> torch.Tensor:
        """Preprocess market data for the transformer model."""
        # Extract features
        features = np.array([
            data.open,
            data.high,
            data.low,
            data.close,
            data.volume,
            data.vwap,
            data.trades,
            data.bid_volume,
            data.ask_volume,
            data.spread
        ])
        
        # Normalize features
        features = (features - features.mean()) / features.std()
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return features
    
    def predict(self, data: MarketData) -> Prediction:
        """Generate prediction using the transformer model."""
        try:
            # Preprocess data
            features = self.preprocess_data(data)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self.model(features)
                confidence = torch.sigmoid(prediction).item()
            
            # Create prediction object
            pred = Prediction(
                timestamp=get_timestamp(),
                symbol=data.symbol,
                price=data.close,
                prediction=prediction.item(),
                confidence=confidence,
                metadata={
                    'model_type': 'transformer',
                    'model_path': self.model_path
                }
            )
            
            return pred
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise
    
    def generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate trading signal based on model prediction."""
        try:
            # Get prediction
            prediction = self.predict(data)
            
            # Check confidence threshold
            if prediction.confidence < self.confidence_threshold:
                return None
            
            # Determine signal type and strength
            if prediction.prediction > 0:
                signal_type = SignalType.BUY
                strength = SignalStrength.STRONG if prediction.confidence > 0.8 else SignalStrength.MODERATE
            else:
                signal_type = SignalType.SELL
                strength = SignalStrength.STRONG if prediction.confidence > 0.8 else SignalStrength.MODERATE
            
            # Create signal
            signal = Signal(
                timestamp=get_timestamp(),
                symbol=data.symbol,
                type=signal_type,
                strength=strength,
                price=data.close,
                confidence=prediction.confidence,
                metadata={
                    'model_type': 'transformer',
                    'model_path': self.model_path,
                    'prediction': prediction.prediction
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    def process_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Process market data and generate trading signal using transformer model.
        
        Args:
            market_data (MarketData): Market data to process
            
        Returns:
            Dict[str, Any]: Processed signal with prediction and metadata
        """
        try:
            # Preprocess data
            input_data = self.preprocess_data(market_data)
            
            # Get model prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_data.unsqueeze(0))
                prediction = prediction.item()
            
            # Calculate confidence based on model uncertainty
            confidence = 1.0 - min(abs(prediction), 1.0)
            
            # Generate signal
            signal = self.generate_signal(market_data)
            
            # Update performance tracking
            self.prediction_history.append({
                'timestamp': datetime.utcnow(),
                'prediction': prediction,
                'confidence': confidence,
                'signal': signal
            })
            
            return {
                'agent_name': self.name,
                'agent_type': 'transformer',
                'signal': signal.type.value,
                'strength': signal.strength.value,
                'confidence': confidence,
                'prediction': prediction,
                'metadata': {
                    'model_version': '1.0.0',
                    'prediction_horizon': self.prediction_horizon,
                    'sequence_length': self.seq_length
                }
            }
            
        except Exception as e:
            logger.error(f"Error in transformer agent: {str(e)}")
            return {
                'agent_name': self.name,
                'agent_type': 'transformer',
                'signal': 'hold',
                'strength': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def train(self, market_data: MarketData, target: np.ndarray):
        """Train the transformer model on new data"""
        try:
            self.model.train()
            input_data = self.preprocess_data(market_data)
            target_tensor = torch.FloatTensor(target)
            
            self.optimizer.zero_grad()
            output = self.model(input_data.unsqueeze(0))
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Training error in transformer agent: {str(e)}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'average_confidence': 0.0,
                'accuracy': 0.0
            }
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        confidences = [p['confidence'] for p in recent_predictions]
        
        return {
            'total_predictions': len(self.prediction_history),
            'average_confidence': np.mean(confidences),
            'recent_accuracy': self._calculate_accuracy(recent_predictions)
        }
    
    def _calculate_accuracy(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate prediction accuracy"""
        if not predictions:
            return 0.0
        
        correct = 0
        for pred in predictions:
            if pred['signal'].type != SignalType.HOLD:
                # Compare prediction direction with actual price movement
                # This is a simplified version - you might want to implement
                # a more sophisticated accuracy calculation
                if (pred['prediction'] > 0 and pred['signal'].type == SignalType.BUY) or \
                   (pred['prediction'] < 0 and pred['signal'].type == SignalType.SELL):
                    correct += 1
        
        return correct / len(predictions) if predictions else 0.0 