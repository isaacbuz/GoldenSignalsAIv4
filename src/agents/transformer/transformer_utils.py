import torch
import numpy as np
from typing import List, Tuple, Optional
from src.agents.common.models import MarketData

def get_transformer_device() -> torch.device:
    """Get the appropriate device for transformer model."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_transformer_input(market_data: List[MarketData]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Prepare input data for transformer model.
    
    Args:
        market_data: List of market data points
        
    Returns:
        Tuple of (X, y) where:
            X: Input features array of shape [batch_size, seq_len, input_dim]
            y: Target values array of shape [batch_size, 2] (price, confidence)
    """
    try:
        if not market_data:
            return None, None
            
        # Extract features
        features = []
        targets = []
        
        for data in market_data:
            # Create feature vector
            feature_vector = [
                data.open,
                data.high,
                data.low,
                data.close,
                data.volume,
                data.vwap,
                data.trades,
                data.timestamp.timestamp(),
                data.interval_seconds,
                data.symbol_hash  # Using symbol hash as a feature
            ]
            features.append(feature_vector)
            
            # Create target vector (price and confidence)
            target_vector = [data.close, 1.0]  # Using 1.0 as default confidence
            targets.append(target_vector)
            
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        return X, y
        
    except Exception as e:
        print(f"Error preparing transformer input: {str(e)}")
        return None, None

def calculate_attention_weights(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Calculate attention weights for the input sequence.
    
    Args:
        model: Transformer model
        x: Input tensor of shape [batch_size, seq_len, input_dim]
        
    Returns:
        Attention weights array
    """
    try:
        model.eval()
        with torch.no_grad():
            # Get attention weights from the first layer
            encoder_output = model.transformer_encoder.layers[0].self_attn(
                x, x, x, need_weights=True
            )[1]
            
            # Convert to numpy and take mean across heads
            attention_weights = encoder_output.cpu().numpy().mean(axis=1)
            
            return attention_weights
            
    except Exception as e:
        print(f"Error calculating attention weights: {str(e)}")
        return np.zeros((x.size(1), x.size(1))) 