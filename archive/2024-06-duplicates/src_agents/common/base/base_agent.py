import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from agents.common.models.market_data import MarketData
from agents.common.models.signal import Signal
from agents.common.models import Prediction

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all trading agents."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    def train(self, market_data: List[MarketData]) -> bool:
        """Train the agent on historical market data.
        
        Args:
            market_data: List of historical market data points
            
        Returns:
            bool: True if training completed successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def predict(self, market_data: MarketData) -> Optional[Prediction]:
        """Generate predictions using the agent.
        
        Args:
            market_data: Current market data point
            
        Returns:
            Optional[Prediction]: Prediction object if successful, None otherwise
        """
        pass
        
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Generate trading signals based on predictions.
        
        Args:
            market_data: Current market data point
            
        Returns:
            Optional[Signal]: Trading signal if generated, None otherwise
        """
        pass
        
    def load_model(self, model_path: str) -> bool:
        """Load a trained model.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        raise NotImplementedError("load_model not implemented")
        
    def save_model(self, model_path: str) -> bool:
        """Save the current model.
        
        Args:
            model_path: Path to save the model file
            
        Returns:
            bool: True if model saved successfully, False otherwise
        """
        raise NotImplementedError("save_model not implemented")
    
    def log_signal(self, signal: Signal) -> None:
        """Log generated signal."""
        self.logger.info(
            f"Generated {signal.type.value} signal with {signal.strength.value} strength "
            f"and {signal.confidence:.2f} confidence"
        ) 