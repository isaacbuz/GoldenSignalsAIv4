"""
Machine learning enhanced signal generation agent.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from agents.base.base_agent import BaseAgent
from agents.common.utils.validation import validate_market_data

class EnhancedSignalAgent(BaseAgent):
    """Agent that combines technical analysis with machine learning for signal generation."""
    
    def __init__(
        self,
        name: str = "EnhancedSignal",
        lookback_period: int = 20,
        feature_windows: List[int] = [5, 10, 20],
        model_path: Optional[str] = None
    ):
        """
        Initialize enhanced signal agent.
        
        Args:
            name: Agent name
            lookback_period: Period for feature calculation
            feature_windows: Windows for technical indicators
            model_path: Path to pre-trained model
        """
        super().__init__(name=name, agent_type="ml")
        self.lookback_period = lookback_period
        self.feature_windows = feature_windows
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_model(model_path)
            
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for ML model."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        for window in self.feature_windows:
            # Returns
            features[f'return_{window}'] = data['close'].pct_change(window)
            
            # Volatility
            features[f'volatility_{window}'] = data['close'].pct_change().rolling(window).std()
            
            # Price momentum
            features[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1
            
            # Moving averages
            features[f'ma_{window}'] = data['close'].rolling(window).mean()
            features[f'ma_ratio_{window}'] = data['close'] / features[f'ma_{window}']
            
        # Volume features
        features['volume_ma'] = data['volume'].rolling(10).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma']
        
        # Volatility features
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        return features.fillna(0)
        
    def train(self, data: pd.DataFrame, labels: pd.Series):
        """Train the ML model."""
        features = self.calculate_features(data)
        X = self.scaler.fit_transform(features)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X, labels)
        
    def save_model(self, path: str):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
        
    def load_model(self, path: str):
        """Load trained model and scaler."""
        try:
            saved = joblib.load(path)
            self.model = saved['model']
            self.scaler = saved['scaler']
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals using ML model."""
        try:
            if not validate_market_data(data):
                raise ValueError("Invalid market data format")
                
            if self.model is None:
                raise ValueError("Model not trained or loaded")
                
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Calculate features
            features = self.calculate_features(df)
            X = self.scaler.transform(features.iloc[[-1]])
            
            # Get model prediction and probability
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Map prediction to action
            action_map = {
                1: "buy",
                0: "hold",
                -1: "sell"
            }
            action = action_map.get(prediction, "hold")
            
            # Use highest probability as confidence
            confidence = float(max(probabilities))
            
            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "probabilities": probabilities.tolist(),
                    "features": features.iloc[-1].to_dict()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 