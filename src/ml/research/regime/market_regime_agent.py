"""
Market regime detection agent using statistical analysis.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from agents.base.base_agent import BaseAgent
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

class MarketRegimeAgent(BaseAgent):
    """Agent that detects market regimes using statistical methods."""
    
    def __init__(
        self,
        name: str = "MarketRegime",
        n_regimes: int = 3,
        lookback_period: int = 60,
        min_samples: int = 30,
        features: List[str] = None
    ):
        """
        Initialize market regime agent.
        
        Args:
            name: Agent name
            n_regimes: Number of market regimes to detect
            lookback_period: Period for regime detection
            min_samples: Minimum samples needed for detection
            features: List of features to use for regime detection
        """
        super().__init__(name=name, agent_type="predictive")
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.min_samples = min_samples
        self.features = features or ["returns", "volatility", "volume_change"]
        self.model = None
        
    def calculate_features(self, data: Dict[str, List[float]]) -> pd.DataFrame:
        """Calculate features for regime detection."""
        try:
            prices = pd.Series(data.get("close_prices", []))
            volumes = pd.Series(data.get("volumes", []))
            
            if len(prices) < 2:
                return pd.DataFrame()
                
            features_dict = {
                "returns": prices.pct_change(),
                "volatility": prices.pct_change().rolling(window=20).std(),
                "volume_change": volumes.pct_change()
            }
            
            return pd.DataFrame(features_dict).fillna(0)
            
        except Exception as e:
            logger.error(f"Feature calculation failed: {str(e)}")
            return pd.DataFrame()
            
    def fit_regime_model(self, features: pd.DataFrame) -> Optional[GaussianMixture]:
        """Fit Gaussian Mixture Model for regime detection."""
        try:
            if len(features) < self.min_samples:
                return None
                
            # Standardize features
            features_std = (features - features.mean()) / features.std()
            
            # Fit GMM
            model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=42
            )
            model.fit(features_std.values)
            
            return model
            
        except Exception as e:
            logger.error(f"Regime model fitting failed: {str(e)}")
            return None
            
    def classify_regime(
        self,
        features: pd.DataFrame,
        model: GaussianMixture
    ) -> Dict[str, Any]:
        """Classify current market regime."""
        try:
            if features.empty or model is None:
                return {"regime": -1, "probability": 0.0}
                
            # Standardize features
            features_std = (features - features.mean()) / features.std()
            
            # Get regime probabilities
            probs = model.predict_proba(features_std.values)
            regime = model.predict(features_std.values)
            
            # Get current regime and probability
            current_regime = int(regime[-1])
            current_prob = float(probs[-1, current_regime])
            
            return {
                "regime": current_regime,
                "probability": current_prob
            }
            
        except Exception as e:
            logger.error(f"Regime classification failed: {str(e)}")
            return {"regime": -1, "probability": 0.0}
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and detect current regime."""
        try:
            # Calculate features
            features = self.calculate_features(data)
            
            if features.empty:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Insufficient data for regime detection"
                    }
                }
            
            # Fit or update model
            if self.model is None:
                self.model = self.fit_regime_model(features)
                
            if self.model is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Failed to fit regime model"
                    }
                }
            
            # Classify current regime
            regime_info = self.classify_regime(features, self.model)
            
            # Generate trading signal based on regime
            if regime_info["regime"] == -1:
                action = "hold"
                confidence = 0.0
            else:
                # Map regime to action based on characteristics
                regime_actions = {
                    0: "hold",    # Low volatility regime
                    1: "buy",     # Bullish regime
                    2: "sell"     # Bearish regime
                }
                action = regime_actions.get(regime_info["regime"], "hold")
                confidence = regime_info["probability"]
            
            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "regime": regime_info["regime"],
                    "probability": regime_info["probability"],
                    "n_regimes": self.n_regimes,
                    "lookback_period": self.lookback_period,
                    "features": self.features
                }
            }
            
        except Exception as e:
            logger.error(f"Regime detection failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 