from typing import Dict, Any

class StrategySelector:
    def __init__(self, regime_map: Dict[str, str] = None):
        # Map regime features to strategy names
        self.regime_map = regime_map or {
            "bull": "breakout_agent",
            "bear": "reversal_agent",
            "sideways": "mean_reversion_agent"
        }

    def predict(self, regime_features: Dict[str, Any]) -> str:
        # Example: regime_features = {"regime": "bull"}
        regime = regime_features.get("regime", "sideways")
        return self.regime_map.get(regime, "mean_reversion_agent")
