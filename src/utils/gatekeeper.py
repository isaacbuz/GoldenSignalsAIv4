from typing import Optional

from src.models.signals import Signal


class SignalGatekeeper:
    """Simple gatekeeper that decides if a signal is publishable.

    This first cut focuses only on confidence and risk score thresholds so we
    can iterate quickly.  Future versions can inject market regimes, macro
    overrides, smart-money overlays, etc.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_risk_score: float = 0.8,
    ):
        self.min_confidence = min_confidence
        self.max_risk_score = max_risk_score

    def allow(self, signal: Signal) -> bool:
        """Return True if the signal should be broadcast."""
        if signal.confidence < self.min_confidence:
            return False
        if signal.risk_score is not None and signal.risk_score > self.max_risk_score:
            return False
        return True

    def reason(self, signal: Signal) -> Optional[str]:
        if signal.confidence < self.min_confidence:
            return "confidence_below_threshold"
        if signal.risk_score is not None and signal.risk_score > self.max_risk_score:
            return "risk_score_too_high"
        return None 