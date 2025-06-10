import pytest

# Dummy service for illustration; replace with actual imports
class RiskManager:
    def assess(self, risk):
        return 'low' if risk < 0.5 else 'high'

def test_risk_manager():
    rm = RiskManager()
    assert rm.assess(0.3) == 'low'
    assert rm.assess(0.7) == 'high'
