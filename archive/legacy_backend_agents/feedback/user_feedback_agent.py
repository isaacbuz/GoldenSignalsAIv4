from archive.legacy_backend_agents.base import BaseSignalAgent

class UserFeedbackAgent(BaseSignalAgent):
    """Agent that incorporates user feedback into signal generation."""
    def run(self) -> dict:
        # Placeholder: In reality, aggregate user ratings
        return {
            "symbol": self.symbol,
            "action": "buy",
            "confidence": 55,
            "agent": "UserFeedbackAgent",
            "explanation": "Recent user feedback is bullish."
        }
