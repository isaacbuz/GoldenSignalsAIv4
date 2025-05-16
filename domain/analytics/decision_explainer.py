"""
decision_explainer.py
Purpose: Provides utilities for explaining model and agent trading decisions in GoldenSignalsAI. Used for transparency, debugging, and user-facing explanations of automated actions.
"""

class DecisionExplainer:
    """
    A class used to explain trading decisions made by models and agents in GoldenSignalsAI.
    """
    def explain(self, decision):
        """
        Returns a dictionary containing a human-readable explanation of a trading decision.
        
        Parameters:
        decision (object): A trading decision object containing symbol, action, confidence, and rationale.
        
        Returns:
        dict: A dictionary containing the symbol, action, confidence, and rationale of the trading decision.
        """
        return {
            "symbol": decision.symbol,
            "action": decision.action.name,
            "confidence": decision.confidence,
            "rationale": decision.rationale
        }
