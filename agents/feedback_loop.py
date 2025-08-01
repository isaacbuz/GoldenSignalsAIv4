"""
feedback_loop.py
Collects user or trading outcome feedback to refine models and strategies.
"""
from typing import Any, Dict, List


class FeedbackLoop:
    def __init__(self):
        self.feedback_log: List[Dict[str, Any]] = []

    def add_feedback(self, signal: Dict[str, Any], outcome: Any, user_rating: float = None):
        entry = {
            'signal': signal,
            'outcome': outcome,
            'user_rating': user_rating
        }
        self.feedback_log.append(entry)

    def get_feedback(self) -> List[Dict[str, Any]]:
        return self.feedback_log

    def use_feedback_for_training(self, agent):
        # Placeholder: Use feedback to retrain or calibrate agent
        pass
