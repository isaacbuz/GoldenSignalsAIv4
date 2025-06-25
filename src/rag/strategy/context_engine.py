"""Strategy Performance Context Engine"""

class StrategyContextEngine:
    def __init__(self):
        self.strategies = {}
        
    async def get_strategy_context(self, strategy_id):
        """Get performance context for strategy"""
        return {"performance": 0.0, "context": {}}
