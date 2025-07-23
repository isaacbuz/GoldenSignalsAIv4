"""
Sentiment Aggregator Agent that combines signals from multiple sentiment sources.
"""
import asyncio
from typing import Dict, Any, List
import logging
from agents.base import BaseAgent

logger = logging.getLogger(__name__)

class SentimentAggregatorAgent(BaseAgent):
    """Agent that aggregates sentiment signals from multiple sources."""

    def __init__(self, sources: List[BaseAgent] = None):
        super().__init__(name="SentimentAggregator", agent_type="meta")
        self.sources = sources or []

    def add_source(self, source: BaseAgent) -> None:
        """Add a new sentiment source to the aggregator."""
        if source not in self.sources:
            self.sources.append(source)

    async def gather_sentiments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather sentiment from all sources asynchronously."""
        tasks = []
        for source in self.sources:
            if asyncio.iscoroutinefunction(source.process):
                tasks.append(source.process(data))
            else:
                tasks.append(asyncio.to_thread(source.process, data))
        return await asyncio.gather(*tasks, return_exceptions=True)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and aggregate sentiment from all sources."""
        if not self.sources:
            logger.warning("No sentiment sources configured")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "No sentiment sources configured"}
            }

        # Gather sentiments from all sources
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.gather_sentiments(data))

        # Filter out errors and calculate weighted average
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Source error: {result}")
                continue
            if result.get("action") != "hold":
                valid_results.append(result)

        if not valid_results:
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "No valid sentiment signals"}
            }

        # Calculate weighted action
        buy_weight = sum(r["confidence"] for r in valid_results if r["action"] == "buy")
        sell_weight = sum(r["confidence"] for r in valid_results if r["action"] == "sell")

        # Determine final action
        if buy_weight > sell_weight and buy_weight > 0.5:
            action = "buy"
            confidence = buy_weight / len(valid_results)
        elif sell_weight > buy_weight and sell_weight > 0.5:
            action = "sell"
            confidence = sell_weight / len(valid_results)
        else:
            action = "hold"
            confidence = 0.0

        return {
            "action": action,
            "confidence": confidence,
            "metadata": {
                "sources": len(self.sources),
                "valid_signals": len(valid_results),
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "source_results": [
                    {
                        "action": r["action"],
                        "confidence": r["confidence"],
                        "source": r.get("metadata", {}).get("source", "unknown")
                    }
                    for r in valid_results
                ]
            }
        }
