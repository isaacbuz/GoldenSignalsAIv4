# optimization/performance_tracker.py
# Purpose: Tracks performance metrics for agents and strategies, enabling continuous
# improvement of options trading strategies by logging profits, drawdowns, and other metrics.

import logging
from typing import Dict

import pandas as pd

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks performance metrics for agents and strategies."""

    def __init__(self):
        """Initialize with an empty metrics store."""
        self.metrics = pd.DataFrame(
            columns=["agent_type", "profit", "drawdown", "sharpe_ratio", "timestamp"]
        )
        logger.info({"message": "PerformanceTracker initialized"})

    def log_performance(self, agent: Dict, trade_result: Dict):
        """Log performance metrics for an agent's trade.

        Args:
            agent (Dict): Agent details (e.g., {'type': 'breakout'}).
            trade_result (Dict): Trade result with 'profit', 'drawdown', 'sharpe_ratio'.
        """
        logger.info({"message": f"Logging performance for agent: {agent}"})
        try:
            new_record = pd.DataFrame(
                [
                    {
                        "agent_type": agent.get("type", "unknown"),
                        "profit": trade_result.get("profit", 0.0),
                        "drawdown": trade_result.get("drawdown", 0.0),
                        "sharpe_ratio": trade_result.get("sharpe_ratio", 0.0),
                        "timestamp": pd.Timestamp.now(tz='UTC'),
                    }
                ]
            )
            self.metrics = pd.concat([self.metrics, new_record], ignore_index=True)
            logger.debug({"message": "Performance logged successfully"})
        except Exception as e:
            logger.error({"message": f"Failed to log performance: {str(e)}"})

    def get_metrics(self, agent_type: str) -> pd.DataFrame:
        """Retrieve performance metrics for a specific agent type.

        Args:
            agent_type (str): Type of agent (e.g., 'breakout').

        Returns:
            pd.DataFrame: Metrics for the specified agent.
        """
        logger.debug({"message": f"Retrieving metrics for agent_type: {agent_type}"})
        try:
            metrics = self.metrics[self.metrics["agent_type"] == agent_type]
            logger.debug(
                {"message": f"Retrieved {len(metrics)} metrics for {agent_type}"}
            )
            return metrics
        except Exception as e:
            logger.error({"message": f"Failed to retrieve metrics: {str(e)}"})
            return pd.DataFrame()

    def analyze_performance(self, agent_type: str) -> Dict:
        """Analyze performance metrics for an agent type.

        Args:
            agent_type (str): Type of agent to analyze.

        Returns:
            Dict: Performance analysis (e.g., average profit, max drawdown).
        """
        logger.info({"message": f"Analyzing performance for agent_type: {agent_type}"})
        try:
            metrics = self.get_metrics(agent_type)
            if metrics.empty:
                logger.warning({"message": f"No metrics available for {agent_type}"})
                return {
                    "average_profit": 0.0,
                    "max_drawdown": 0.0,
                    "average_sharpe": 0.0,
                }

            analysis = {
                "average_profit": metrics["profit"].mean(),
                "max_drawdown": metrics["drawdown"].max(),
                "average_sharpe": metrics["sharpe_ratio"].mean(),
            }
            logger.info({"message": f"Performance analysis: {analysis}"})
            return analysis
        except Exception as e:
            logger.error({"message": f"Failed to analyze performance: {str(e)}"})
            return {"average_profit": 0.0, "max_drawdown": 0.0, "average_sharpe": 0.0}
