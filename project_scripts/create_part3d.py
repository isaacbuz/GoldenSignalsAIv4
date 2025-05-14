# create_part3d.py
# Purpose: Creates files in the agents/ directory for the GoldenSignalsAI project,
# including agent factory and various predictive, sentiment, and risk agents. Incorporates
# improvements like options-specific agents (OptionsFlowAgent, OptionsChainAgent, OptionsRiskAgent).

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part3d():
    """Create files in agents/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating agents files in {base_dir}"})

    # Define agents directory files
    agents_files = {
        "agents/__init__.py": """# agents/__init__.py
# Purpose: Marks the agents directory as a Python package, enabling imports
# for agent factory and various agent implementations.
""",
        "agents/factory.py": """# agents/factory.py
# Purpose: Implements a factory pattern for creating agent instances used in GoldenSignalsAI,
# supporting predictive, sentiment, risk, and portfolio agents, with options trading enhancements.

from abc import ABC, abstractmethod
import logging
from typing import Dict
from .predictive.breakout import BreakoutAgent
from .predictive.options_flow import OptionsFlowAgent
from .predictive.reversion import ReversionAgent
from .predictive.options_chain import OptionsChainAgent
from .predictive.regime import RegimeAgent
from .sentiment.news import NewsSentimentAgent
from .sentiment.social_media import SocialMediaSentimentAgent
from .risk.options_risk import OptionsRiskAgent
from .portfolio.portfolio import PortfolioAgent
from .backtest_research.backtest_research import BacktestResearchAgent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class Agent(ABC):
    \"\"\"Abstract base class for all agents.\"\"\"
    @abstractmethod
    def process(self, data: Dict) -> Dict:
        \"\"\"Process market data and return a decision.
        
        Args:
            data (Dict): Market observation with stock_data, options_data, etc.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        pass

    @abstractmethod
    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new market data.
        
        Args:
            new_data (pd.DataFrame): New market data for learning.
        \"\"\"
        pass

class AgentFactory:
    \"\"\"Factory for creating agent instances.\"\"\"
    _agents = {
        'breakout': BreakoutAgent,
        'options_flow': OptionsFlowAgent,
        'reversion': ReversionAgent,
        'options_chain': OptionsChainAgent,
        'news_sentiment': NewsSentimentAgent,
        'social_media_sentiment': SocialMediaSentimentAgent,
        'options_risk': OptionsRiskAgent,
        'regime': RegimeAgent,
        'portfolio': PortfolioAgent,
        'backtest_research': BacktestResearchAgent
    }

    @staticmethod
    def create_agent(agent_type: str, config: Dict) -> Agent:
        \"\"\"Create an agent instance based on the specified type.
        
        Args:
            agent_type (str): Type of agent to create (e.g., 'breakout').
            config (Dict): Configuration parameters for the agent.
        
        Returns:
            Agent: Instantiated agent object.
        
        Raises:
            ValueError: If the agent type is unknown.
        \"\"\"
        logger.info({"message": f"Creating agent: {agent_type}"})
        agent_class = AgentFactory._agents.get(agent_type)
        if not agent_class:
            logger.error({"message": f"Unknown agent type: {agent_type}"})
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(**config)
""",
        "agents/predictive/__init__.py": """# agents/predictive/__init__.py
# Purpose: Marks the predictive agents directory as a Python subpackage.
""",
        "agents/predictive/breakout.py": """# agents/predictive/breakout.py
# Purpose: Implements a BreakoutAgent that identifies breakout patterns in stock prices,
# suitable for directional options trading strategies.

import pandas as pd
import numpy as np
import logging
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class BreakoutAgent(Agent):
    \"\"\"Agent that identifies breakout patterns in stock prices.\"\"\"
    def __init__(self, window: int = 20, threshold: float = 0.05):
        \"\"\"Initialize the BreakoutAgent.
        
        Args:
            window (int): Lookback period for breakout detection.
            threshold (float): Breakout threshold as a percentage.
        \"\"\"
        self.window = window
        self.threshold = threshold
        logger.info({"message": f"BreakoutAgent initialized with window={window}, threshold={threshold}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process market data to identify breakout patterns.
        
        Args:
            data (Dict): Market observation with 'stock_data', 'options_data', etc.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": f"Processing data for BreakoutAgent"})
        try:
            stock_data = pd.DataFrame(data['stock_data'])
            if stock_data.empty:
                logger.warning({"message": "No stock data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            prices = stock_data['Close']
            high = prices[-self.window:].max()
            low = prices[-self.window:].min()
            current_price = prices.iloc[-1]

            # Detect breakout
            if current_price > high * (1 + self.threshold):
                action = "buy"
                confidence = (current_price - high) / high
            elif current_price < low * (1 - self.threshold):
                action = "sell"
                confidence = (low - current_price) / low
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"high": high, "low": low, "current_price": current_price}
            }
            logger.info({"message": f"BreakoutAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"BreakoutAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new market data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New market data.
        \"\"\"
        logger.info({"message": "BreakoutAgent adapting to new data"})
        try:
            # Placeholder: In a real implementation, adjust window or threshold based on performance
            pass
        except Exception as e:
            logger.error({"message": f"BreakoutAgent adaptation failed: {str(e)}"})
""",
        "agents/predictive/options_flow.py": """# agents/predictive/options_flow.py
# Purpose: Implements an OptionsFlowAgent that analyzes options flow data to detect
# unusual activity, supporting options trading strategies by identifying bullish/bearish signals.

import pandas as pd
import logging
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class OptionsFlowAgent(Agent):
    \"\"\"Agent that analyzes options flow data for trading signals.\"\"\"
    def __init__(self, iv_skew_threshold: float = 0.1):
        \"\"\"Initialize the OptionsFlowAgent.
        
        Args:
            iv_skew_threshold (float): Threshold for detecting unusual IV skew.
        \"\"\"
        self.iv_skew_threshold = iv_skew_threshold
        logger.info({"message": f"OptionsFlowAgent initialized with iv_skew_threshold={iv_skew_threshold}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process options data to detect unusual activity.
        
        Args:
            data (Dict): Market observation with 'options_data'.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": "Processing data for OptionsFlowAgent"})
        try:
            options_data = pd.DataFrame(data['options_data'])
            if options_data.empty:
                logger.warning({"message": "No options data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # Calculate IV skew (simplified)
            call_iv = options_data[options_data['call_put'] == 'call']['iv'].mean()
            put_iv = options_data[options_data['call_put'] == 'put']['iv'].mean()
            iv_skew = call_iv - put_iv

            # Detect bullish/bearish signals based on IV skew
            if iv_skew > self.iv_skew_threshold:
                action = "buy"  # Bullish signal
                confidence = iv_skew / self.iv_skew_threshold
            elif iv_skew < -self.iv_skew_threshold:
                action = "sell"  # Bearish signal
                confidence = abs(iv_skew) / self.iv_skew_threshold
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"iv_skew": iv_skew, "call_iv": call_iv, "put_iv": put_iv}
            }
            logger.info({"message": f"OptionsFlowAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"OptionsFlowAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new options data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New options data.
        \"\"\"
        logger.info({"message": "OptionsFlowAgent adapting to new data"})
        try:
            # Placeholder: Adjust threshold based on historical IV skew trends
            pass
        except Exception as e:
            logger.error({"message": f"OptionsFlowAgent adaptation failed: {str(e)}"})
""",
        "agents/predictive/reversion.py": """# agents/predictive/reversion.py
# Purpose: Implements a ReversionAgent that identifies mean-reversion opportunities,
# suitable for options trading strategies like straddles in mean-reverting markets.

import pandas as pd
import numpy as np
import logging
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class ReversionAgent(Agent):
    \"\"\"Agent that identifies mean-reversion opportunities.\"\"\"
    def __init__(self, mean_reversion_window: int = 20):
        \"\"\"Initialize the ReversionAgent.
        
        Args:
            mean_reversion_window (int): Lookback period for mean-reversion calculation.
        \"\"\"
        self.mean_reversion_window = mean_reversion_window
        logger.info({"message": f"ReversionAgent initialized with mean_reversion_window={mean_reversion_window}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process market data to identify mean-reversion opportunities.
        
        Args:
            data (Dict): Market observation with 'stock_data'.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": "Processing data for ReversionAgent"})
        try:
            stock_data = pd.DataFrame(data['stock_data'])
            if stock_data.empty:
                logger.warning({"message": "No stock data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            prices = stock_data['Close']
            if len(prices) < self.mean_reversion_window:
                logger.warning({"message": f"Insufficient data: {len(prices)} < {self.mean_reversion_window}"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            mean_price = prices[-self.mean_reversion_window:].mean()
            current_price = prices.iloc[-1]
            deviation = (current_price - mean_price) / mean_price

            # Detect mean-reversion opportunity
            if deviation > 0.05:
                action = "sell"  # Overbought
                confidence = deviation
            elif deviation < -0.05:
                action = "buy"  # Oversold
                confidence = abs(deviation)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"deviation": deviation, "mean_price": mean_price}
            }
            logger.info({"message": f"ReversionAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"ReversionAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new market data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New market data.
        \"\"\"
        logger.info({"message": "ReversionAgent adapting to new data"})
        try:
            # Placeholder: Adjust window based on volatility
            pass
        except Exception as e:
            logger.error({"message": f"ReversionAgent adaptation failed: {str(e)}"})
""",
        "agents/predictive/options_chain.py": """# agents/predictive/options_chain.py
# Purpose: Implements an OptionsChainAgent that analyzes options chain data to identify
# significant volume or open interest changes, supporting options trading strategies.

import pandas as pd
import logging
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class OptionsChainAgent(Agent):
    \"\"\"Agent that analyzes options chain data for trading signals.\"\"\"
    def __init__(self, volume_threshold: int = 1000, oi_threshold: int = 5000):
        \"\"\"Initialize the OptionsChainAgent.
        
        Args:
            volume_threshold (int): Threshold for significant options volume.
            oi_threshold (int): Threshold for significant open interest.
        \"\"\"
        self.volume_threshold = volume_threshold
        self.oi_threshold = oi_threshold
        logger.info({"message": f"OptionsChainAgent initialized with volume_threshold={volume_threshold}, oi_threshold={oi_threshold}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process options chain data to identify significant activity.
        
        Args:
            data (Dict): Market observation with 'options_data'.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": "Processing data for OptionsChainAgent"})
        try:
            options_data = pd.DataFrame(data['options_data'])
            if options_data.empty:
                logger.warning({"message": "No options data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # Analyze call and put activity
            call_volume = options_data[options_data['call_put'] == 'call']['call_volume'].sum()
            put_volume = options_data[options_data['call_put'] == 'put']['put_volume'].sum()
            call_oi = options_data[options_data['call_put'] == 'call']['call_oi'].sum()
            put_oi = options_data[options_data['call_put'] == 'put']['put_oi'].sum()

            # Detect significant activity
            if (call_volume > self.volume_threshold or call_oi > self.oi_threshold) and call_volume > put_volume:
                action = "buy"  # Bullish signal
                confidence = max(call_volume / self.volume_threshold, call_oi / self.oi_threshold)
            elif (put_volume > self.volume_threshold or put_oi > self.oi_threshold) and put_volume > call_volume:
                action = "sell"  # Bearish signal
                confidence = max(put_volume / self.volume_threshold, put_oi / self.oi_threshold)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                    "call_oi": call_oi,
                    "put_oi": put_oi
                }
            }
            logger.info({"message": f"OptionsChainAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"OptionsChainAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new options data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New options data.
        \"\"\"
        logger.info({"message": "OptionsChainAgent adapting to new data"})
        try:
            # Placeholder: Adjust thresholds based on historical activity
            pass
        except Exception as e:
            logger.error({"message": f"OptionsChainAgent adaptation failed: {str(e)}"})
""",
        "agents/predictive/regime.py": """# agents/predictive/regime.py
# Purpose: Implements a RegimeAgent that uses the RegimeDetector to identify market regimes,
# helping adjust options trading strategies based on market conditions.

import pandas as pd
import logging
from ..factory import Agent
from domain.trading.regime_detector import RegimeDetector

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class RegimeAgent(Agent):
    \"\"\"Agent that identifies market regimes to adjust trading strategies.\"\"\"
    def __init__(self, regime_window: int = 30):
        \"\"\"Initialize the RegimeAgent.
        
        Args:
            regime_window (int): Lookback period for regime detection.
        \"\"\"
        self.regime_detector = RegimeDetector(window=regime_window)
        logger.info({"message": f"RegimeAgent initialized with regime_window={regime_window}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process market data to identify the current market regime.
        
        Args:
            data (Dict): Market observation with 'stock_data'.
        
        Returns:
            Dict: Decision with 'regime' and metadata.
        \"\"\"
        logger.info({"message": "Processing data for RegimeAgent"})
        try:
            stock_data = pd.DataFrame(data['stock_data'])
            if stock_data.empty:
                logger.warning({"message": "No stock data available"})
                return {"regime": "mean_reverting", "confidence": 0.0, "metadata": {}}

            prices = stock_data['Close']
            regime = self.regime_detector.detect(prices)

            decision = {
                "regime": regime,
                "confidence": 0.8,  # Simplified confidence
                "metadata": {"regime": regime}
            }
            logger.info({"message": f"RegimeAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"RegimeAgent processing failed: {str(e)}"})
            return {"regime": "mean_reverting", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new market data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New market data.
        \"\"\"
        logger.info({"message": "RegimeAgent adapting to new data"})
        try:
            # Placeholder: Adjust regime detection parameters if needed
            pass
        except Exception as e:
            logger.error({"message": f"RegimeAgent adaptation failed: {str(e)}"})
""",
        "agents/sentiment/__init__.py": """# agents/sentiment/__init__.py
# Purpose: Marks the sentiment agents directory as a Python subpackage.
""",
        "agents/sentiment/news.py": """# agents/sentiment/news.py
# Purpose: Implements a NewsSentimentAgent that analyzes news sentiment to generate
# trading signals, useful for options trading during news-driven volatility.

import pandas as pd
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class NewsSentimentAgent(Agent):
    \"\"\"Agent that analyzes news sentiment for trading signals.\"\"\"
    def __init__(self, sentiment_threshold: float = 0.3):
        \"\"\"Initialize the NewsSentimentAgent.
        
        Args:
            sentiment_threshold (float): Threshold for significant sentiment score.
        \"\"\"
        self.sentiment_threshold = sentiment_threshold
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info({"message": f"NewsSentimentAgent initialized with sentiment_threshold={sentiment_threshold}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process news data to analyze sentiment.
        
        Args:
            data (Dict): Market observation with 'news_articles'.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": "Processing data for NewsSentimentAgent"})
        try:
            news_articles = data['news_articles']
            if not news_articles:
                logger.warning({"message": "No news articles available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # Analyze sentiment
            sentiment_scores = [
                self.analyzer.polarity_scores(article.get("description", ""))["compound"]
                for article in news_articles if article.get("description")
            ]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            # Generate trading signal
            if avg_sentiment > self.sentiment_threshold:
                action = "buy"  # Positive sentiment
                confidence = avg_sentiment
            elif avg_sentiment < -self.sentiment_threshold:
                action = "sell"  # Negative sentiment
                confidence = abs(avg_sentiment)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"avg_sentiment": avg_sentiment}
            }
            logger.info({"message": f"NewsSentimentAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"NewsSentimentAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new news data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New news data.
        \"\"\"
        logger.info({"message": "NewsSentimentAgent adapting to new data"})
        try:
            # Placeholder: Adjust threshold based on sentiment trends
            pass
        except Exception as e:
            logger.error({"message": f"NewsSentimentAgent adaptation failed: {str(e)}"})
""",
        "agents/sentiment/social_media.py": """# agents/sentiment/social_media.py
# Purpose: Implements a SocialMediaSentimentAgent that analyzes social media sentiment
# to generate trading signals, useful for options trading during sentiment-driven volatility.

import pandas as pd
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class SocialMediaSentimentAgent(Agent):
    \"\"\"Agent that analyzes social media sentiment for trading signals.\"\"\"
    def __init__(self, sentiment_threshold: float = 0.3):
        \"\"\"Initialize the SocialMediaSentimentAgent.
        
        Args:
            sentiment_threshold (float): Threshold for significant sentiment score.
        \"\"\"
        self.sentiment_threshold = sentiment_threshold
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info({"message": f"SocialMediaSentimentAgent initialized with sentiment_threshold={sentiment_threshold}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process social media data to analyze sentiment.
        
        Args:
            data (Dict): Market observation with 'social_media'.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": "Processing data for SocialMediaSentimentAgent"})
        try:
            social_media = data['social_media']
            if not social_media:
                logger.warning({"message": "No social media data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # Analyze sentiment
            sentiment_scores = [
                self.analyzer.polarity_scores(post.get("text", ""))["compound"]
                for post in social_media if post.get("text")
            ]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            # Generate trading signal
            if avg_sentiment > self.sentiment_threshold:
                action = "buy"  # Positive sentiment
                confidence = avg_sentiment
            elif avg_sentiment < -self.sentiment_threshold:
                action = "sell"  # Negative sentiment
                confidence = abs(avg_sentiment)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"avg_sentiment": avg_sentiment}
            }
            logger.info({"message": f"SocialMediaSentimentAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"SocialMediaSentimentAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new social media data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New social media data.
        \"\"\"
        logger.info({"message": "SocialMediaSentimentAgent adapting to new data"})
        try:
            # Placeholder: Adjust threshold based on sentiment trends
            pass
        except Exception as e:
            logger.error({"message": f"SocialMediaSentimentAgent adaptation failed: {str(e)}"})
""",
        "agents/risk/__init__.py": """# agents/risk/__init__.py
# Purpose: Marks the risk agents directory as a Python subpackage.
""",
        "agents/risk/options_risk.py": """# agents/risk/options_risk.py
# Purpose: Implements an OptionsRiskAgent that evaluates risks in options trading using Greeks,
# ensuring safe position sizing and risk management for options strategies.

import pandas as pd
import numpy as np
import logging
from ..factory import Agent
from application.services.risk_manager import RiskManager

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class OptionsRiskAgent(Agent):
    \"\"\"Agent that evaluates risks in options trading using Greeks.\"\"\"
    def __init__(self, max_delta: float = 0.7, max_gamma: float = 0.1, max_theta: float = -0.05):
        \"\"\"Initialize the OptionsRiskAgent.
        
        Args:
            max_delta (float): Maximum allowable delta exposure.
            max_gamma (float): Maximum allowable gamma exposure.
            max_theta (float): Maximum allowable theta exposure (negative).
        \"\"\"
        self.max_delta = max_delta
        self.max_gamma = max_gamma
        self.max_theta = max_theta
        self.risk_manager = RiskManager()
        logger.info({"message": f"OptionsRiskAgent initialized with max_delta={max_delta}, max_gamma={max_gamma}, max_theta={max_theta}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process options data to evaluate risk.
        
        Args:
            data (Dict): Market observation with 'options_data'.
        
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        \"\"\"
        logger.info({"message": "Processing data for OptionsRiskAgent"})
        try:
            options_data = pd.DataFrame(data['options_data'])
            if options_data.empty:
                logger.warning({"message": "No options data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # Check Greeks
            delta = options_data['delta'].mean()
            gamma = options_data['gamma'].mean()
            theta = options_data['theta'].mean()

            if delta > self.max_delta or gamma > self.max_gamma or theta < self.max_theta:
                action = "hold"
                confidence = 1.0
                metadata = {
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "reason": "Excessive risk exposure"
                }
            else:
                action = "proceed"
                confidence = 0.8
                metadata = {
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "reason": "Risk within limits"
                }

            decision = {
                "action": action,
                "confidence": confidence,
                "metadata": metadata
            }
            logger.info({"message": f"OptionsRiskAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"OptionsRiskAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def evaluate(self, trade: Dict, stock_data: pd.DataFrame, options_data: pd.DataFrame) -> bool:
        \"\"\"Evaluate if a trade is safe based on risk parameters.
        
        Args:
            trade (Dict): Proposed trade with 'symbol', 'action', 'size'.
            stock_data (pd.DataFrame): Stock data.
            options_data (pd.DataFrame): Options data.
        
        Returns:
            bool: True if trade is safe, False otherwise.
        \"\"\"
        logger.info({"message": f"Evaluating trade for {trade['symbol']}"})
        try:
            decision = self.process({"options_data": options_data})
            if decision["action"] == "hold":
                logger.warning({"message": f"Trade rejected due to risk: {decision['metadata']['reason']}"})
                return False
            return self.risk_manager.evaluate(trade, stock_data, options_data)
        except Exception as e:
            logger.error({"message": f"OptionsRiskAgent evaluation failed: {str(e)}"})
            return False

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new options data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New options data.
        \"\"\"
        logger.info({"message": "OptionsRiskAgent adapting to new data"})
        try:
            # Placeholder: Adjust risk thresholds based on historical Greeks
            pass
        except Exception as e:
            logger.error({"message": f"OptionsRiskAgent adaptation failed: {str(e)}"})
""",
        "agents/portfolio/__init__.py": """# agents/portfolio/__init__.py
# Purpose: Marks the portfolio agents directory as a Python subpackage.
""",
        "agents/portfolio/portfolio.py": """# agents/portfolio/portfolio.py
# Purpose: Implements a PortfolioAgent that manages portfolio positions and adjusts allocations
# based on risk profiles, supporting options trading by maintaining balanced exposures.

import pandas as pd
import logging
from ..factory import Agent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class PortfolioAgent(Agent):
    \"\"\"Agent that manages portfolio positions and allocations.\"\"\"
    def __init__(self, risk_profile: str = "balanced"):
        \"\"\"Initialize the PortfolioAgent.
        
        Args:
            risk_profile (str): Risk profile ('conservative', 'balanced', 'aggressive').
        \"\"\"
        self.risk_profile = risk_profile
        self.positions = {}  # {symbol: {'quantity': int, 'entry_price': float}}
        self.max_position_size = 0.3 if risk_profile == "conservative" else 0.5 if risk_profile == "balanced" else 0.7
        logger.info({"message": f"PortfolioAgent initialized with risk_profile={risk_profile}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process portfolio data to manage positions.
        
        Args:
            data (Dict): Market observation with 'stock_data', 'trade'.
        
        Returns:
            Dict: Decision with portfolio status and metadata.
        \"\"\"
        logger.info({"message": "Processing data for PortfolioAgent"})
        try:
            trade = data.get('trade', {})
            symbol = trade.get('symbol', 'UNKNOWN')
            total_value = sum(
                pos['quantity'] * pos['entry_price']
                for pos in self.positions.values()
            )

            decision = {
                "portfolio_value": total_value,
                "positions": self.positions,
                "metadata": {"risk_profile": self.risk_profile}
            }
            logger.info({"message": f"PortfolioAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"PortfolioAgent processing failed: {str(e)}"})
            return {"portfolio_value": 0.0, "positions": {}, "metadata": {"error": str(e)}}

    def update_positions(self, trade: Dict, current_price: float):
        \"\"\"Update portfolio positions based on a trade.
        
        Args:
            trade (Dict): Trade with 'symbol', 'action', 'size'.
            current_price (float): Current price of the asset.
        \"\"\"
        logger.info({"message": f"Updating portfolio positions for trade: {trade}"})
        try:
            symbol = trade['symbol']
            action = trade['action']
            size = trade['size']

            if action == "buy":
                if symbol in self.positions:
                    existing = self.positions[symbol]
                    avg_price = (existing['quantity'] * existing['entry_price'] + size * current_price) / (existing['quantity'] + size)
                    existing['quantity'] += size
                    existing['entry_price'] = avg_price
                else:
                    self.positions[symbol] = {'quantity': size, 'entry_price': current_price}
            elif action == "sell":
                if symbol in self.positions:
                    existing = self.positions[symbol]
                    existing['quantity'] -= size
                    if existing['quantity'] <= 0:
                        del self.positions[symbol]
            logger.info({"message": f"Updated positions: {self.positions}"})
        except Exception as e:
            logger.error({"message": f"PortfolioAgent position update failed: {str(e)}"})

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new market data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New market data.
        \"\"\"
        logger.info({"message": "PortfolioAgent adapting to new data"})
        try:
            # Placeholder: Adjust risk profile based on market conditions
            pass
        except Exception as e:
            logger.error({"message": f"PortfolioAgent adaptation failed: {str(e)}"})
""",
        "agents/backtest_research/__init__.py": """# agents/backtest_research/__init__.py
# Purpose: Marks the backtest research agents directory as a Python subpackage.
""",
        "agents/backtest_research/backtest_research.py": """# agents/backtest_research/backtest_research.py
# Purpose: Implements a BacktestResearchAgent that runs backtests on multiple strategies
# to identify optimal parameters for options trading.

import pandas as pd
import logging
from ..factory import Agent
from application.services.backtest import Backtester

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class BacktestResearchAgent(Agent):
    \"\"\"Agent that researches optimal trading strategies through backtesting.\"\"\"
    def __init__(self, max_strategies: int = 10):
        \"\"\"Initialize the BacktestResearchAgent.
        
        Args:
            max_strategies (int): Maximum number of strategies to test.
        \"\"\"
        self.max_strategies = max_strategies
        self.results = []
        logger.info({"message": f"BacktestResearchAgent initialized with max_strategies={max_strategies}"})

    def process(self, data: Dict) -> Dict:
        \"\"\"Process historical data to run backtests on multiple strategies.
        
        Args:
            data (Dict): Market observation with 'stock_data', 'signals'.
        
        Returns:
            Dict: Decision with best strategy and metadata.
        \"\"\"
        logger.info({"message": "Processing data for BacktestResearchAgent"})
        try:
            stock_data = pd.DataFrame(data['stock_data'])
            signals = pd.DataFrame(data.get('signals', []))
            if stock_data.empty or signals.empty:
                logger.warning({"message": "No stock data or signals available"})
                return {"best_strategy": None, "performance": {}, "metadata": {}}

            # Run backtest with mock variations (simplified)
            backtester = Backtester(stock_data, signals)
            result = backtester.run(initial_capital=10000)
            if "error" in result:
                logger.error({"message": f"Backtest failed: {result['error']}"})
                return {"best_strategy": None, "performance": {}, "metadata": {"error": result['error']}}

            self.results.append(result)
            if len(self.results) > self.max_strategies:
                self.results.pop(0)

            # Select best strategy (highest Sharpe ratio)
            best_result = max(self.results, key=lambda x: x['sharpe_ratio'])
            decision = {
                "best_strategy": "mock_strategy",  # Placeholder
                "performance": {
                    "sharpe_ratio": best_result['sharpe_ratio'],
                    "total_return": best_result['total_return']
                },
                "metadata": {"num_strategies_tested": len(self.results)}
            }
            logger.info({"message": f"BacktestResearchAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"BacktestResearchAgent processing failed: {str(e)}"})
            return {"best_strategy": None, "performance": {}, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        \"\"\"Adapt the agent to new market data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New market data.
        \"\"\"
        logger.info({"message": "BacktestResearchAgent adapting to new data"})
        try:
            # Placeholder: Adjust strategy parameters based on new data
            pass
        except Exception as e:
            logger.error({"message": f"BacktestResearchAgent adaptation failed: {str(e)}"})
""",
    }

    # Write agents directory files
    for file_path, content in agents_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 3d: agents/ created successfully")


if __name__ == "__main__":
    create_part3d()
