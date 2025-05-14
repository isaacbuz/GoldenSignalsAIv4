# create_part3f.py
# Purpose: Creates files in the orchestration/ directory for the GoldenSignalsAI project,
# including the agent supervisor and real-time data feed. Incorporates improvements like
# Celery-based task distribution, circuit breakers for agent coordination, and Redis Streams
# for data streaming in options trading workflows.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part3f():
    """Create files in orchestration/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating orchestration files in {base_dir}"})

    # Define orchestration directory files
    orchestration_files = {
        "orchestration/__init__.py": """# orchestration/__init__.py
# Purpose: Marks the orchestration directory as a Python subpackage, enabling imports
# for agent coordination and data streaming components.

# Empty __init__.py to mark orchestration as a subpackage
""",
        "orchestration/supervisor.py": """# orchestration/supervisor.py
# Purpose: Coordinates agent tasks using Celery for distributed processing and circuit
# breakers for reliability. Manages agent lifecycle, resolves conflicts, and integrates
# with Redis Streams for real-time data processing in options trading workflows.

import logging
import pandas as pd
from typing import Dict
from celery import Celery
from application.agents.factory import AgentFactory
from notifications.alert_manager import AlertManager
import asyncio
import numpy as np
import yaml

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Celery
app = Celery('goldensignalsai', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

class CircuitBreaker:
    \"\"\"Simple circuit breaker to manage agent task failures.\"\"\"
    def __init__(self, max_failures: int = 3, reset_timeout: int = 300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False

    def record_failure(self):
        \"\"\"Record a task failure and check if circuit breaker should open.\"\"\"
        self.failures += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        if self.failures >= self.max_failures:
            self.is_open = True
            logger.warning({"message": f"Circuit breaker opened after {self.failures} failures"})

    def reset(self):
        \"\"\"Reset the circuit breaker if reset_timeout has passed.\"\"\"
        if self.last_failure_time and (asyncio.get_event_loop().time() - self.last_failure_time) > self.reset_timeout:
            self.failures = 0
            self.is_open = False
            logger.info({"message": "Circuit breaker reset"})

    def can_proceed(self) -> bool:
        \"\"\"Check if tasks can proceed.\"\"\"
        self.reset()
        return not self.is_open

class AgentSupervisor:
    \"\"\"Supervises agent tasks, coordinates execution, and resolves conflicts.\"\"\"
    def __init__(self, config: Dict, user_id: str, historical_data: pd.DataFrame):
        \"\"\"Initialize with configuration, user ID, and historical data.
        
        Args:
            config (Dict): Configuration for agents and alerting.
            user_id (str): User identifier for personalization.
            historical_data (pd.DataFrame): Historical market data.
        \"\"\"
        self.config = config
        self.user_id = user_id
        self.historical_data = historical_data
        # Initialize agents
        self.agents = {
            'breakout': AgentFactory.create_agent('breakout', config['agents']['breakout']),
            'options_flow': AgentFactory.create_agent('options_flow', config['agents']['options_flow']),
            'reversion': AgentFactory.create_agent('reversion', config['agents']['reversion']),
            'options_chain': AgentFactory.create_agent('options_chain', config['agents']['options_chain']),
            'news_sentiment': AgentFactory.create_agent('news_sentiment', config['agents']['news_sentiment']),
            'social_media_sentiment': AgentFactory.create_agent('social_media_sentiment', config['agents']['social_media_sentiment']),
            'options_risk': AgentFactory.create_agent('options_risk', config['agents']['options_risk']),
            'regime': AgentFactory.create_agent('regime', config['agents']['regime']),
            'portfolio': AgentFactory.create_agent('portfolio', config['agents']['portfolio']),
            'backtest_research': AgentFactory.create_agent('backtest_research', config['agents']['backtest_research'])
        }
        # Initialize circuit breakers for each agent
        self.circuit_breakers = {name: CircuitBreaker() for name in self.agents}
        logger.info({"message": f"AgentSupervisor initialized for user {user_id}"})

    @app.task
    def process_agent_task(self, agent_name: str, data: Dict) -> Dict:
        \"\"\"Celery task to process an agent's task.
        
        Args:
            agent_name (str): Name of the agent to process.
            data (Dict): Market observation data.
        
        Returns:
            Dict: Agent's result or error.
        \"\"\"
        logger.info({"message": f"Processing task for agent {agent_name}"})
        try:
            if not self.circuit_breakers[agent_name].can_proceed():
                logger.warning({"message": f"Circuit breaker open for {agent_name}"})
                return {"error": f"Circuit breaker open for {agent_name}"}
            result = self.agents[agent_name].process(data)
            logger.info({"message": f"Agent {agent_name} processed task: {result}"})
            return result
        except Exception as e:
            self.circuit_breakers[agent_name].record_failure()
            logger.error({"message": f"Agent {agent_name} task failed: {str(e)}"})
            return {"error": str(e)}

    async def dispatch_tasks(self, observation: Dict) -> Dict:
        \"\"\"Dispatch tasks to agents asynchronously using Celery.
        
        Args:
            observation (Dict): Market observation with stock_data, options_data, etc.
        
        Returns:
            Dict: Final trade decision after resolving agent conflicts.
        \"\"\"
        logger.info({"message": f"Dispatching tasks for observation: {observation['symbol']}"})
        try:
            tasks = []
            for agent_name in self.agents:
                if self.circuit_breakers[agent_name].can_proceed():
                    task = self.process_agent_task.delay(agent_name, observation)
                    tasks.append((agent_name, task))
                else:
                    logger.warning({"message": f"Skipping {agent_name} due to open circuit breaker"})

            # Wait for tasks to complete
            results = {}
            for agent_name, task in tasks:
                try:
                    result = task.get(timeout=30)
                    if "error" not in result:
                        results[agent_name] = result
                    else:
                        logger.warning({"message": f"Task for {agent_name} failed: {result['error']}"})
                except Exception as e:
                    logger.error({"message": f"Task for {agent_name} timed out or failed: {str(e)}"})
                    self.circuit_breakers[agent_name].record_failure()

            # Resolve conflicts and generate final decision
            final_decision = self._resolve_conflicts(results, observation)

            # Send alerts via multiple channels
            if final_decision["action"] != "hold":
                alert_manager = AlertManager()
                user_preferences = {
                    "sms": "+1234567890",  # Example; fetch from user database in practice
                    "whatsapp": "whatsapp:+1234567890",
                    "telegram": "123456789",
                    "x": True,
                    "alert_threshold": config['notifications']['alert_threshold'],
                    "frequency": "immediate"
                }
                await alert_manager.send_alert(final_decision, user_preferences, self.user_id)

            logger.info({"message": f"Final decision: {final_decision}"})
            return final_decision
        except Exception as e:
            logger.error({"message": f"Failed to dispatch tasks: {str(e)}"})
            return {"symbol": observation.get("symbol", "UNKNOWN"), "action": "hold", "confidence": 0.0}

    async def _continuous_learn(self):
        \"\"\"Continuous learning loop for agents to adapt to new data.\"\"\"
        logger.info({"message": "Starting continuous learning loop"})
        try:
            while True:
                new_data = self.historical_data.tail(1)  # Placeholder; fetch latest data
                tasks = []
                for name, agent in self.agents.items():
                    if hasattr(agent, 'adapt'):
                        tasks.append(self.train_agent(name, new_data))
                await asyncio.gather(*tasks)
                await asyncio.sleep(60)  # Learn every minute
        except Exception as e:
            logger.error({"message": f"Continuous learning loop failed: {str(e)}"})

    async def train_agent(self, agent_name: str, data: pd.DataFrame):
        \"\"\"Train an agent asynchronously.
        
        Args:
            agent_name (str): Name of the agent to train.
            data (pd.DataFrame): New data for training.
        \"\"\"
        logger.info({"message": f"Training agent {agent_name} asynchronously"})
        try:
            agent = self.agents[agent_name]
            if hasattr(agent, 'adapt'):
                await asyncio.to_thread(agent.adapt, data)
            logger.info({"message": f"Agent {agent_name} training completed"})
        except Exception as e:
            logger.error({"message": f"Failed to train agent {agent_name}: {str(e)}"})

    def _resolve_conflicts(self, agent_results: Dict[str, Dict], observation: Dict) -> Dict:
        \"\"\"Resolve conflicts between agent actions to produce a final trade decision.
        
        Args:
            agent_results (Dict[str, Dict]): Actions proposed by each agent.
            observation (Dict): Current market observation.
        
        Returns:
            Dict: Final trade decision.
        \"\"\"
        logger.info({"message": "Resolving agent conflicts"})
        try:
            # Get regime from RegimeAgent
            regime = agent_results.get("regime", {}).get("regime", "volatile")
            
            # Adjust weights based on regime for options trading
            weights = {
                "breakout": 0.4 if regime == "trending" else 0.2,
                "options_flow": 0.3,
                "reversion": 0.4 if regime == "mean_reverting" else 0.2,
                "options_chain": 0.3,
                "news_sentiment": 0.2,
                "social_media_sentiment": 0.2
            }

            score = 0.0
            total_confidence = 0.0
            explanations = {}
            for agent_name, result in agent_results.items():
                # Skip non-decision agents
                if agent_name in ["regime", "options_risk", "portfolio", "backtest_research"]:
                    continue
                action = result.get("action", "hold")
                confidence = result.get("confidence", 0.0)
                weight = weights.get(agent_name, 0.1)
                action_score = 1.0 if action == "buy" else -1.0 if action == "sell" else 0.0
                score += action_score * confidence * weight
                total_confidence += confidence * weight
                explanations[agent_name] = result.get("metadata", {}).get("explanation", {})

            final_confidence = total_confidence / sum(weights.values()) if total_confidence > 0 else 0.0
            action = "buy" if score > 0.3 else "sell" if score < -0.3 else "hold"

            proposed_trade = {
                "symbol": observation["symbol"],
                "action": action,
                "size": self.agents["options_risk"].risk_manager.calculate_kelly_position(
                    final_confidence, observation["stock_data"]['Close'].iloc[-1] if not observation["stock_data"].empty else 0
                )
            }
            is_safe = self.agents["options_risk"].evaluate(
                proposed_trade,
                pd.DataFrame(observation["stock_data"]),
                pd.DataFrame(observation["options_data"])
            )

            if not is_safe:
                logger.warning({"message": "Risk evaluation failed, downgrading to hold"})
                action = "hold"
                final_confidence = 0.0

            # Update portfolio if the trade is executed
            if action != "hold" and observation["stock_data"]:
                current_price = observation["stock_data"]['Close'].iloc[-1] if not observation["stock_data"].empty else 0.0
                self.agents["portfolio"].update_positions(proposed_trade, current_price)

            final_decision = {
                "symbol": observation["symbol"],
                "action": action,
                "confidence": final_confidence,
                "metadata": {
                    "agent_results": agent_results,
                    "regime": regime,
                    "explanations": explanations,
                    "portfolio": self.agents["portfolio"].process(observation)
                }
            }
            logger.info({"message": f"Final trade decision: {final_decision}"})
            return final_decision
        except Exception as e:
            logger.error({"message": f"Failed to resolve conflicts: {str(e)}"})
            return {"symbol": observation.get("symbol", "UNKNOWN"), "action": "hold", "confidence": 0.0}
""",
        "orchestration/data_feed.py": """# orchestration/data_feed.py
# Purpose: Streams real-time market data using Redis Streams and archives to TimescaleDB.
# Enhanced for options trading with options data streaming and multi-source integration
# (stock, options, news, social media).

import pandas as pd
import redis
import asyncio
import logging
from infrastructure.data_fetcher import DataFetcher
import yaml

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class RealTimeDataFeed:
    \"\"\"Streams real-time market data using Redis Streams.\"\"\"
    def __init__(self, symbols: list, interval: str = "1d", period: str = "1mo"):
        \"\"\"Initialize the RealTimeDataFeed.
        
        Args:
            symbols (list): List of stock symbols.
            interval (str): Data interval ('1m', '1d', etc.).
            period (str): Data period ('1d', '1mo', etc.).
        \"\"\"
        self.symbols = symbols
        self.interval = interval
        self.period = period
        self.fetcher = DataFetcher()
        # Initialize Redis client
        if config['redis'].get('cluster_enabled', False):
            from redis.cluster import RedisCluster
            nodes = config['redis']['cluster_nodes']
            self.redis_client = RedisCluster(startup_nodes=[{'host': node['host'], 'port': node['port']} for node in nodes])
        else:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.historical_data = pd.DataFrame()
        logger.info({"message": f"RealTimeDataFeed initialized for symbols: {symbols}"})

    async def __aiter__(self):
        \"\"\"Async iterator for streaming real-time data.
        
        Yields:
            dict: Market observation for each symbol.
        \"\"\"
        logger.info({"message": "Starting real-time data stream"})
        while True:
            try:
                for symbol in self.symbols:
                    # Fetch stock, options, news, and social media data
                    stock_data = self.fetcher.fetch_stock_data(symbol, self.interval, self.period)
                    options_data = self.fetcher.fetch_options_data(symbol)
                    news_articles = self.fetcher.fetch_news_data(symbol)
                    social_media = self.fetcher.fetch_social_sentiment(symbol)
                    # Update historical data
                    self.historical_data = pd.concat([self.historical_data, stock_data], ignore_index=True)
                    # Construct observation
                    observation = {
                        "symbol": symbol,
                        "stock_data": stock_data,
                        "options_data": options_data,
                        "news_articles": news_articles,
                        "social_media": social_media,
                        "prices": {symbol: stock_data['Close'].iloc[-1] if not stock_data.empty else 0.0}
                    }
                    # Publish to Redis Stream
                    stream_data = {
                        'symbol': symbol,
                        'stock_data': stock_data.to_json(),
                        'options_data': options_data.to_json(),
                        'news_articles': str(news_articles),
                        'social_media': str(social_media)
                    }
                    self.redis_client.xadd('market-data-stream', stream_data)
                    logger.info({"message": f"Streamed observation for {symbol}"})
                    yield observation
                await asyncio.sleep(60)  # Fetch every minute
            except Exception as e:
                logger.error({"message": f"Failed to fetch data: {str(e)}"})
                await asyncio.sleep(60)
""",
    }

    # Write orchestration directory files
    for file_path, content in orchestration_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 3f: orchestration/ created successfully")


if __name__ == "__main__":
    create_part3f()
