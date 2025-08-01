"""
Agent Data Bus - Shared Context System
Allows agents to publish and subscribe to shared market insights
"""

import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class AgentDataBus:
    """
    Centralized data sharing system for trading agents

    Features:
    - Publish/Subscribe pattern for agent communication
    - Time-based data expiration
    - Thread-safe operations
    - Context enrichment for agents
    """

    def __init__(self, data_ttl_seconds: int = 300):
        self.data_ttl = data_ttl_seconds
        self._data_store = defaultdict(lambda: defaultdict(dict))
        self._subscribers = defaultdict(list)
        self._lock = threading.RLock()
        self._message_queue = defaultdict(lambda: deque(maxlen=100))

    def publish(self, agent_name: str, symbol: str, data_type: str, data: Any):
        """Publish data from an agent"""
        with self._lock:
            timestamp = datetime.now()

            # Store data with timestamp
            self._data_store[symbol][data_type][agent_name] = {
                'data': data,
                'timestamp': timestamp,
                'agent': agent_name
            }

            # Add to message queue
            message = {
                'agent': agent_name,
                'symbol': symbol,
                'data_type': data_type,
                'data': data,
                'timestamp': timestamp
            }
            self._message_queue[symbol].append(message)

            # Notify subscribers
            self._notify_subscribers(symbol, data_type, message)

            # Clean old data
            self._clean_expired_data(symbol)

            logger.debug(f"{agent_name} published {data_type} for {symbol}")

    def subscribe(self, agent_name: str, symbol: str, data_type: str, callback: Callable):
        """Subscribe to data updates"""
        with self._lock:
            key = f"{symbol}:{data_type}"
            self._subscribers[key].append({
                'agent': agent_name,
                'callback': callback
            })
            logger.debug(f"{agent_name} subscribed to {data_type} for {symbol}")

    def get_context(self, symbol: str, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get enriched context for a symbol"""
        with self._lock:
            context = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': {}
            }

            symbol_data = self._data_store.get(symbol, {})

            # Filter by requested data types
            if data_types:
                for data_type in data_types:
                    if data_type in symbol_data:
                        context['data'][data_type] = self._get_latest_data(symbol_data[data_type])
            else:
                # Get all available data
                for data_type, agent_data in symbol_data.items():
                    context['data'][data_type] = self._get_latest_data(agent_data)

            # Add recent messages
            context['recent_messages'] = list(self._message_queue[symbol])[-10:]

            return context

    def get_agent_insights(self, symbol: str, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get all insights from a specific agent for a symbol"""
        with self._lock:
            insights = {}
            symbol_data = self._data_store.get(symbol, {})

            for data_type, agent_data in symbol_data.items():
                if agent_name in agent_data:
                    entry = agent_data[agent_name]
                    if self._is_data_valid(entry):
                        insights[data_type] = entry['data']

            return insights if insights else None

    def broadcast_market_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast market-wide events to all agents"""
        with self._lock:
            event = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.now()
            }

            # Notify all subscribers of market events
            for key, subscribers in self._subscribers.items():
                if key.startswith('MARKET:'):
                    for sub in subscribers:
                        try:
                            sub['callback'](event)
                        except Exception as e:
                            logger.error(f"Error in market event callback: {e}")

    def _notify_subscribers(self, symbol: str, data_type: str, message: Dict):
        """Notify subscribers of new data"""
        key = f"{symbol}:{data_type}"
        subscribers = self._subscribers.get(key, [])

        for sub in subscribers:
            try:
                sub['callback'](message)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")

    def _get_latest_data(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get latest valid data from all agents"""
        latest = {}

        for agent, entry in agent_data.items():
            if self._is_data_valid(entry):
                latest[agent] = {
                    'data': entry['data'],
                    'age_seconds': (datetime.now() - entry['timestamp']).total_seconds()
                }

        return latest

    def _is_data_valid(self, entry: Dict[str, Any]) -> bool:
        """Check if data is still valid (not expired)"""
        age = (datetime.now() - entry['timestamp']).total_seconds()
        return age < self.data_ttl

    def _clean_expired_data(self, symbol: str):
        """Remove expired data entries"""
        symbol_data = self._data_store.get(symbol, {})

        for data_type in list(symbol_data.keys()):
            agent_data = symbol_data[data_type]

            for agent in list(agent_data.keys()):
                if not self._is_data_valid(agent_data[agent]):
                    del agent_data[agent]

            # Remove empty data types
            if not agent_data:
                del symbol_data[data_type]


class EnrichedAgent:
    """Base class for agents that use the shared data bus"""

    def __init__(self, name: str, data_bus: AgentDataBus):
        self.name = name
        self.data_bus = data_bus

    def publish_insight(self, symbol: str, insight_type: str, data: Any):
        """Publish insight to the data bus"""
        self.data_bus.publish(self.name, symbol, insight_type, data)

    def get_shared_context(self, symbol: str, required_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get enriched context from other agents"""
        return self.data_bus.get_context(symbol, required_data)

    def subscribe_to_insights(self, symbol: str, insight_type: str, callback: Callable):
        """Subscribe to specific insights from other agents"""
        self.data_bus.subscribe(self.name, symbol, insight_type, callback)

    def get_peer_insights(self, symbol: str, peer_agent: str) -> Optional[Dict[str, Any]]:
        """Get insights from a specific peer agent"""
        return self.data_bus.get_agent_insights(symbol, peer_agent)


# Example: Shared data types that agents can publish
class SharedDataTypes:
    """Standard data types for agent communication"""

    # Price Action
    SUPPORT_RESISTANCE = "support_resistance_levels"
    TREND_DIRECTION = "trend_direction"
    PRICE_PATTERNS = "price_patterns"
    KEY_LEVELS = "key_price_levels"

    # Volume Analysis
    VOLUME_PROFILE = "volume_profile"
    VOLUME_SPIKES = "volume_spikes"
    ORDER_FLOW = "order_flow_imbalance"
    ACCUMULATION_DISTRIBUTION = "accumulation_distribution"

    # Market Structure
    MARKET_REGIME = "market_regime"
    VOLATILITY_STATE = "volatility_state"
    LIQUIDITY_LEVELS = "liquidity_levels"

    # Sentiment
    SENTIMENT_SCORE = "sentiment_score"
    NEWS_IMPACT = "news_impact"
    SOCIAL_BUZZ = "social_buzz"

    # Technical Indicators
    OVERBOUGHT_OVERSOLD = "overbought_oversold"
    MOMENTUM_STATE = "momentum_state"
    DIVERGENCES = "divergences"

    # Options Flow
    OPTIONS_FLOW_BIAS = "options_flow_bias"
    UNUSUAL_OPTIONS = "unusual_options_activity"
    PUT_CALL_RATIO = "put_call_ratio"


# Example usage:
if __name__ == "__main__":
    # Create data bus
    data_bus = AgentDataBus()

    # Example: Volume Profile Agent publishes data
    data_bus.publish(
        "VolumeProfileAgent",
        "AAPL",
        SharedDataTypes.VOLUME_PROFILE,
        {
            "poc": 175.50,  # Point of Control
            "value_area_high": 177.00,
            "value_area_low": 174.00,
            "volume_nodes": [
                {"price": 175.50, "volume": 5000000},
                {"price": 176.00, "volume": 3000000}
            ]
        }
    )

    # Example: Pattern Agent subscribes to volume data
    def on_volume_update(message):
        print(f"Pattern Agent received: {message}")

    data_bus.subscribe("PatternAgent", "AAPL", SharedDataTypes.VOLUME_PROFILE, on_volume_update)

    # Example: RSI Agent gets enriched context
    context = data_bus.get_context("AAPL", [SharedDataTypes.VOLUME_PROFILE, SharedDataTypes.TREND_DIRECTION])
    print(f"RSI Agent context: {context}")
