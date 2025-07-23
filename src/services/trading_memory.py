"""
Trading Memory Service - Lightweight RAG Implementation
Stores and retrieves trading patterns, outcomes, and market conditions
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TradingMemory:
    """
    Lightweight RAG system for trading intelligence
    Uses TF-IDF for simple but effective similarity search
    """

    def __init__(self, memory_file: str = "trading_memory.json"):
        self.memory_file = memory_file
        self.memories: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.vectors = None
        self.load_memory()

    def load_memory(self):
        """Load existing memories from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memories = json.load(f)
                self._update_vectors()
                logger.info(f"Loaded {len(self.memories)} trading memories")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self.memories = []

    def save_memory(self):
        """Persist memories to file"""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memories, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def remember_signal(
        self,
        signal_id: str,
        symbol: str,
        action: str,
        price: float,
        reasoning: str,
        market_conditions: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ):
        """Store a trading signal and its context"""
        memory = {
            "id": signal_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "reasoning": reasoning,
            "market_conditions": market_conditions,
            "outcome": outcome,
            "context": self._create_context(symbol, action, reasoning, market_conditions),
        }

        self.memories.append(memory)
        self._update_vectors()
        self.save_memory()

        logger.info(f"Remembered signal {signal_id} for {symbol}")

    def find_similar_setups(
        self, symbol: str, current_conditions: Dict[str, Any], reasoning: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar historical trading setups"""
        if not self.memories or self.vectors is None:
            return []

        # Create query context
        query_context = self._create_context(symbol, "UNKNOWN", reasoning, current_conditions)

        # Vectorize query
        query_vector = self.vectorizer.transform([query_context])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # Get top k similar memories
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        similar_memories = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                memory = self.memories[idx].copy()
                memory["similarity"] = float(similarities[idx])
                similar_memories.append(memory)

        return similar_memories

    def get_pattern_success_rate(self, pattern: str) -> Dict[str, float]:
        """Calculate success rate for a specific pattern"""
        pattern_memories = [
            m
            for m in self.memories
            if pattern.lower() in m.get("reasoning", "").lower() and m.get("outcome") is not None
        ]

        if not pattern_memories:
            return {"success_rate": 0.5, "sample_size": 0}

        successful = sum(1 for m in pattern_memories if m["outcome"].get("profitable", False))

        return {
            "success_rate": successful / len(pattern_memories),
            "sample_size": len(pattern_memories),
            "avg_return": np.mean([m["outcome"].get("return_pct", 0) for m in pattern_memories]),
        }

    def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get historical performance for a symbol"""
        symbol_memories = [
            m for m in self.memories if m["symbol"] == symbol and m.get("outcome") is not None
        ]

        if not symbol_memories:
            return {"total_trades": 0, "win_rate": 0.5, "avg_return": 0}

        wins = sum(1 for m in symbol_memories if m["outcome"].get("profitable", False))

        returns = [m["outcome"].get("return_pct", 0) for m in symbol_memories]

        return {
            "total_trades": len(symbol_memories),
            "win_rate": wins / len(symbol_memories),
            "avg_return": np.mean(returns),
            "best_return": max(returns),
            "worst_return": min(returns),
            "volatility": np.std(returns),
        }

    def update_outcome(
        self, signal_id: str, exit_price: float, profitable: bool, return_pct: float
    ):
        """Update the outcome of a remembered signal"""
        for memory in self.memories:
            if memory["id"] == signal_id:
                memory["outcome"] = {
                    "exit_price": exit_price,
                    "profitable": profitable,
                    "return_pct": return_pct,
                    "updated_at": datetime.now().isoformat(),
                }
                self.save_memory()
                logger.info(f"Updated outcome for signal {signal_id}")
                break

    def _create_context(
        self, symbol: str, action: str, reasoning: str, conditions: Dict[str, Any]
    ) -> str:
        """Create searchable context string"""
        context_parts = [
            f"Symbol: {symbol}",
            f"Action: {action}",
            f"Reasoning: {reasoning}",
        ]

        # Add market conditions
        for key, value in conditions.items():
            context_parts.append(f"{key}: {value}")

        return " ".join(context_parts)

    def _update_vectors(self):
        """Update TF-IDF vectors for all memories"""
        if not self.memories:
            return

        contexts = [m["context"] for m in self.memories]
        self.vectors = self.vectorizer.fit_transform(contexts)

    def get_market_regime_performance(self, regime: str) -> Dict[str, Any]:
        """Get performance statistics for a market regime"""
        regime_memories = [
            m
            for m in self.memories
            if m.get("market_conditions", {}).get("regime") == regime
            and m.get("outcome") is not None
        ]

        if not regime_memories:
            return {"trades": 0, "win_rate": 0.5}

        wins = sum(1 for m in regime_memories if m["outcome"].get("profitable", False))

        return {
            "trades": len(regime_memories),
            "win_rate": wins / len(regime_memories),
            "best_performers": self._get_best_patterns(regime_memories),
            "worst_performers": self._get_worst_patterns(regime_memories),
        }

    def _get_best_patterns(self, memories: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
        """Extract best performing patterns from memories"""
        pattern_returns = {}

        for memory in memories:
            if memory.get("outcome"):
                # Extract key phrases from reasoning
                reasoning = memory.get("reasoning", "")
                for phrase in ["RSI", "MACD", "Volume", "Breakout", "Support", "Resistance"]:
                    if phrase.lower() in reasoning.lower():
                        if phrase not in pattern_returns:
                            pattern_returns[phrase] = []
                        pattern_returns[phrase].append(memory["outcome"].get("return_pct", 0))

        # Calculate average returns
        avg_returns = {
            pattern: np.mean(returns)
            for pattern, returns in pattern_returns.items()
            if len(returns) > 2  # Minimum sample size
        }

        # Sort by return
        sorted_patterns = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)

        return [p[0] for p in sorted_patterns[:top_k]]

    def _get_worst_patterns(self, memories: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
        """Extract worst performing patterns from memories"""
        pattern_returns = {}

        for memory in memories:
            if memory.get("outcome"):
                reasoning = memory.get("reasoning", "")
                for phrase in ["RSI", "MACD", "Volume", "Breakout", "Support", "Resistance"]:
                    if phrase.lower() in reasoning.lower():
                        if phrase not in pattern_returns:
                            pattern_returns[phrase] = []
                        pattern_returns[phrase].append(memory["outcome"].get("return_pct", 0))

        avg_returns = {
            pattern: np.mean(returns)
            for pattern, returns in pattern_returns.items()
            if len(returns) > 2
        }

        sorted_patterns = sorted(avg_returns.items(), key=lambda x: x[1])

        return [p[0] for p in sorted_patterns[:top_k]]


# Singleton instance
trading_memory = TradingMemory()


def remember_trade(signal_data: Dict[str, Any], market_conditions: Dict[str, Any]):
    """Helper function to remember a trade"""
    trading_memory.remember_signal(
        signal_id=signal_data["id"],
        symbol=signal_data["symbol"],
        action=signal_data["action"],
        price=signal_data["price"],
        reasoning=signal_data["reasoning"],
        market_conditions=market_conditions,
    )


def find_similar_trades(
    symbol: str, conditions: Dict[str, Any], reasoning: str
) -> List[Dict[str, Any]]:
    """Helper function to find similar historical trades"""
    return trading_memory.find_similar_setups(symbol, conditions, reasoning)
