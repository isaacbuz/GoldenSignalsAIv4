"""
Historical Market Context RAG
Provides historical context for current market conditions
Issue #180: RAG-1: Implement Historical Market Context RAG
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class MarketScenario:
    """Represents a historical market scenario"""
    date: datetime
    regime: str  # bull, bear, sideways, crisis
    vix: float
    spy_return: float
    volume_ratio: float
    events: List[str]
    outcome: str
    price_move: float
    duration_days: int
    key_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'regime': self.regime,
            'vix': self.vix,
            'spy_return': self.spy_return,
            'volume_ratio': self.volume_ratio,
            'events': self.events,
            'outcome': self.outcome,
            'price_move': self.price_move,
            'duration_days': self.duration_days,
            'key_factors': self.key_factors
        }

    def to_text(self) -> str:
        """Convert scenario to searchable text"""
        return f"""
        Date: {self.date.strftime('%Y-%m-%d')}
        Market Regime: {self.regime}
        VIX: {self.vix:.2f}, SPY Return: {self.spy_return:.2f}%
        Volume: {self.volume_ratio:.2f}x average
        Events: {', '.join(self.events)}
        Outcome: {self.outcome}
        Price Move: {self.price_move:.2f}% over {self.duration_days} days
        Key Factors: {', '.join(self.key_factors)}
        """


class HistoricalMarketContextRAG:
    """
    RAG system for retrieving relevant historical market scenarios
    Provides context for better trading decisions
    """

    def __init__(self, use_mock_db: bool = True):
        """
        Initialize the RAG system
        Args:
            use_mock_db: Use in-memory mock database for testing
        """
        self.use_mock_db = use_mock_db
        self.scenarios: List[MarketScenario] = []
        self.embeddings: Dict[str, np.ndarray] = {}

        if use_mock_db:
            self._load_mock_scenarios()

    def _load_mock_scenarios(self):
        """Load mock historical scenarios for testing"""
        mock_scenarios = [
            # 2020 COVID Crash
            MarketScenario(
                date=datetime(2020, 3, 12),
                regime="crisis",
                vix=75.47,
                spy_return=-9.51,
                volume_ratio=3.2,
                events=["COVID-19 pandemic", "WHO declaration", "Travel bans"],
                outcome="V-shaped recovery after Fed intervention",
                price_move=28.5,
                duration_days=45,
                key_factors=["Pandemic fear", "Liquidity crisis", "Fed stimulus"]
            ),
            # 2008 Financial Crisis
            MarketScenario(
                date=datetime(2008, 9, 15),
                regime="crisis",
                vix=81.17,
                spy_return=-8.8,
                volume_ratio=4.1,
                events=["Lehman Brothers bankruptcy", "Banking crisis"],
                outcome="Extended bear market",
                price_move=-45.0,
                duration_days=180,
                key_factors=["Credit crisis", "Housing collapse", "Systemic risk"]
            ),
            # 2022 Fed Tightening
            MarketScenario(
                date=datetime(2022, 6, 13),
                regime="bear",
                vix=34.02,
                spy_return=-5.8,
                volume_ratio=1.8,
                events=["Fed rate hike 75bps", "Inflation concerns"],
                outcome="Bear market continuation",
                price_move=-12.0,
                duration_days=90,
                key_factors=["Inflation", "Rate hikes", "Recession fears"]
            ),
            # 2021 Meme Stock Rally
            MarketScenario(
                date=datetime(2021, 1, 27),
                regime="bull",
                vix=37.21,
                spy_return=-2.6,
                volume_ratio=2.5,
                events=["GME short squeeze", "Retail trading surge"],
                outcome="Sector rotation after volatility",
                price_move=5.0,
                duration_days=30,
                key_factors=["Retail frenzy", "Short covering", "Social media"]
            ),
            # Normal Bull Market
            MarketScenario(
                date=datetime(2019, 10, 15),
                regime="bull",
                vix=12.56,
                spy_return=0.5,
                volume_ratio=0.9,
                events=["Earnings season", "Trade optimism"],
                outcome="Continued uptrend",
                price_move=3.5,
                duration_days=30,
                key_factors=["Low volatility", "Earnings growth", "Buybacks"]
            )
        ]

        self.scenarios.extend(mock_scenarios)
        self._generate_mock_embeddings()

    def _generate_mock_embeddings(self):
        """Generate mock embeddings for scenarios"""
        for i, scenario in enumerate(self.scenarios):
            # Create a simple embedding based on key features
            embedding = np.array([
                scenario.vix / 100,  # Normalized VIX
                scenario.spy_return / 10,  # Normalized return
                scenario.volume_ratio / 5,  # Normalized volume
                1.0 if scenario.regime == "crisis" else 0.0,
                1.0 if scenario.regime == "bear" else 0.0,
                1.0 if scenario.regime == "bull" else 0.0,
                len(scenario.events) / 5,  # Event intensity
                scenario.price_move / 50,  # Normalized outcome
            ])
            self.embeddings[scenario.date.isoformat()] = embedding

    def _calculate_similarity(self, query_embedding: np.ndarray,
                            scenario_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(query_embedding, scenario_embedding)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(scenario_embedding)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    async def index_scenario(self, scenario: MarketScenario) -> bool:
        """
        Index a new historical scenario
        Args:
            scenario: Market scenario to index
        Returns:
            Success status
        """
        try:
            self.scenarios.append(scenario)

            # Generate embedding for the scenario
            embedding = np.array([
                scenario.vix / 100,
                scenario.spy_return / 10,
                scenario.volume_ratio / 5,
                1.0 if scenario.regime == "crisis" else 0.0,
                1.0 if scenario.regime == "bear" else 0.0,
                1.0 if scenario.regime == "bull" else 0.0,
                len(scenario.events) / 5,
                scenario.price_move / 50,
            ])

            self.embeddings[scenario.date.isoformat()] = embedding

            logger.info(f"Indexed scenario from {scenario.date}")
            return True

        except Exception as e:
            logger.error(f"Failed to index scenario: {e}")
            return False

    async def retrieve_similar_scenarios(self,
                                       current_market: Dict[str, Any],
                                       top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve historical scenarios similar to current market conditions

        Args:
            current_market: Current market conditions
            top_k: Number of similar scenarios to retrieve

        Returns:
            Dictionary with current conditions, historical matches, and insights
        """
        # Create embedding for current market
        query_embedding = np.array([
            current_market.get('vix', 20) / 100,
            current_market.get('spy_change', 0) / 10,
            current_market.get('volume_ratio', 1.0) / 5,
            1.0 if current_market.get('regime') == "crisis" else 0.0,
            1.0 if current_market.get('regime') == "bear" else 0.0,
            1.0 if current_market.get('regime') == "bull" else 0.0,
            len(current_market.get('events', [])) / 5,
            0.0  # Unknown outcome
        ])

        # Calculate similarities
        similarities = []
        for scenario in self.scenarios:
            scenario_embedding = self.embeddings[scenario.date.isoformat()]
            similarity = self._calculate_similarity(query_embedding, scenario_embedding)
            similarities.append((similarity, scenario))

        # Sort by similarity and get top K
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_matches = similarities[:top_k]

        # Extract insights
        insights = []
        for similarity, scenario in top_matches:
            insights.append({
                'similarity': float(similarity),
                'date': scenario.date.isoformat(),
                'regime': scenario.regime,
                'outcome': scenario.outcome,
                'price_move': scenario.price_move,
                'duration': scenario.duration_days,
                'key_factors': scenario.key_factors,
                'events': scenario.events
            })

        # Calculate aggregate statistics
        price_moves = [s[1].price_move for s in top_matches]
        avg_move = np.mean(price_moves) if price_moves else 0

        # Determine most likely outcome
        outcomes = [s[1].outcome for s in top_matches]
        outcome_counts = {}
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        likely_outcome = max(outcome_counts, key=outcome_counts.get) if outcome_counts else "Unknown"

        return {
            'current_conditions': current_market,
            'historical_matches': insights,
            'avg_expected_move': float(avg_move),
            'likely_outcome': likely_outcome,
            'confidence': float(np.mean([s[0] for s in top_matches])) if top_matches else 0.0,
            'recommendation': self._generate_recommendation(insights, avg_move)
        }

    def _generate_recommendation(self, insights: List[Dict], avg_move: float) -> Dict[str, Any]:
        """Generate trading recommendation based on historical insights"""
        if not insights:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'rationale': 'Insufficient historical data'
            }

        # Analyze historical outcomes
        positive_outcomes = sum(1 for i in insights if i['price_move'] > 0)
        negative_outcomes = len(insights) - positive_outcomes

        # Determine action
        if positive_outcomes > negative_outcomes * 2:
            action = 'buy'
            confidence = positive_outcomes / len(insights)
        elif negative_outcomes > positive_outcomes * 2:
            action = 'sell'
            confidence = negative_outcomes / len(insights)
        else:
            action = 'hold'
            confidence = 0.5

        # Calculate risk metrics
        moves = [i['price_move'] for i in insights]
        max_drawdown = min(moves) if moves else 0
        max_gain = max(moves) if moves else 0

        return {
            'action': action,
            'confidence': float(confidence),
            'expected_move': float(avg_move),
            'max_drawdown': float(max_drawdown),
            'max_gain': float(max_gain),
            'rationale': f"Based on {len(insights)} similar historical scenarios",
            'risk_reward_ratio': abs(max_gain / max_drawdown) if max_drawdown != 0 else float('inf')
        }

    async def get_regime_context(self, regime: str) -> Dict[str, Any]:
        """Get historical context for a specific market regime"""
        regime_scenarios = [s for s in self.scenarios if s.regime == regime]

        if not regime_scenarios:
            return {
                'regime': regime,
                'historical_count': 0,
                'avg_duration': 0,
                'avg_move': 0
            }

        durations = [s.duration_days for s in regime_scenarios]
        moves = [s.price_move for s in regime_scenarios]

        return {
            'regime': regime,
            'historical_count': len(regime_scenarios),
            'avg_duration': float(np.mean(durations)),
            'avg_move': float(np.mean(moves)),
            'typical_events': list(set(e for s in regime_scenarios for e in s.events))[:5],
            'typical_factors': list(set(f for s in regime_scenarios for f in s.key_factors))[:5]
        }


# Example usage
async def demo_historical_rag():
    """Demonstrate the Historical Market Context RAG"""
    rag = HistoricalMarketContextRAG(use_mock_db=True)

    # Current market conditions (example)
    current_market = {
        'vix': 28.5,
        'spy_change': -3.2,
        'volume_ratio': 1.8,
        'events': ['Fed meeting', 'Earnings warnings'],
        'regime': 'bear'
    }

    # Retrieve similar scenarios
    results = await rag.retrieve_similar_scenarios(current_market, top_k=3)

    print("Historical Market Context Analysis")
    print("="*50)
    print(f"Current VIX: {current_market['vix']}")
    print(f"Current SPY Change: {current_market['spy_change']}%")
    print(f"Current Events: {', '.join(current_market['events'])}")
    print("\nSimilar Historical Scenarios:")

    for i, match in enumerate(results['historical_matches'], 1):
        print(f"\n{i}. {match['date'][:10]} (Similarity: {match['similarity']:.2%})")
        print(f"   Regime: {match['regime']}")
        print(f"   Outcome: {match['outcome']}")
        print(f"   Price Move: {match['price_move']:.1f}% over {match['duration']} days")

    print(f"\nRecommendation: {results['recommendation']['action'].upper()}")
    print(f"Confidence: {results['recommendation']['confidence']:.1%}")
    print(f"Expected Move: {results['recommendation']['expected_move']:.1f}%")
    print(f"Risk/Reward: {results['recommendation']['risk_reward_ratio']:.2f}")


if __name__ == "__main__":
    asyncio.run(demo_historical_rag())
