"""
Meta signal agent for combining signals from multiple agents.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# from agents.technical.momentum.momentum_analyzer import MomentumAnalyzer
# from agents.core.technical.trend_following_agent import TrendFollowingAgent
# from agents.core.technical.mean_reversion_agent import MeanReversionAgent
# from agents.core.technical.breakout_agent import BreakoutAgent
# from agents.core.technical.rsi_agent import RSIAgent
# from agents.core.volume.volume_profile_agent import VolumeProfileAgent
# from agents.core.volume.vwap_agent import VWAPAgent
# from agents.core.macro.macro_momentum_agent import MacroMomentumAgent
# from agents.core.macro.intermarket_analysis_agent import IntermarketAnalysisAgent
# from agents.core.options.options_gamma_agent import OptionsGammaAgent
# from agents.core.options.put_call_ratio_agent import PutCallRatioAgent
# from agents.core.risk.vix_sentiment_agent import VIXSentimentAgent
# from agents.core.risk.risk_management_agent import RiskManagementAgent
from agents.rag.historical_market_context_rag import HistoricalMarketContextRAG
from agents.rag.news_impact_rag import NewsImpactRAG
from agents.rag.options_flow_intelligence_rag import OptionsFlowIntelligenceRAG


# Mock momentum analyzer for demo
class MomentumAnalyzer:
    """Simple mock momentum analyzer"""
    async def analyze(self, symbol, timeframe, context):
        return {
            'signal': 'buy' if context.get('market_data', {}).get('rsi', 50) > 50 else 'hold',
            'confidence': 0.7
        }

class MetaSignalAgent:
    """
    An agent that combines signals from multiple agents using weighted voting.
    """
    def __init__(self, weight_config: Dict[str, float] = None, agent_registry=None):
        """
        Initialize the meta signal agent.

        Args:
            weight_config (Dict[str, float], optional): Weights for each agent type.
                Defaults to technical: 0.4, sentiment: 0.3, regime: 0.3
        """
        self.logger = logging.getLogger(__name__)

        # Register all available agents
        self.agents = {
            # Technical agents
            # 'trend_following': TrendFollowingAgent(),
            # 'mean_reversion': MeanReversionAgent(),
            # 'breakout': BreakoutAgent(),
            'momentum': MomentumAnalyzer(),
            # 'rsi': RSIAgent(),

            # Volume agents
            # 'volume_analysis': VolumeProfileAgent(),
            # 'vwap': VWAPAgent(),

            # Macro agents
            # 'macro_momentum': MacroMomentumAgent(),
            # 'intermarket': IntermarketAnalysisAgent(),

            # Options agents
            # 'options_gamma': OptionsGammaAgent(),
            # 'put_call': PutCallRatioAgent(),

            # Risk agents
            # 'vix': VIXSentimentAgent(),
            # 'risk_manager': RiskManagementAgent(),

            # RAG agents
            'historical_context': HistoricalMarketContextRAG(use_mock_db=True),
            'news_impact': NewsImpactRAG(use_mock_db=True),
            'options_flow': OptionsFlowIntelligenceRAG(use_mock_db=True)
        }

        # Default weights for each agent type
        self.weights = weight_config or {
            "technical": 0.4,
            "sentiment": 0.3,
            "regime": 0.3
        }

        # Agent type mappings
        self.agent_types = {
            # Technical
            # 'trend_following': 'technical',
            # 'mean_reversion': 'technical',
            # 'breakout': 'technical',
            'momentum': 'technical',
            # 'rsi': 'technical',
            # Volume
            # 'volume_analysis': 'volume',
            # 'vwap': 'volume',
            # Macro
            # 'macro_momentum': 'macro',
            # 'intermarket': 'macro',
            # Options
            # 'options_gamma': 'options',
            # 'put_call': 'options',
            # Risk
            # 'vix': 'risk',
            # 'risk_manager': 'risk',
            # RAG
            'historical_context': 'rag',
            'news_impact': 'rag',
            'options_flow': 'rag'
        }

        # Enhanced weights with more granularity
        self.enhanced_weights = {
            'technical': 0.25,
            'volume': 0.15,
            'macro': 0.15,
            'options': 0.15,
            'risk': 0.10,
            'rag': 0.20  # Higher weight for RAG insights
        }

    async def aggregate_signals(self, symbol: str, timeframe: str = '1d',
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregate signals from all available agents including RAG agents

        Args:
            symbol: Stock symbol to analyze
            timeframe: Time period for analysis
            context: Additional context (market data, news, etc.)

        Returns:
            Aggregated signal with comprehensive analysis
        """
        all_signals = {}
        signal_scores = {'buy': 0, 'sell': 0, 'hold': 0}
        agent_contributions = {}

        # Collect signals from all agents
        for agent_name, agent in self.agents.items():
            try:
                signal = None

                # RAG agents have different interfaces
                if agent_name == 'historical_context':
                    result = await agent.get_historical_context(
                        symbol=symbol,
                        current_price=context.get('current_price', 100),
                        timeframe=timeframe
                    )
                    # Extract signal from RAG response
                    signal = self._extract_rag_signal(result, 'historical')

                elif agent_name == 'news_impact':
                    result = await agent.analyze_news_impact(
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    signal = self._extract_rag_signal(result, 'news')

                elif agent_name == 'options_flow':
                    # Analyze recent options flow
                    flow_data = context.get('options_flow', {
                        'symbol': symbol,
                        'underlying_price': context.get('current_price', 100),
                        'strike': context.get('current_price', 100) * 1.05,
                        'days_to_expiry': 30,
                        'call_put': 'C',
                        'side': 'BUY',
                        'size': 1000,
                        'price': 2.0,
                        'implied_volatility': 0.3,
                        'delta': 0.5
                    })
                    result = await agent.analyze_options_flow(flow_data)
                    signal = self._extract_options_flow_signal(result)

                else:
                    # Traditional agents
                    if hasattr(agent, 'analyze'):
                        signal = await agent.analyze(symbol, timeframe, context)
                    elif hasattr(agent, 'generate_signal'):
                        signal = agent.generate_signal(context.get('market_data', {}))
                    else:
                        continue

                if signal:
                    all_signals[agent_name] = signal
                    agent_contributions[agent_name] = signal

                    # Calculate weighted score
                    agent_type = self.agent_types.get(agent_name, 'technical')
                    weight = self.enhanced_weights.get(agent_type, 0.1)

                    signal_type = signal.get('signal', 'hold').lower()
                    confidence = signal.get('confidence', 0.5)

                    signal_scores[signal_type] += confidence * weight

            except Exception as e:
                self.logger.warning(f"Error getting signal from {agent_name}: {e}")
                continue

        # Determine final signal
        total_score = sum(signal_scores.values())
        if total_score > 0:
            normalized_scores = {k: v/total_score for k, v in signal_scores.items()}
        else:
            normalized_scores = signal_scores

        final_signal = max(signal_scores, key=signal_scores.get)

        # Calculate conviction level
        signal_diff = abs(normalized_scores.get('buy', 0) - normalized_scores.get('sell', 0))
        if signal_diff > 0.6:
            conviction = 'high'
        elif signal_diff > 0.3:
            conviction = 'medium'
        else:
            conviction = 'low'

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': final_signal,
            'confidence': normalized_scores[final_signal],
            'conviction': conviction,
            'scores': normalized_scores,
            'agent_signals': all_signals,
            'top_contributors': self._get_top_contributors(agent_contributions, final_signal),
            'risk_assessment': self._aggregate_risk_assessment(all_signals),
            'key_insights': self._extract_key_insights(all_signals),
            'timestamp': datetime.now().isoformat()
        }

    def _extract_rag_signal(self, rag_result: Dict[str, Any], rag_type: str) -> Dict[str, Any]:
        """Extract trading signal from RAG analysis"""
        if rag_type == 'historical':
            pattern = rag_result.get('current_pattern', {})
            if pattern.get('pattern_type') == 'bullish_breakout':
                return {'signal': 'buy', 'confidence': pattern.get('confidence', 0.7)}
            elif pattern.get('pattern_type') == 'bearish_breakdown':
                return {'signal': 'sell', 'confidence': pattern.get('confidence', 0.7)}

        elif rag_type == 'news':
            sentiment = rag_result.get('aggregated_sentiment', {})
            score = sentiment.get('compound_score', 0)
            if score > 0.3:
                return {'signal': 'buy', 'confidence': min(score, 0.9)}
            elif score < -0.3:
                return {'signal': 'sell', 'confidence': min(abs(score), 0.9)}

        return {'signal': 'hold', 'confidence': 0.5}

    def _extract_options_flow_signal(self, flow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signal from options flow analysis"""
        signals = flow_result.get('trading_signals', {})

        if signals.get('follow_smart_money'):
            action = signals.get('action', 'hold')
            confidence = signals.get('confidence', 0.5)
            return {'signal': action, 'confidence': confidence}

        return {'signal': 'hold', 'confidence': 0.5}

    def _get_top_contributors(self, contributions: Dict[str, Dict],
                            final_signal: str) -> List[Dict[str, Any]]:
        """Get top agents contributing to the final signal"""
        contributors = []

        for agent_name, signal in contributions.items():
            if signal.get('signal') == final_signal:
                contributors.append({
                    'agent': agent_name,
                    'confidence': signal.get('confidence', 0),
                    'type': self.agent_types.get(agent_name, 'unknown')
                })

        # Sort by confidence
        contributors.sort(key=lambda x: x['confidence'], reverse=True)
        return contributors[:5]  # Top 5

    def _aggregate_risk_assessment(self, all_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate risk assessments from all agents"""
        risk_scores = []
        stop_losses = []
        take_profits = []

        for agent_name, signal in all_signals.items():
            if 'risk' in signal:
                risk_scores.append(signal['risk'])
            if 'stop_loss' in signal:
                stop_losses.append(signal['stop_loss'])
            if 'take_profit' in signal:
                take_profits.append(signal['take_profit'])

        return {
            'average_risk': np.mean(risk_scores) if risk_scores else 0.5,
            'suggested_stop_loss': np.mean(stop_losses) if stop_losses else 2.0,
            'suggested_take_profit': np.mean(take_profits) if take_profits else 5.0,
            'position_size_recommendation': self._calculate_position_size(risk_scores)
        }

    def _calculate_position_size(self, risk_scores: List[float]) -> float:
        """Calculate recommended position size based on risk"""
        if not risk_scores:
            return 0.5

        avg_risk = np.mean(risk_scores)
        if avg_risk < 0.3:
            return 1.0  # Full position
        elif avg_risk < 0.5:
            return 0.75
        elif avg_risk < 0.7:
            return 0.5
        else:
            return 0.25  # Quarter position

    def _extract_key_insights(self, all_signals: Dict[str, Dict]) -> List[str]:
        """Extract key insights from all agent signals"""
        insights = []

        # Check for strong consensus
        buy_count = sum(1 for s in all_signals.values() if s.get('signal') == 'buy')
        sell_count = sum(1 for s in all_signals.values() if s.get('signal') == 'sell')

        if buy_count > len(all_signals) * 0.7:
            insights.append("Strong bullish consensus across agents")
        elif sell_count > len(all_signals) * 0.7:
            insights.append("Strong bearish consensus across agents")

        # Add specific insights from each agent type
        for agent_name, signal in all_signals.items():
            if 'insight' in signal:
                insights.append(f"{agent_name}: {signal['insight']}")

        return insights[:10]  # Limit to top 10 insights

    def predict(self, agent_signals: Dict[str, Dict]) -> Dict:
        """
        Combine signals from multiple agents using weighted voting.

        Args:
            agent_signals (Dict[str, Dict]): Signals from different agents.
                Format: { "technical": {"signal": "buy", "confidence": 0.8}, ... }

        Returns:
            Dict: Combined signal with format:
                {
                    "signal": str,  # "buy", "sell", or "hold"
                    "score": float,  # Confidence score
                    "details": Dict  # Vote scores for each signal
                }
        """
        vote_scores = {"buy": 0, "sell": 0, "hold": 0}

        for agent, data in agent_signals.items():
            signal = data["signal"]
            confidence = data.get("confidence", 0.5)
            weight = self.weights.get(agent, 0.0)
            vote_scores[signal] += confidence * weight

        final_signal = max(vote_scores, key=vote_scores.get)
        return {
            "signal": final_signal,
            "score": vote_scores[final_signal],
            "details": vote_scores
        }
