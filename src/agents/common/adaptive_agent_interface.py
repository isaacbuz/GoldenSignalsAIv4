"""
Adaptive Agent Interface for GoldenSignalsAI
Enables agents to receive and apply learning recommendations from the adaptive learning system
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Local imports
from src.utils.timezone_utils import now_utc
from src.domain.backtesting.adaptive_learning_system import (
    AgentPerformanceProfile,
    LearningFeedback,
    AdaptiveLearningSystem
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfiguration:
    """Dynamic configuration for an agent"""
    # Core parameters
    confidence_threshold: float = 0.6
    position_size_multiplier: float = 1.0
    max_positions: int = 5
    
    # Risk management
    stop_loss_percent: float = 0.02
    take_profit_percent: float = 0.05
    trailing_stop_enabled: bool = False
    
    # Feature selection
    enabled_features: List[str] = field(default_factory=list)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    
    # Market regime filters
    allowed_regimes: List[str] = field(default_factory=list)
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Learning parameters
    learning_enabled: bool = True
    exploration_rate: float = 0.1
    adaptation_speed: float = 0.01
    
    # Model parameters
    model_version: int = 1
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    recent_performance: List[float] = field(default_factory=list)
    performance_window: int = 100


class AdaptiveAgentInterface(ABC):
    """
    Base interface that all agents must implement to support adaptive learning
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = AgentConfiguration()
        self.performance_profile = None
        self.learning_system = None
        self.is_learning = True
        self.last_update = now_utc()
        
    async def initialize_learning(self, learning_system: AdaptiveLearningSystem):
        """Initialize connection to the learning system"""
        self.learning_system = learning_system
        
        # Load existing profile if available
        if self.agent_id in learning_system.agent_profiles:
            self.performance_profile = learning_system.agent_profiles[self.agent_id]
            await self._apply_profile_to_config()
            
        logger.info(f"Agent {self.agent_id} initialized with learning system")
        
    async def _apply_profile_to_config(self):
        """Apply performance profile insights to configuration"""
        if not self.performance_profile:
            return
            
        profile = self.performance_profile
        
        # Adjust confidence threshold based on accuracy
        if profile.accuracy < 0.5:
            self.config.confidence_threshold = min(0.8, self.config.confidence_threshold + 0.1)
        elif profile.accuracy > 0.7:
            self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - 0.05)
            
        # Adjust risk parameters based on drawdown
        if profile.max_drawdown > 0.15:
            self.config.stop_loss_percent = max(0.01, self.config.stop_loss_percent * 0.8)
            self.config.position_size_multiplier *= 0.9
            
        # Update feature weights based on importance
        if profile.feature_importance:
            self.config.feature_weights = profile.feature_importance.copy()
            
        # Set allowed regimes based on performance
        good_regimes = []
        for regime, perf in profile.performance_by_regime.items():
            if perf['accuracy'] > 0.55 and perf['count'] > 5:
                good_regimes.append(regime)
        if good_regimes:
            self.config.allowed_regimes = good_regimes
            
    async def receive_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Receive and process learning recommendations"""
        logger.info(f"Agent {self.agent_id} received {len(recommendations)} recommendations")
        
        for rec in recommendations:
            try:
                await self._apply_recommendation(rec)
            except Exception as e:
                logger.error(f"Error applying recommendation: {e}")
                
    async def _apply_recommendation(self, recommendation: Dict[str, Any]):
        """Apply a single recommendation"""
        rec_type = recommendation.get('type')
        action = recommendation.get('action')
        
        if rec_type == 'parameter_adjustment':
            await self._adjust_parameters(recommendation)
        elif rec_type == 'model_recalibration':
            await self._recalibrate_model(recommendation)
        elif rec_type == 'regime_specific_training':
            await self._update_regime_strategy(recommendation)
        elif rec_type == 'feature_engineering':
            await self._update_features(recommendation)
        elif rec_type == 'risk_management':
            await self._update_risk_management(recommendation)
            
    async def _adjust_parameters(self, rec: Dict[str, Any]):
        """Adjust agent parameters based on recommendation"""
        action = rec.get('action')
        
        if action == 'increase_confidence_threshold':
            old_threshold = self.config.confidence_threshold
            self.config.confidence_threshold = rec.get('suggested_value', 0.7)
            logger.info(f"Agent {self.agent_id}: Confidence threshold {old_threshold:.2f} -> {self.config.confidence_threshold:.2f}")
            
        elif action == 'reduce_position_size':
            self.config.position_size_multiplier *= 0.8
            logger.info(f"Agent {self.agent_id}: Reduced position size multiplier to {self.config.position_size_multiplier:.2f}")
            
    async def _recalibrate_model(self, rec: Dict[str, Any]):
        """Recalibrate model confidence scores"""
        calibration_data = rec.get('calibration_data', {})
        
        # Create calibration mapping
        self.config.model_params['calibration_map'] = {}
        
        for bucket, cal_info in calibration_data.items():
            expected = cal_info.get('expected_accuracy', 0.5)
            actual = cal_info.get('actual_accuracy', 0.5)
            
            # Calculate adjustment factor
            if expected > 0:
                adjustment = actual / expected
            else:
                adjustment = 1.0
                
            self.config.model_params['calibration_map'][float(bucket)] = adjustment
            
        logger.info(f"Agent {self.agent_id}: Model recalibrated with {len(calibration_data)} buckets")
        
    async def _update_regime_strategy(self, rec: Dict[str, Any]):
        """Update strategy for specific market regime"""
        regime = rec.get('training_focus')
        
        if regime:
            # Reduce activity in poorly performing regimes
            if regime not in self.config.regime_adjustments:
                self.config.regime_adjustments[regime] = {}
                
            self.config.regime_adjustments[regime]['confidence_multiplier'] = 0.8
            self.config.regime_adjustments[regime]['position_size_multiplier'] = 0.7
            
            logger.info(f"Agent {self.agent_id}: Updated strategy for {regime} regime")
            
    async def _update_features(self, rec: Dict[str, Any]):
        """Update feature selection and weights"""
        top_features = rec.get('top_features', {})
        
        if top_features:
            # Update enabled features
            self.config.enabled_features = list(top_features.keys())
            
            # Update feature weights
            self.config.feature_weights.update(top_features)
            
            # Normalize weights
            total_weight = sum(self.config.feature_weights.values())
            if total_weight > 0:
                self.config.feature_weights = {
                    k: v/total_weight for k, v in self.config.feature_weights.items()
                }
                
            logger.info(f"Agent {self.agent_id}: Updated to use {len(top_features)} top features")
            
    async def _update_risk_management(self, rec: Dict[str, Any]):
        """Update risk management parameters"""
        suggested_actions = rec.get('suggested_actions', [])
        
        for action in suggested_actions:
            if action == 'reduce_position_size':
                self.config.position_size_multiplier *= 0.8
            elif action == 'implement_trailing_stops':
                self.config.trailing_stop_enabled = True
            elif action == 'add_volatility_filter':
                self.config.model_params['volatility_filter'] = True
                
        logger.info(f"Agent {self.agent_id}: Updated risk management with {len(suggested_actions)} actions")
        
    @abstractmethod
    async def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals - must be implemented by each agent"""
        pass
        
    def apply_confidence_calibration(self, raw_confidence: float) -> float:
        """Apply confidence calibration based on learning"""
        if 'calibration_map' not in self.config.model_params:
            return raw_confidence
            
        # Find nearest calibration bucket
        cal_map = self.config.model_params['calibration_map']
        bucket = round(raw_confidence * 10) / 10
        
        if bucket in cal_map:
            return raw_confidence * cal_map[bucket]
        else:
            # Interpolate between nearest buckets
            buckets = sorted(cal_map.keys())
            for i in range(len(buckets) - 1):
                if buckets[i] <= raw_confidence <= buckets[i+1]:
                    # Linear interpolation
                    weight = (raw_confidence - buckets[i]) / (buckets[i+1] - buckets[i])
                    adjustment = cal_map[buckets[i]] * (1 - weight) + cal_map[buckets[i+1]] * weight
                    return raw_confidence * adjustment
                    
        return raw_confidence
        
    def apply_regime_adjustment(
        self,
        signal: Dict[str, Any],
        market_regime: str
    ) -> Dict[str, Any]:
        """Apply regime-specific adjustments to signal"""
        if market_regime in self.config.regime_adjustments:
            adjustments = self.config.regime_adjustments[market_regime]
            
            # Adjust confidence
            if 'confidence_multiplier' in adjustments:
                signal['confidence'] *= adjustments['confidence_multiplier']
                
            # Adjust position size
            if 'position_size_multiplier' in adjustments:
                if 'position_size' in signal:
                    signal['position_size'] *= adjustments['position_size_multiplier']
                    
        return signal
        
    def filter_by_regime(self, market_regime: str) -> bool:
        """Check if trading is allowed in current regime"""
        if not self.config.allowed_regimes:
            return True  # No restrictions
            
        return market_regime in self.config.allowed_regimes
        
    def calculate_position_size(
        self,
        base_size: float,
        confidence: float,
        volatility: float
    ) -> float:
        """Calculate position size with learning adjustments"""
        # Base adjustment from config
        size = base_size * self.config.position_size_multiplier
        
        # Confidence-based sizing
        if confidence > 0.7:
            size *= 1.2
        elif confidence < 0.5:
            size *= 0.8
            
        # Volatility adjustment
        if volatility > 0.3:  # High volatility
            size *= 0.7
        elif volatility < 0.1:  # Low volatility
            size *= 1.1
            
        return size
        
    def should_explore(self) -> bool:
        """Determine if agent should explore (try new strategies)"""
        if not self.config.learning_enabled:
            return False
            
        return np.random.random() < self.config.exploration_rate
        
    async def update_performance(self, signal_result: Dict[str, Any]):
        """Update agent's performance tracking"""
        # Add to recent performance
        accuracy = 1.0 if signal_result.get('profitable', False) else 0.0
        self.config.recent_performance.append(accuracy)
        
        # Keep window size
        if len(self.config.recent_performance) > self.config.performance_window:
            self.config.recent_performance.pop(0)
            
        # Adjust exploration rate based on performance
        if len(self.config.recent_performance) >= 20:
            recent_accuracy = np.mean(self.config.recent_performance[-20:])
            
            if recent_accuracy > 0.7:
                # Reduce exploration when performing well
                self.config.exploration_rate *= 0.95
            elif recent_accuracy < 0.4:
                # Increase exploration when performing poorly
                self.config.exploration_rate = min(0.3, self.config.exploration_rate * 1.1)
                
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'confidence_threshold': self.config.confidence_threshold,
            'position_size_multiplier': self.config.position_size_multiplier,
            'risk_params': {
                'stop_loss': self.config.stop_loss_percent,
                'take_profit': self.config.take_profit_percent,
                'trailing_stop': self.config.trailing_stop_enabled
            },
            'enabled_features': self.config.enabled_features,
            'allowed_regimes': self.config.allowed_regimes,
            'learning_params': {
                'enabled': self.config.learning_enabled,
                'exploration_rate': self.config.exploration_rate,
                'model_version': self.config.model_version
            },
            'recent_accuracy': np.mean(self.config.recent_performance) if self.config.recent_performance else 0.0
        }
        
    async def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'config': self.get_current_config(),
            'performance_profile': self.performance_profile.__dict__ if self.performance_profile else None,
            'last_update': self.last_update.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
    async def load_state(self, filepath: str):
        """Load agent state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Restore configuration
            if 'config' in state:
                config_data = state['config']
                self.config.confidence_threshold = config_data.get('confidence_threshold', 0.6)
                self.config.position_size_multiplier = config_data.get('position_size_multiplier', 1.0)
                self.config.enabled_features = config_data.get('enabled_features', [])
                self.config.allowed_regimes = config_data.get('allowed_regimes', [])
                
            logger.info(f"Agent {self.agent_id} state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")


# Example implementation for a momentum agent
class AdaptiveMomentumAgent(AdaptiveAgentInterface):
    """Example of an adaptive momentum trading agent"""
    
    def __init__(self):
        super().__init__("momentum_agent_v2", "technical")
        self.lookback_period = 20
        self.momentum_threshold = 0.02
        
    async def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate momentum-based signals with adaptive learning"""
        signals = []
        
        for symbol, df in market_data.items():
            if len(df) < self.lookback_period:
                continue
                
            # Calculate momentum
            returns = df['close'].pct_change(self.lookback_period)
            current_momentum = returns.iloc[-1]
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Identify market regime
            market_regime = self._identify_regime(df)
            
            # Check if regime is allowed
            if not self.filter_by_regime(market_regime):
                continue
                
            # Generate signal if momentum exceeds threshold
            if abs(current_momentum) > self.momentum_threshold:
                # Base confidence from momentum strength
                raw_confidence = min(0.9, abs(current_momentum) / 0.1)
                
                # Apply calibration
                confidence = self.apply_confidence_calibration(raw_confidence)
                
                # Check confidence threshold
                if confidence >= self.config.confidence_threshold:
                    # Determine action
                    action = 'buy' if current_momentum > 0 else 'sell'
                    
                    # Calculate position size
                    base_size = 1000  # Base position size
                    position_size = self.calculate_position_size(
                        base_size, confidence, volatility
                    )
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'action': action,
                        'confidence': confidence,
                        'position_size': position_size,
                        'entry_price': df['close'].iloc[-1],
                        'stop_loss': df['close'].iloc[-1] * (1 - self.config.stop_loss_percent),
                        'take_profit': df['close'].iloc[-1] * (1 + self.config.take_profit_percent),
                        'momentum': current_momentum,
                        'volatility': volatility,
                        'regime': market_regime,
                        'agent_id': self.agent_id,
                        'timestamp': df.index[-1]
                    }
                    
                    # Apply regime adjustments
                    signal = self.apply_regime_adjustment(signal, market_regime)
                    
                    # Add exploration if needed
                    if self.should_explore():
                        signal['exploration'] = True
                        signal['confidence'] *= 0.8  # Reduce confidence for exploration
                        
                    signals.append(signal)
                    
        return signals
        
    def _identify_regime(self, df: pd.DataFrame) -> str:
        """Identify current market regime"""
        if len(df) < 20:
            return "unknown"
            
        returns = df['close'].pct_change().dropna()
        sma_20 = df['close'].rolling(20).mean()
        
        # Trend
        if df['close'].iloc[-1] > sma_20.iloc[-1] * 1.02:
            trend = "uptrend"
        elif df['close'].iloc[-1] < sma_20.iloc[-1] * 0.98:
            trend = "downtrend"
        else:
            trend = "sideways"
            
        # Volatility
        vol = returns.std() * np.sqrt(252)
        if vol < 0.1:
            vol_regime = "low_vol"
        elif vol < 0.2:
            vol_regime = "normal_vol"
        else:
            vol_regime = "high_vol"
            
        return f"{trend}_{vol_regime}"


# Agent factory with learning support
class AdaptiveAgentFactory:
    """Factory for creating adaptive agents"""
    
    def __init__(self, learning_system: AdaptiveLearningSystem):
        self.learning_system = learning_system
        self.registered_agents = {}
        
    def register_agent(self, agent_class: type, agent_id: str):
        """Register an agent class"""
        self.registered_agents[agent_id] = agent_class
        
    async def create_agent(self, agent_id: str) -> Optional[AdaptiveAgentInterface]:
        """Create and initialize an agent with learning support"""
        if agent_id not in self.registered_agents:
            logger.error(f"Agent {agent_id} not registered")
            return None
            
        # Create agent instance
        agent_class = self.registered_agents[agent_id]
        agent = agent_class()
        
        # Initialize learning
        await agent.initialize_learning(self.learning_system)
        
        # Get latest recommendations
        recommendations = await self.learning_system.get_agent_recommendations(agent_id)
        if recommendations['status'] == 'success':
            await agent.receive_recommendations(recommendations.get('recommendations', []))
            
        return agent 