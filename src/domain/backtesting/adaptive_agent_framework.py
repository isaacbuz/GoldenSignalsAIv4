"""
Adaptive Agent Framework - Online learning and performance optimization
Enables agents to learn from trading outcomes and adapt their strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import logging
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Represents a trading decision made by an agent"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    features: Dict[str, float]
    predicted_return: float
    reasoning: str
    agent_id: str
    model_version: int


@dataclass
class TradingOutcome:
    """Actual outcome of a trading decision"""
    decision: TradingDecision
    actual_return: float
    market_conditions: Dict[str, float]
    execution_quality: float  # 0-1 score
    timestamp: datetime


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an adaptive agent"""
    total_decisions: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    accuracy: float  # Direction prediction accuracy
    information_ratio: float
    learning_rate_decay: float
    model_stability: float  # How stable predictions are
    last_updated: datetime


class OnlineLearner:
    """
    Online learning component using incremental algorithms
    Supports both classification (direction) and regression (returns)
    """
    
    def __init__(self, learning_rate: float = 0.01):
        # Incremental models
        self.direction_model = SGDClassifier(
            loss='log_loss',
            learning_rate='adaptive',
            eta0=learning_rate,
            random_state=42
        )
        
        self.magnitude_model = SGDRegressor(
            loss='squared_error',
            learning_rate='adaptive',
            eta0=learning_rate,
            random_state=42
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # State
        self.is_initialized = False
        self.feature_names = []
        self.update_count = 0
        
    def partial_fit(
        self, 
        features: np.ndarray, 
        directions: np.ndarray,
        returns: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Incrementally update models with new data
        
        Args:
            features: Feature matrix (n_samples, n_features)
            directions: Direction labels (1 for up, -1 for down)
            returns: Actual returns
            sample_weight: Optional sample weights
        """
        if not self.is_initialized:
            # Initialize models with first batch
            self.scaler.partial_fit(features)
            scaled_features = self.scaler.transform(features)
            
            self.direction_model.partial_fit(
                scaled_features, directions, 
                classes=[-1, 1],
                sample_weight=sample_weight
            )
            self.magnitude_model.partial_fit(
                scaled_features, returns,
                sample_weight=sample_weight
            )
            self.is_initialized = True
        else:
            # Update scaler and models
            self.scaler.partial_fit(features)
            scaled_features = self.scaler.transform(features)
            
            self.direction_model.partial_fit(
                scaled_features, directions,
                sample_weight=sample_weight
            )
            self.magnitude_model.partial_fit(
                scaled_features, returns,
                sample_weight=sample_weight
            )
        
        self.update_count += 1
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Returns:
            Tuple of (directions, returns, confidence_scores)
        """
        if not self.is_initialized:
            raise ValueError("Model not initialized. Call partial_fit first.")
        
        scaled_features = self.scaler.transform(features)
        
        # Get predictions
        directions = self.direction_model.predict(scaled_features)
        confidence = self.direction_model.predict_proba(scaled_features).max(axis=1)
        returns = self.magnitude_model.predict(scaled_features)
        
        return directions, returns, confidence


class AdaptiveAgent(ABC):
    """
    Base class for adaptive trading agents with online learning
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_capital: float = 100000,
        learning_config: Dict[str, Any] = None
    ):
        self.agent_id = agent_id
        self.initial_capital = initial_capital
        self.learning_config = learning_config or {}
        
        # Online learner
        self.learner = OnlineLearner(
            learning_rate=self.learning_config.get('learning_rate', 0.01)
        )
        
        # Performance tracking
        self.decision_buffer = deque(maxlen=1000)
        self.outcome_buffer = deque(maxlen=1000)
        self.performance_history = []
        self.model_version = 1
        
        # Learning parameters
        self.min_samples_before_update = self.learning_config.get('min_samples', 10)
        self.update_frequency = self.learning_config.get('update_frequency', 50)
        self.exploration_rate = self.learning_config.get('exploration_rate', 0.1)
        self.confidence_threshold = self.learning_config.get('confidence_threshold', 0.6)
        
        # A/B testing state
        self.is_control_group = False
        self.experiment_id = None
        
    @abstractmethod
    def extract_features(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Extract features from market data - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_base_signals(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Get base trading signals before learning enhancement"""
        pass
    
    async def make_decision(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        current_position: Optional[Dict] = None
    ) -> TradingDecision:
        """
        Make a trading decision with optional learning enhancement
        
        Args:
            market_data: Historical market data
            symbol: Trading symbol
            current_position: Current position information
            
        Returns:
            Trading decision
        """
        # Extract features
        features = self.extract_features(market_data, symbol)
        
        # Get base signals
        base_signals = self.get_base_signals(features)
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate and not self.is_control_group:
            # Exploration: make random decision for learning
            action = np.random.choice(['BUY', 'SELL', 'HOLD'])
            confidence = self.exploration_rate
            predicted_return = np.random.normal(0, 0.02)  # Random return prediction
            reasoning = "Exploration for learning"
        else:
            # Exploitation: use learned model if available
            if self.learner.is_initialized and not self.is_control_group:
                # Prepare features for model
                feature_array = np.array([list(features.values())])
                
                # Get model predictions
                directions, returns, confidence_scores = self.learner.predict(feature_array)
                
                direction = directions[0]
                predicted_return = returns[0]
                confidence = confidence_scores[0]
                
                # Combine with base signals
                if confidence > self.confidence_threshold:
                    # Trust the learned model
                    if direction > 0:
                        action = 'BUY'
                    elif direction < 0:
                        action = 'SELL'
                    else:
                        action = 'HOLD'
                    reasoning = f"ML enhanced: {base_signals.get('reasoning', '')}"
                else:
                    # Fall back to base signals
                    action = base_signals.get('action', 'HOLD')
                    predicted_return = base_signals.get('predicted_return', 0)
                    confidence = base_signals.get('confidence', 0.5)
                    reasoning = f"Base signal (low ML confidence): {base_signals.get('reasoning', '')}"
            else:
                # Use base signals
                action = base_signals.get('action', 'HOLD')
                predicted_return = base_signals.get('predicted_return', 0)
                confidence = base_signals.get('confidence', 0.5)
                reasoning = base_signals.get('reasoning', 'Base signal')
        
        # Create decision
        decision = TradingDecision(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            confidence=confidence,
            features=features,
            predicted_return=predicted_return,
            reasoning=reasoning,
            agent_id=self.agent_id,
            model_version=self.model_version
        )
        
        # Record decision
        self.decision_buffer.append(decision)
        
        return decision
    
    def record_outcome(self, outcome: TradingOutcome):
        """Record the outcome of a trading decision for learning"""
        self.outcome_buffer.append(outcome)
        
        # Check if we should update the model
        if len(self.outcome_buffer) >= self.min_samples_before_update:
            if len(self.outcome_buffer) % self.update_frequency == 0:
                self.update_model()
    
    def update_model(self):
        """Update the online learning model with recent outcomes"""
        if len(self.outcome_buffer) < self.min_samples_before_update:
            return
        
        # Prepare training data
        features_list = []
        directions = []
        returns = []
        weights = []
        
        for outcome in self.outcome_buffer:
            # Extract features
            feature_values = list(outcome.decision.features.values())
            features_list.append(feature_values)
            
            # Direction: 1 for positive return, -1 for negative
            directions.append(1 if outcome.actual_return > 0 else -1)
            
            # Actual returns
            returns.append(outcome.actual_return)
            
            # Weight by recency and execution quality
            recency_weight = np.exp(-0.01 * (datetime.now() - outcome.timestamp).days)
            quality_weight = outcome.execution_quality
            weights.append(recency_weight * quality_weight)
        
        # Convert to arrays
        X = np.array(features_list)
        y_direction = np.array(directions)
        y_returns = np.array(returns)
        sample_weights = np.array(weights)
        
        # Normalize weights
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        # Update model
        try:
            self.learner.partial_fit(X, y_direction, y_returns, sample_weights)
            self.model_version += 1
            logger.info(f"Agent {self.agent_id} updated model to version {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to update model for agent {self.agent_id}: {e}")
    
    def calculate_performance_metrics(self) -> AgentPerformanceMetrics:
        """Calculate current performance metrics"""
        if not self.outcome_buffer:
            return AgentPerformanceMetrics(
                total_decisions=0,
                win_rate=0,
                avg_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                accuracy=0,
                information_ratio=0,
                learning_rate_decay=0,
                model_stability=0,
                last_updated=datetime.now()
            )
        
        # Calculate metrics
        returns = [o.actual_return for o in self.outcome_buffer]
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns)
        avg_return = np.mean(returns)
        
        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # Direction accuracy
        correct_directions = sum(
            1 for o in self.outcome_buffer
            if (o.decision.action == 'BUY' and o.actual_return > 0) or
               (o.decision.action == 'SELL' and o.actual_return < 0)
        )
        accuracy = correct_directions / len(self.outcome_buffer)
        
        # Information ratio (simplified)
        predictions = [o.decision.predicted_return for o in self.outcome_buffer]
        actuals = [o.actual_return for o in self.outcome_buffer]
        if len(predictions) > 1:
            tracking_error = np.std(np.array(predictions) - np.array(actuals))
            if tracking_error > 0:
                information_ratio = np.mean(returns) / tracking_error
            else:
                information_ratio = 0
        else:
            information_ratio = 0
        
        # Model stability (how consistent predictions are)
        if len(predictions) > 10:
            recent_predictions = predictions[-10:]
            model_stability = 1 - np.std(recent_predictions) / (np.mean(np.abs(recent_predictions)) + 1e-6)
        else:
            model_stability = 0.5
        
        return AgentPerformanceMetrics(
            total_decisions=len(self.decision_buffer),
            win_rate=win_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            accuracy=accuracy,
            information_ratio=information_ratio,
            learning_rate_decay=1 / (1 + self.model_version * 0.1),
            model_stability=model_stability,
            last_updated=datetime.now()
        )
    
    def should_retrain(self) -> bool:
        """Determine if the agent needs retraining"""
        metrics = self.calculate_performance_metrics()
        
        # Retrain if performance is poor
        if metrics.accuracy < 0.45:  # Worse than random
            return True
        
        if metrics.sharpe_ratio < 0:  # Losing money
            return True
        
        if metrics.model_stability < 0.3:  # Unstable predictions
            return True
        
        return False
    
    def save_state(self, filepath: str):
        """Save agent state including learned model"""
        state = {
            'agent_id': self.agent_id,
            'model_version': self.model_version,
            'learner': pickle.dumps(self.learner),
            'performance_history': self.performance_history,
            'learning_config': self.learning_config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load agent state including learned model"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.agent_id = state['agent_id']
        self.model_version = state['model_version']
        self.learner = pickle.loads(state['learner'])
        self.performance_history = state['performance_history']
        self.learning_config = state['learning_config']


class AgentPerformanceTracker:
    """
    Tracks and analyzes performance of multiple adaptive agents
    """
    
    def __init__(self):
        self.agents: Dict[str, AdaptiveAgent] = {}
        self.performance_history: Dict[str, List[AgentPerformanceMetrics]] = {}
        self.experiments: Dict[str, Dict] = {}
        
    def register_agent(self, agent: AdaptiveAgent):
        """Register an agent for tracking"""
        self.agents[agent.agent_id] = agent
        self.performance_history[agent.agent_id] = []
    
    def track_decision(self, agent_id: str, decision: TradingDecision, outcome: TradingOutcome):
        """Track a decision and its outcome"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.record_outcome(outcome)
            
            # Update performance history periodically
            if len(agent.outcome_buffer) % 100 == 0:
                metrics = agent.calculate_performance_metrics()
                self.performance_history[agent_id].append(metrics)
    
    def get_agent_ranking(self) -> List[Tuple[str, float]]:
        """Rank agents by performance"""
        rankings = []
        
        for agent_id, agent in self.agents.items():
            metrics = agent.calculate_performance_metrics()
            # Composite score
            score = (
                metrics.sharpe_ratio * 0.4 +
                metrics.accuracy * 0.3 +
                metrics.information_ratio * 0.2 +
                (1 - abs(metrics.max_drawdown)) * 0.1
            )
            rankings.append((agent_id, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def trigger_retraining(self, agent_id: str):
        """Check if an agent needs retraining"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if agent.should_retrain():
                logger.info(f"Agent {agent_id} triggered for retraining")
                # In production, this would trigger a retraining pipeline
                return True
        return False
    
    def run_ab_test(
        self,
        test_name: str,
        control_agent_id: str,
        treatment_agent_id: str,
        duration_days: int = 30
    ):
        """Run an A/B test between two agent configurations"""
        if control_agent_id in self.agents and treatment_agent_id in self.agents:
            control_agent = self.agents[control_agent_id]
            treatment_agent = self.agents[treatment_agent_id]
            
            # Set up experiment
            experiment = {
                'name': test_name,
                'control': control_agent_id,
                'treatment': treatment_agent_id,
                'start_date': datetime.now(),
                'end_date': datetime.now() + timedelta(days=duration_days),
                'status': 'active'
            }
            
            self.experiments[test_name] = experiment
            
            # Configure agents
            control_agent.is_control_group = True
            control_agent.experiment_id = test_name
            treatment_agent.is_control_group = False
            treatment_agent.experiment_id = test_name
            
            logger.info(f"Started A/B test: {test_name}")
    
    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze results of an A/B test"""
        if test_name not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[test_name]
        control_metrics = self.agents[experiment['control']].calculate_performance_metrics()
        treatment_metrics = self.agents[experiment['treatment']].calculate_performance_metrics()
        
        # Statistical significance test (simplified)
        improvement = {
            'sharpe_ratio': (treatment_metrics.sharpe_ratio - control_metrics.sharpe_ratio) / (control_metrics.sharpe_ratio + 1e-6),
            'accuracy': treatment_metrics.accuracy - control_metrics.accuracy,
            'win_rate': treatment_metrics.win_rate - control_metrics.win_rate
        }
        
        # Determine winner
        if improvement['sharpe_ratio'] > 0.1 and improvement['accuracy'] > 0.05:
            winner = experiment['treatment']
            confidence = 'high'
        elif improvement['sharpe_ratio'] > 0.05:
            winner = experiment['treatment']
            confidence = 'medium'
        else:
            winner = experiment['control']
            confidence = 'low'
        
        return {
            'experiment': test_name,
            'winner': winner,
            'confidence': confidence,
            'improvement': improvement,
            'control_metrics': control_metrics,
            'treatment_metrics': treatment_metrics
        }


# Example concrete implementation
class RSIAdaptiveAgent(AdaptiveAgent):
    """Example adaptive agent using RSI strategy"""
    
    def extract_features(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Extract technical features"""
        # Calculate indicators
        close = market_data['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # Volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std()
        
        # Volume
        volume_ratio = market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
        
        return {
            'rsi': rsi.iloc[-1],
            'rsi_change': rsi.diff().iloc[-1],
            'sma_ratio': sma_20.iloc[-1] / sma_50.iloc[-1],
            'price_position': (close.iloc[-1] - close.rolling(20).min().iloc[-1]) / (close.rolling(20).max().iloc[-1] - close.rolling(20).min().iloc[-1]),
            'volatility': volatility.iloc[-1],
            'volume_ratio': volume_ratio,
            'return_1d': returns.iloc[-1],
            'return_5d': close.pct_change(5).iloc[-1]
        }
    
    def get_base_signals(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Get RSI-based signals"""
        rsi = features['rsi']
        
        if rsi < 30:
            return {
                'action': 'BUY',
                'confidence': 0.7 + (30 - rsi) / 100,
                'predicted_return': 0.02,
                'reasoning': f'RSI oversold at {rsi:.1f}'
            }
        elif rsi > 70:
            return {
                'action': 'SELL',
                'confidence': 0.7 + (rsi - 70) / 100,
                'predicted_return': -0.02,
                'reasoning': f'RSI overbought at {rsi:.1f}'
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'predicted_return': 0,
                'reasoning': f'RSI neutral at {rsi:.1f}'
            }


# Demo usage
async def demo_adaptive_agent():
    """Demonstrate adaptive agent functionality"""
    # Create an adaptive RSI agent
    agent = RSIAdaptiveAgent(
        agent_id="RSI_Adaptive_001",
        learning_config={
            'learning_rate': 0.01,
            'momentum': 0.9,
            'min_samples': 10,
            'update_frequency': 20,
            'exploration_rate': 0.1
        }
    )
    
    # Create performance tracker
    tracker = AgentPerformanceTracker()
    tracker.register_agent(agent)
    
    print("Adaptive Agent Framework Demo")
    print("=" * 50)
    
    # Simulate some market data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    market_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100),
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices + np.random.randn(100) * 0.5
    }, index=dates)
    
    # Generate some decisions and outcomes
    for i in range(50, 90):
        # Make decision
        current_data = market_data.iloc[:i]
        decision = await agent.make_decision(current_data, 'TEST')
        
        # Simulate outcome (simplified)
        actual_return = np.random.normal(0, 0.02)
        
        outcome = TradingOutcome(
            decision=decision,
            actual_return=actual_return,
            market_conditions={'volatility': 0.02},
            execution_quality=0.95,
            timestamp=datetime.now()
        )
        
        # Track outcome
        tracker.track_decision(agent.agent_id, decision, outcome)
        
        if i % 10 == 0:
            metrics = agent.calculate_performance_metrics()
            print(f"\nAfter {i-50} decisions:")
            print(f"  Win Rate: {metrics.win_rate:.1%}")
            print(f"  Accuracy: {metrics.accuracy:.1%}")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"  Model Version: {agent.model_version}")
    
    # Final performance
    print("\nFinal Performance:")
    final_metrics = agent.calculate_performance_metrics()
    print(f"  Total Decisions: {final_metrics.total_decisions}")
    print(f"  Win Rate: {final_metrics.win_rate:.1%}")
    print(f"  Sharpe Ratio: {final_metrics.sharpe_ratio:.2f}")
    print(f"  Model Stability: {final_metrics.model_stability:.2f}")
    
    # Check if retraining needed
    if agent.should_retrain():
        print("\n⚠️  Agent needs retraining!")
    else:
        print("\n✅ Agent performance is satisfactory")


if __name__ == "__main__":
    asyncio.run(demo_adaptive_agent()) 