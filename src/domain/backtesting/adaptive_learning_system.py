"""
Adaptive Learning System for GoldenSignalsAI
Enables agents to learn from backtest results and improve signal generation
"""

import asyncio
import json
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Database
import asyncpg
import joblib
import numpy as np
import pandas as pd
import redis.asyncio as redis

# Machine Learning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.domain.backtesting.advanced_backtest_engine import BacktestMetrics, BacktestTrade

# Local imports
from src.utils.timezone_utils import now_utc

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformanceProfile:
    """Detailed performance profile for an agent"""
    agent_id: str
    agent_type: str
    
    # Performance metrics
    total_signals: int = 0
    accurate_signals: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Profitability metrics
    total_pnl: float = 0.0
    avg_pnl_per_signal: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Market condition performance
    performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_by_volatility: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_by_time: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Signal characteristics
    avg_confidence: float = 0.0
    confidence_calibration: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Learning parameters
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    model_version: int = 1
    last_updated: datetime = field(default_factory=now_utc)


@dataclass
class LearningFeedback:
    """Feedback data for agent learning"""
    trade_id: str
    agent_id: str
    signal_timestamp: datetime
    
    # Signal details
    symbol: str
    action: str
    confidence: float
    predicted_return: float
    
    # Actual outcome
    actual_return: float
    pnl: float
    holding_period: timedelta
    exit_reason: str
    
    # Market context
    market_regime: str
    volatility_level: float
    volume_profile: Dict[str, float]
    
    # Feature snapshot
    features_at_signal: Dict[str, float]
    indicators_at_signal: Dict[str, float]
    
    # Performance metrics
    accuracy_contribution: float
    sharpe_contribution: float
    
    # Learning signals
    reward: float
    regret: float
    surprise: float  # Difference between expected and actual


class AdaptiveLearningSystem:
    """
    Comprehensive learning system that:
    - Analyzes backtest results
    - Generates learning feedback for agents
    - Updates agent models and parameters
    - Implements meta-learning strategies
    - Manages agent evolution and versioning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.db_pool = None
        self.redis_client = None
        self.agent_profiles = {}
        self.learning_models = {}
        self.meta_learner = None
        self.performance_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 10000,
            'update_frequency': 100,
            'meta_learning_enabled': True,
            'ensemble_learning': True,
            'online_learning': True,
            'performance_threshold': 0.6,
            'exploration_decay': 0.995,
            'min_samples_for_update': 50
        }
        
    async def initialize(self):
        """Initialize the learning system"""
        # Database connections
        await self._init_database()
        
        # Redis for caching
        await self._init_redis()
        
        # Load existing agent profiles
        await self._load_agent_profiles()
        
        # Initialize meta-learner
        if self.config['meta_learning_enabled']:
            self._init_meta_learner()
            
        logger.info("Adaptive learning system initialized")
        
    async def _init_database(self):
        """Initialize database connections"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5432)),
                user=os.getenv('DB_USER', 'goldensignals'),
                password=os.getenv('DB_PASSWORD', 'password'),
                database=os.getenv('DB_NAME', 'goldensignals'),
                min_size=5,
                max_size=20
            )
            
            await self._create_tables()
            
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            
    async def _init_redis(self):
        """Initialize Redis client"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = await redis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            
    async def _create_tables(self):
        """Create learning system tables"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            # Agent profiles table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    agent_id VARCHAR(100) PRIMARY KEY,
                    agent_type VARCHAR(50) NOT NULL,
                    performance_data JSONB NOT NULL,
                    model_parameters JSONB,
                    learning_history JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Learning feedback table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_feedback (
                    id SERIAL PRIMARY KEY,
                    trade_id VARCHAR(36) NOT NULL,
                    agent_id VARCHAR(100) NOT NULL,
                    signal_timestamp TIMESTAMPTZ NOT NULL,
                    feedback_data JSONB NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Model versions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_model_versions (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(100) NOT NULL,
                    version INT NOT NULL,
                    model_data BYTEA NOT NULL,
                    performance_metrics JSONB NOT NULL,
                    training_config JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(agent_id, version)
                )
            """)
            
            # Meta-learning table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_learning_insights (
                    id SERIAL PRIMARY KEY,
                    insight_type VARCHAR(50) NOT NULL,
                    insight_data JSONB NOT NULL,
                    applicable_agents TEXT[],
                    confidence FLOAT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
    async def process_backtest_results(
        self,
        backtest_metrics: BacktestMetrics,
        trades: List[BacktestTrade],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Process backtest results and generate learning feedback
        
        Returns:
            Dictionary containing learning insights and recommendations
        """
        logger.info(f"Processing backtest results with {len(trades)} trades")
        
        # Generate feedback for each trade
        feedback_list = []
        for trade in trades:
            feedback = await self._generate_trade_feedback(trade, market_data)
            if feedback:
                feedback_list.append(feedback)
                
        # Analyze agent performance
        agent_analysis = await self._analyze_agent_performance(feedback_list)
        
        # Generate learning recommendations
        recommendations = await self._generate_learning_recommendations(
            agent_analysis, backtest_metrics
        )
        
        # Update agent profiles
        await self._update_agent_profiles(agent_analysis)
        
        # Train updated models
        if self.config['online_learning']:
            training_results = await self._train_agent_models(feedback_list)
        else:
            training_results = {}
            
        # Meta-learning insights
        if self.config['meta_learning_enabled']:
            meta_insights = await self._extract_meta_insights(
                feedback_list, agent_analysis
            )
        else:
            meta_insights = {}
            
        return {
            'feedback_generated': len(feedback_list),
            'agent_analysis': agent_analysis,
            'recommendations': recommendations,
            'training_results': training_results,
            'meta_insights': meta_insights,
            'performance_improvement': await self._calculate_improvement()
        }
        
    async def _generate_trade_feedback(
        self,
        trade: BacktestTrade,
        market_data: Dict[str, pd.DataFrame]
    ) -> Optional[LearningFeedback]:
        """Generate learning feedback from a trade"""
        try:
            signal = trade.signal
            symbol = signal.symbol
            
            if symbol not in market_data:
                return None
                
            # Get market context at signal time
            df = market_data[symbol]
            signal_idx = df.index.get_loc(signal.timestamp, method='nearest')
            
            # Calculate market regime
            market_regime = self._identify_market_regime(
                df.iloc[max(0, signal_idx-50):signal_idx+1]
            )
            
            # Calculate volatility
            returns = df['close'].pct_change()
            volatility = returns.iloc[max(0, signal_idx-20):signal_idx].std() * np.sqrt(252)
            
            # Calculate actual return
            if trade.exit_price and signal.entry_price:
                if signal.action == 'buy':
                    actual_return = (trade.exit_price - signal.entry_price) / signal.entry_price
                else:
                    actual_return = (signal.entry_price - trade.exit_price) / signal.entry_price
            else:
                actual_return = 0
                
            # Calculate reward (combination of profit and risk-adjusted return)
            if trade.pnl != 0:
                reward = trade.pnl_percent * (1 - volatility)  # Risk-adjusted
            else:
                reward = -0.01  # Small penalty for no exit
                
            # Calculate regret (opportunity cost)
            market_return = (df.iloc[min(len(df)-1, signal_idx+20)]['close'] - 
                           df.iloc[signal_idx]['close']) / df.iloc[signal_idx]['close']
            regret = max(0, market_return - actual_return)
            
            # Calculate surprise (prediction error)
            expected_return = signal.expected_return if hasattr(signal, 'expected_return') else 0.03
            surprise = abs(actual_return - expected_return)
            
            # Extract features at signal time
            features_at_signal = {}
            indicators_at_signal = {}
            
            for col in df.columns:
                if col in ['RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower']:
                    indicators_at_signal[col] = float(df.iloc[signal_idx][col])
                elif col not in ['open', 'high', 'low', 'close', 'volume']:
                    features_at_signal[col] = float(df.iloc[signal_idx][col])
                    
            # Create feedback object
            feedback = LearningFeedback(
                trade_id=f"{symbol}_{signal.timestamp.timestamp()}",
                agent_id=self._get_primary_agent(signal.agent_scores),
                signal_timestamp=signal.timestamp,
                symbol=symbol,
                action=signal.action,
                confidence=signal.confidence,
                predicted_return=expected_return,
                actual_return=actual_return,
                pnl=trade.pnl,
                holding_period=trade.time_in_trade or timedelta(0),
                exit_reason=trade.exit_reason,
                market_regime=market_regime,
                volatility_level=volatility,
                volume_profile=self._get_volume_profile(df, signal_idx),
                features_at_signal=features_at_signal,
                indicators_at_signal=indicators_at_signal,
                accuracy_contribution=1.0 if trade.pnl > 0 else 0.0,
                sharpe_contribution=trade.pnl_percent / volatility if volatility > 0 else 0,
                reward=reward,
                regret=regret,
                surprise=surprise
            )
            
            # Store feedback
            await self._store_feedback(feedback)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating feedback for trade: {e}")
            return None
            
    def _identify_market_regime(self, data: pd.DataFrame) -> str:
        """Identify market regime from price data"""
        if len(data) < 20:
            return "unknown"
            
        returns = data['close'].pct_change().dropna()
        sma_20 = data['close'].rolling(20).mean()
        
        # Trend identification
        if data['close'].iloc[-1] > sma_20.iloc[-1] * 1.02:
            trend = "uptrend"
        elif data['close'].iloc[-1] < sma_20.iloc[-1] * 0.98:
            trend = "downtrend"
        else:
            trend = "sideways"
            
        # Volatility regime
        vol = returns.std() * np.sqrt(252)
        if vol < 0.1:
            vol_regime = "low_vol"
        elif vol < 0.2:
            vol_regime = "normal_vol"
        else:
            vol_regime = "high_vol"
            
        return f"{trend}_{vol_regime}"
        
    def _get_primary_agent(self, agent_scores: Dict[str, float]) -> str:
        """Get the primary agent from scores"""
        if not agent_scores:
            return "unknown"
        return max(agent_scores.items(), key=lambda x: x[1])[0]
        
    def _get_volume_profile(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Get volume profile around signal"""
        try:
            recent_volume = data['volume'].iloc[max(0, idx-20):idx].mean()
            current_volume = data['volume'].iloc[idx]
            
            return {
                'relative_volume': current_volume / recent_volume if recent_volume > 0 else 1.0,
                'volume_trend': 'increasing' if current_volume > recent_volume else 'decreasing'
            }
        except:
            return {'relative_volume': 1.0, 'volume_trend': 'unknown'}
            
    async def _analyze_agent_performance(
        self,
        feedback_list: List[LearningFeedback]
    ) -> Dict[str, AgentPerformanceProfile]:
        """Analyze performance by agent"""
        agent_performance = {}
        
        # Group feedback by agent
        agent_feedback = defaultdict(list)
        
        for feedback in feedback_list:
            agent_feedback[feedback.agent_id].append(feedback)
            
        # Analyze each agent
        for agent_id, feedbacks in agent_feedback.items():
            profile = await self._calculate_agent_profile(agent_id, feedbacks)
            agent_performance[agent_id] = profile
            
        return agent_performance
        
    async def _calculate_agent_profile(
        self,
        agent_id: str,
        feedbacks: List[LearningFeedback]
    ) -> AgentPerformanceProfile:
        """Calculate detailed performance profile for an agent"""
        # Get or create profile
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
        else:
            profile = AgentPerformanceProfile(
                agent_id=agent_id,
                agent_type=self._infer_agent_type(agent_id)
            )
            
        # Update basic metrics
        profile.total_signals = len(feedbacks)
        profile.accurate_signals = sum(1 for f in feedbacks if f.accuracy_contribution > 0)
        profile.accuracy = profile.accurate_signals / profile.total_signals if profile.total_signals > 0 else 0
        
        # Calculate profitability metrics
        total_pnl = sum(f.pnl for f in feedbacks)
        profile.total_pnl = total_pnl
        profile.avg_pnl_per_signal = total_pnl / len(feedbacks) if feedbacks else 0
        
        # Calculate Sharpe ratio
        returns = [f.actual_return for f in feedbacks]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            profile.sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
        # Performance by market regime
        regime_performance = defaultdict(lambda: {'count': 0, 'accuracy': 0, 'pnl': 0})
        for feedback in feedbacks:
            regime = feedback.market_regime
            regime_performance[regime]['count'] += 1
            regime_performance[regime]['accuracy'] += feedback.accuracy_contribution
            regime_performance[regime]['pnl'] += feedback.pnl
            
        profile.performance_by_regime = {
            regime: {
                'accuracy': stats['accuracy'] / stats['count'] if stats['count'] > 0 else 0,
                'avg_pnl': stats['pnl'] / stats['count'] if stats['count'] > 0 else 0,
                'count': stats['count']
            }
            for regime, stats in regime_performance.items()
        }
        
        # Confidence calibration
        confidence_buckets = defaultdict(lambda: {'predicted': 0, 'actual': 0, 'count': 0})
        for feedback in feedbacks:
            bucket = round(feedback.confidence * 10) / 10  # Round to nearest 0.1
            confidence_buckets[bucket]['predicted'] += feedback.confidence
            confidence_buckets[bucket]['actual'] += feedback.accuracy_contribution
            confidence_buckets[bucket]['count'] += 1
            
        profile.confidence_calibration = {
            str(bucket): {
                'expected_accuracy': stats['predicted'] / stats['count'],
                'actual_accuracy': stats['actual'] / stats['count'],
                'calibration_error': abs(stats['predicted'] / stats['count'] - stats['actual'] / stats['count'])
            }
            for bucket, stats in confidence_buckets.items() if stats['count'] > 0
        }
        
        # Feature importance (based on successful trades)
        successful_feedbacks = [f for f in feedbacks if f.accuracy_contribution > 0]
        if successful_feedbacks:
            feature_importance = defaultdict(float)
            for feedback in successful_feedbacks:
                for feature, value in feedback.features_at_signal.items():
                    feature_importance[feature] += abs(value) * feedback.pnl
                    
            # Normalize
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                profile.feature_importance = {
                    feature: importance / total_importance
                    for feature, importance in feature_importance.items()
                }
                
        profile.last_updated = now_utc()
        
        return profile
        
    def _infer_agent_type(self, agent_id: str) -> str:
        """Infer agent type from ID"""
        if 'momentum' in agent_id.lower():
            return 'technical'
        elif 'sentiment' in agent_id.lower():
            return 'sentiment'
        elif 'ml' in agent_id.lower():
            return 'machine_learning'
        elif 'option' in agent_id.lower():
            return 'options'
        else:
            return 'hybrid'
            
    async def _generate_learning_recommendations(
        self,
        agent_analysis: Dict[str, AgentPerformanceProfile],
        backtest_metrics: BacktestMetrics
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate specific recommendations for each agent"""
        recommendations = {}
        
        for agent_id, profile in agent_analysis.items():
            agent_recommendations = []
            
            # Performance-based recommendations
            if profile.accuracy < 0.5:
                agent_recommendations.append({
                    'type': 'parameter_adjustment',
                    'priority': 'high',
                    'action': 'increase_confidence_threshold',
                    'reason': f'Low accuracy: {profile.accuracy:.2%}',
                    'suggested_value': min(0.8, profile.avg_confidence + 0.1)
                })
                
            # Confidence calibration
            calibration_errors = [
                cal['calibration_error'] 
                for cal in profile.confidence_calibration.values()
            ]
            if calibration_errors and np.mean(calibration_errors) > 0.1:
                agent_recommendations.append({
                    'type': 'model_recalibration',
                    'priority': 'medium',
                    'action': 'recalibrate_confidence_scores',
                    'reason': 'Poor confidence calibration',
                    'calibration_data': profile.confidence_calibration
                })
                
            # Regime-specific recommendations
            for regime, performance in profile.performance_by_regime.items():
                if performance['count'] > 10 and performance['accuracy'] < 0.4:
                    agent_recommendations.append({
                        'type': 'regime_specific_training',
                        'priority': 'medium',
                        'action': f'retrain_for_{regime}',
                        'reason': f'Poor performance in {regime}: {performance["accuracy"]:.2%}',
                        'training_focus': regime
                    })
                    
            # Feature importance insights
            if profile.feature_importance:
                top_features = sorted(
                    profile.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                agent_recommendations.append({
                    'type': 'feature_engineering',
                    'priority': 'low',
                    'action': 'focus_on_top_features',
                    'reason': 'Optimize feature usage',
                    'top_features': dict(top_features)
                })
                
            # Risk management
            if profile.max_drawdown > 0.15:
                agent_recommendations.append({
                    'type': 'risk_management',
                    'priority': 'high',
                    'action': 'tighten_risk_controls',
                    'reason': f'High drawdown: {profile.max_drawdown:.2%}',
                    'suggested_actions': [
                        'reduce_position_size',
                        'implement_trailing_stops',
                        'add_volatility_filter'
                    ]
                })
                
            recommendations[agent_id] = agent_recommendations
            
        return recommendations
        
    async def _update_agent_profiles(
        self,
        agent_analysis: Dict[str, AgentPerformanceProfile]
    ):
        """Update agent profiles in database and memory"""
        for agent_id, profile in agent_analysis.items():
            # Update in memory
            self.agent_profiles[agent_id] = profile
            
            # Update in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO agent_profiles 
                        (agent_id, agent_type, performance_data, updated_at)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (agent_id) DO UPDATE
                        SET performance_data = EXCLUDED.performance_data,
                            updated_at = EXCLUDED.updated_at
                    """,
                    agent_id, profile.agent_type,
                    json.dumps(asdict(profile)), now_utc()
                    )
                    
    async def _train_agent_models(
        self,
        feedback_list: List[LearningFeedback]
    ) -> Dict[str, Dict[str, Any]]:
        """Train updated models for each agent"""
        training_results = {}
        
        # Group feedback by agent
        agent_feedback = defaultdict(list)
        
        for feedback in feedback_list:
            agent_feedback[feedback.agent_id].append(feedback)
            
        # Train each agent
        for agent_id, feedbacks in agent_feedback.items():
            if len(feedbacks) >= self.config['min_samples_for_update']:
                result = await self._train_single_agent(agent_id, feedbacks)
                training_results[agent_id] = result
                
        return training_results
        
    async def _train_single_agent(
        self,
        agent_id: str,
        feedbacks: List[LearningFeedback]
    ) -> Dict[str, Any]:
        """Train a single agent model"""
        try:
            # Prepare training data
            X, y, sample_weights = self._prepare_training_data(feedbacks)
            
            if len(X) < 10:
                return {'status': 'insufficient_data', 'samples': len(X)}
                
            # Get or create model
            if agent_id not in self.learning_models:
                self.learning_models[agent_id] = self._create_agent_model(agent_id)
                
            model = self.learning_models[agent_id]
            
            # Train with cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=min(5, len(X) // 10),
                scoring='accuracy',
                sample_weight=sample_weights
            )
            
            # Fit final model
            model.fit(X, y, sample_weight=sample_weights)
            
            # Calculate improvement
            profile = self.agent_profiles.get(agent_id)
            old_accuracy = profile.accuracy if profile else 0.5
            new_accuracy = scores.mean()
            improvement = new_accuracy - old_accuracy
            
            # Save model version
            await self._save_model_version(agent_id, model, {
                'accuracy': new_accuracy,
                'cv_scores': scores.tolist(),
                'improvement': improvement,
                'samples': len(X)
            })
            
            return {
                'status': 'success',
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'improvement': improvement,
                'cv_scores': scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training agent {agent_id}: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def _prepare_training_data(
        self,
        feedbacks: List[LearningFeedback]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from feedback"""
        features = []
        targets = []
        weights = []
        
        for feedback in feedbacks:
            # Combine all features
            feature_vector = []
            
            # Add indicator values
            for indicator in ['RSI', 'MACD', 'ATR']:
                value = feedback.indicators_at_signal.get(indicator, 0)
                feature_vector.append(value)
                
            # Add market regime as one-hot encoding
            regime_encoding = self._encode_market_regime(feedback.market_regime)
            feature_vector.extend(regime_encoding)
            
            # Add volatility
            feature_vector.append(feedback.volatility_level)
            
            # Add volume profile
            feature_vector.append(feedback.volume_profile.get('relative_volume', 1.0))
            
            features.append(feature_vector)
            
            # Target: 1 if profitable, 0 otherwise
            targets.append(1 if feedback.pnl > 0 else 0)
            
            # Weight by recency and importance
            recency_weight = np.exp(-0.01 * (now_utc() - feedback.signal_timestamp).days)
            importance_weight = abs(feedback.reward) + 1
            weights.append(recency_weight * importance_weight)
            
        X = np.array(features)
        y = np.array(targets)
        sample_weights = np.array(weights)
        
        # Normalize weights
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        return X, y, sample_weights
        
    def _encode_market_regime(self, regime: str) -> List[float]:
        """One-hot encode market regime"""
        regimes = [
            'uptrend_low_vol', 'uptrend_normal_vol', 'uptrend_high_vol',
            'sideways_low_vol', 'sideways_normal_vol', 'sideways_high_vol',
            'downtrend_low_vol', 'downtrend_normal_vol', 'downtrend_high_vol'
        ]
        
        encoding = [0.0] * len(regimes)
        if regime in regimes:
            encoding[regimes.index(regime)] = 1.0
            
        return encoding
        
    def _create_agent_model(self, agent_id: str):
        """Create a new model for an agent"""
        agent_type = self._infer_agent_type(agent_id)
        
        if agent_type == 'technical':
            # Gradient boosting for technical indicators
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif agent_type == 'sentiment':
            # Neural network for sentiment
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42
            )
        else:
            # Random forest as default
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
    async def _save_model_version(
        self,
        agent_id: str,
        model: Any,
        metrics: Dict[str, Any]
    ):
        """Save a model version"""
        if not self.db_pool:
            return
            
        # Get current version
        profile = self.agent_profiles.get(agent_id)
        version = profile.model_version + 1 if profile else 1
        
        # Serialize model
        model_bytes = pickle.dumps(model)
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_model_versions
                (agent_id, version, model_data, performance_metrics)
                VALUES ($1, $2, $3, $4)
            """,
            agent_id, version, model_bytes, json.dumps(metrics)
            )
            
        # Update profile version
        if profile:
            profile.model_version = version
            
    async def _extract_meta_insights(
        self,
        feedback_list: List[LearningFeedback],
        agent_analysis: Dict[str, AgentPerformanceProfile]
    ) -> Dict[str, Any]:
        """Extract meta-learning insights across all agents"""
        insights = {
            'cross_agent_patterns': [],
            'market_regime_insights': [],
            'feature_universality': {},
            'ensemble_opportunities': []
        }
        
        # Find patterns across agents
        successful_patterns = defaultdict(list)
        
        for feedback in feedback_list:
            if feedback.accuracy_contribution > 0:
                pattern_key = f"{feedback.market_regime}_{feedback.action}"
                successful_patterns[pattern_key].append({
                    'agent': feedback.agent_id,
                    'confidence': feedback.confidence,
                    'return': feedback.actual_return
                })
                
        # Identify universal successful patterns
        for pattern, instances in successful_patterns.items():
            if len(instances) > 5:
                unique_agents = set(inst['agent'] for inst in instances)
                if len(unique_agents) > 2:
                    insights['cross_agent_patterns'].append({
                        'pattern': pattern,
                        'agents': list(unique_agents),
                        'avg_return': np.mean([inst['return'] for inst in instances]),
                        'consistency': len(instances)
                    })
                    
        # Market regime insights
        regime_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for feedback in feedback_list:
            regime = feedback.market_regime
            regime_performance[regime]['total'] += 1
            if feedback.accuracy_contribution > 0:
                regime_performance[regime]['success'] += 1
                
        for regime, stats in regime_performance.items():
            if stats['total'] > 10:
                success_rate = stats['success'] / stats['total']
                insights['market_regime_insights'].append({
                    'regime': regime,
                    'success_rate': success_rate,
                    'sample_size': stats['total'],
                    'recommendation': 'increase_activity' if success_rate > 0.6 else 'reduce_activity'
                })
                
        # Feature universality
        all_features = defaultdict(list)
        
        for profile in agent_analysis.values():
            for feature, importance in profile.feature_importance.items():
                all_features[feature].append(importance)
                
        for feature, importances in all_features.items():
            if len(importances) > 2:
                insights['feature_universality'][feature] = {
                    'avg_importance': np.mean(importances),
                    'consistency': np.std(importances),
                    'agent_count': len(importances)
                }
                
        # Ensemble opportunities
        complementary_pairs = []
        
        for agent1, profile1 in agent_analysis.items():
            for agent2, profile2 in agent_analysis.items():
                if agent1 < agent2:  # Avoid duplicates
                    # Check if agents are complementary
                    regime_overlap = set(profile1.performance_by_regime.keys()) & \
                                   set(profile2.performance_by_regime.keys())
                    
                    complementary_score = 0
                    for regime in regime_overlap:
                        perf1 = profile1.performance_by_regime[regime]['accuracy']
                        perf2 = profile2.performance_by_regime[regime]['accuracy']
                        
                        # High score if one performs well where other doesn't
                        if (perf1 > 0.6 and perf2 < 0.4) or (perf1 < 0.4 and perf2 > 0.6):
                            complementary_score += 1
                            
                    if complementary_score > 2:
                        complementary_pairs.append({
                            'agents': [agent1, agent2],
                            'complementary_score': complementary_score,
                            'combined_accuracy': (profile1.accuracy + profile2.accuracy) / 2
                        })
                        
        insights['ensemble_opportunities'] = sorted(
            complementary_pairs,
            key=lambda x: x['complementary_score'],
            reverse=True
        )[:5]
        
        # Store insights
        await self._store_meta_insights(insights)
        
        return insights
        
    async def _store_feedback(self, feedback: LearningFeedback):
        """Store learning feedback"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO learning_feedback
                (trade_id, agent_id, signal_timestamp, feedback_data)
                VALUES ($1, $2, $3, $4)
            """,
            feedback.trade_id, feedback.agent_id,
            feedback.signal_timestamp, json.dumps(asdict(feedback))
            )
            
    async def _store_meta_insights(self, insights: Dict[str, Any]):
        """Store meta-learning insights"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            for insight_type, insight_data in insights.items():
                if isinstance(insight_data, list):
                    for item in insight_data:
                        await conn.execute("""
                            INSERT INTO meta_learning_insights
                            (insight_type, insight_data, confidence)
                            VALUES ($1, $2, $3)
                        """,
                        insight_type, json.dumps(item), 0.8
                        )
                        
    async def _calculate_improvement(self) -> Dict[str, float]:
        """Calculate overall system improvement"""
        if not self.performance_history:
            return {'overall': 0.0}
            
        # Compare recent performance to historical
        recent_performance = self.performance_history[-10:]
        historical_performance = self.performance_history[:-10]
        
        if not historical_performance:
            return {'overall': 0.0}
            
        improvements = {
            'accuracy': np.mean([p['accuracy'] for p in recent_performance]) - \
                       np.mean([p['accuracy'] for p in historical_performance]),
            'sharpe': np.mean([p['sharpe'] for p in recent_performance]) - \
                     np.mean([p['sharpe'] for p in historical_performance]),
            'overall': 0.0
        }
        
        improvements['overall'] = (improvements['accuracy'] + improvements['sharpe']) / 2
        
        return improvements
        
    async def _load_agent_profiles(self):
        """Load existing agent profiles from database"""
        if not self.db_pool:
            return
            
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT agent_id, performance_data
                FROM agent_profiles
            """)
            
            for row in rows:
                agent_id = row['agent_id']
                data = json.loads(row['performance_data'])
                
                # Convert back to AgentPerformanceProfile
                profile = AgentPerformanceProfile(**data)
                self.agent_profiles[agent_id] = profile
                
    def _init_meta_learner(self):
        """Initialize meta-learning model"""
        # This could be a more sophisticated model that learns
        # how to combine different agents
        self.meta_learner = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        
    async def get_agent_recommendations(self, agent_id: str) -> Dict[str, Any]:
        """Get current recommendations for a specific agent"""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return {'status': 'no_profile', 'recommendations': []}
            
        # Generate fresh recommendations based on current profile
        recommendations = []
        
        # Parameter adjustments
        if profile.accuracy < 0.55:
            recommendations.append({
                'type': 'parameter_tuning',
                'action': 'run_hyperparameter_optimization',
                'current_accuracy': profile.accuracy,
                'target_accuracy': 0.65,
                'suggested_method': 'optuna'
            })
            
        # Learning rate adjustment
        if profile.model_version > 5 and profile.accuracy < 0.6:
            recommendations.append({
                'type': 'learning_rate',
                'action': 'increase_exploration',
                'current_rate': profile.learning_rate,
                'suggested_rate': min(0.01, profile.learning_rate * 1.5)
            })
            
        return {
            'status': 'success',
            'agent_id': agent_id,
            'current_performance': {
                'accuracy': profile.accuracy,
                'sharpe_ratio': profile.sharpe_ratio,
                'total_signals': profile.total_signals
            },
            'recommendations': recommendations
        }
        
    async def close(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close() 