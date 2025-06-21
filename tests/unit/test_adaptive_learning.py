"""
Unit tests for the Adaptive Learning System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.backtesting.adaptive_learning_system import (
    AdaptiveLearningSystem,
    AgentPerformanceProfile,
    LearningFeedback
)
from src.domain.backtesting.advanced_backtest_engine import (
    BacktestMetrics,
    BacktestTrade,
    BacktestSignal
)
from src.utils.timezone_utils import now_utc


@pytest.mark.unit
class TestAdaptiveLearningSystem:
    """Test cases for the Adaptive Learning System"""
    
    @pytest.fixture
    def learning_system(self):
        """Create a learning system instance"""
        config = {
            'learning_rate': 0.001,
            'min_samples_for_update': 5,
            'meta_learning_enabled': True,
            'online_learning': True
        }
        return AdaptiveLearningSystem(config)
    
    @pytest.fixture
    def mock_backtest_metrics(self):
        """Create mock backtest metrics"""
        return BacktestMetrics(
            total_return=0.15,
            annualized_return=0.18,
            benchmark_return=0.10,
            alpha=0.08,
            beta=0.95,
            volatility=0.20,
            downside_volatility=0.15,
            max_drawdown=0.12,
            max_drawdown_duration=30,
            var_95=0.02,
            cvar_95=0.03,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.2,
            information_ratio=0.8,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
            avg_win=150.0,
            avg_loss=80.0,
            profit_factor=2.25,
            expectancy=50.0,
            payoff_ratio=1.875,
            avg_trade_duration=timedelta(hours=4),
            trades_per_day=2.5,
            exposure_time=0.75
        )
    
    @pytest.fixture
    def mock_trades(self, sample_backtest_signal):
        """Create mock trades"""
        trades = []
        for i in range(10):
            trade = BacktestTrade(
                signal=sample_backtest_signal,
                entry_time=now_utc() - timedelta(hours=10-i),
                exit_time=now_utc() - timedelta(hours=9-i),
                exit_price=152.0 if i % 2 == 0 else 148.0,
                exit_reason='take_profit' if i % 2 == 0 else 'stop_loss',
                pnl=200.0 if i % 2 == 0 else -100.0,
                pnl_percent=0.02 if i % 2 == 0 else -0.01,
                commission=1.0,
                slippage=0.5,
                max_drawdown=0.01,
                max_profit=0.03,
                time_in_trade=timedelta(hours=1),
                market_regime='uptrend_normal_vol',
                volatility_at_entry=0.15
            )
            trades.append(trade)
        return trades
    
    @pytest.fixture
    def mock_market_data(self, sample_market_data):
        """Create mock market data"""
        return {
            'AAPL': sample_market_data,
            'GOOGL': sample_market_data.copy(),
            'MSFT': sample_market_data.copy()
        }
    
    @pytest.mark.asyncio
    async def test_initialization(self, learning_system):
        """Test learning system initialization"""
        with patch.object(learning_system, '_init_database') as mock_db:
            with patch.object(learning_system, '_init_redis') as mock_redis:
                with patch.object(learning_system, '_load_agent_profiles') as mock_load:
                    mock_db.return_value = None
                    mock_redis.return_value = None
                    mock_load.return_value = None
                    
                    await learning_system.initialize()
                    
                    assert mock_db.called
                    assert mock_redis.called
                    assert mock_load.called
    
    @pytest.mark.asyncio
    async def test_process_backtest_results(
        self, 
        learning_system, 
        mock_backtest_metrics, 
        mock_trades,
        mock_market_data
    ):
        """Test processing backtest results"""
        # Mock internal methods
        learning_system._generate_trade_feedback = AsyncMock(
            side_effect=[MagicMock() for _ in mock_trades]
        )
        learning_system._analyze_agent_performance = AsyncMock(
            return_value={'momentum_agent': MagicMock()}
        )
        learning_system._generate_learning_recommendations = AsyncMock(
            return_value={'momentum_agent': []}
        )
        learning_system._update_agent_profiles = AsyncMock()
        learning_system._train_agent_models = AsyncMock(return_value={})
        learning_system._extract_meta_insights = AsyncMock(return_value={})
        learning_system._calculate_improvement = AsyncMock(
            return_value={'overall': 0.05}
        )
        
        results = await learning_system.process_backtest_results(
            mock_backtest_metrics,
            mock_trades,
            mock_market_data
        )
        
        assert 'feedback_generated' in results
        assert 'agent_analysis' in results
        assert 'recommendations' in results
        assert 'performance_improvement' in results
        assert results['feedback_generated'] == len(mock_trades)
    
    def test_identify_market_regime(self, learning_system, sample_market_data):
        """Test market regime identification"""
        regime = learning_system._identify_market_regime(sample_market_data)
        
        assert regime in [
            'uptrend_low_vol', 'uptrend_normal_vol', 'uptrend_high_vol',
            'sideways_low_vol', 'sideways_normal_vol', 'sideways_high_vol',
            'downtrend_low_vol', 'downtrend_normal_vol', 'downtrend_high_vol',
            'unknown'
        ]
    
    def test_get_primary_agent(self, learning_system):
        """Test getting primary agent from scores"""
        agent_scores = {
            'momentum_agent': 0.9,
            'sentiment_agent': 0.7,
            'ml_agent': 0.8
        }
        
        primary = learning_system._get_primary_agent(agent_scores)
        assert primary == 'momentum_agent'
        
        # Test empty scores
        assert learning_system._get_primary_agent({}) == 'unknown'
    
    def test_encode_market_regime(self, learning_system):
        """Test market regime encoding"""
        encoding = learning_system._encode_market_regime('uptrend_normal_vol')
        
        assert len(encoding) == 9  # 3 trends x 3 volatility levels
        assert sum(encoding) == 1.0  # One-hot encoding
        assert encoding[4] == 1.0  # uptrend_normal_vol position
    
    def test_prepare_training_data(self, learning_system, sample_learning_feedback):
        """Test training data preparation"""
        feedbacks = [sample_learning_feedback] * 10
        
        X, y, weights = learning_system._prepare_training_data(feedbacks)
        
        assert X.shape[0] == 10
        assert y.shape[0] == 10
        assert weights.shape[0] == 10
        assert X.shape[1] > 10  # Features include indicators + regime encoding + others
        assert all(y[i] in [0, 1] for i in range(len(y)))
        assert np.isclose(weights.sum(), len(weights))  # Normalized weights
    
    def test_create_agent_model(self, learning_system):
        """Test agent model creation"""
        # Test technical agent
        model = learning_system._create_agent_model('momentum_agent')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Test sentiment agent
        model = learning_system._create_agent_model('sentiment_agent')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Test default
        model = learning_system._create_agent_model('unknown_agent')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    @pytest.mark.asyncio
    async def test_calculate_agent_profile(
        self, 
        learning_system,
        sample_learning_feedback
    ):
        """Test agent profile calculation"""
        feedbacks = []
        for i in range(20):
            feedback = sample_learning_feedback
            feedback.pnl = 100 if i % 2 == 0 else -50
            feedback.accuracy_contribution = 1.0 if i % 2 == 0 else 0.0
            feedbacks.append(feedback)
        
        profile = await learning_system._calculate_agent_profile(
            'momentum_agent',
            feedbacks
        )
        
        assert profile.agent_id == 'momentum_agent'
        assert profile.total_signals == 20
        assert profile.accurate_signals == 10
        assert profile.accuracy == 0.5
        assert profile.total_pnl == 500  # 10 * 100 - 10 * 50
        assert profile.avg_pnl_per_signal == 25
        assert 'uptrend_normal_vol' in profile.performance_by_regime
    
    @pytest.mark.asyncio
    async def test_generate_learning_recommendations(
        self, 
        learning_system,
        mock_backtest_metrics
    ):
        """Test recommendation generation"""
        # Create agent profiles with different performance levels
        profiles = {
            'low_accuracy_agent': AgentPerformanceProfile(
                agent_id='low_accuracy_agent',
                agent_type='technical',
                accuracy=0.4,
                max_drawdown=0.2,
                avg_confidence=0.7
            ),
            'good_agent': AgentPerformanceProfile(
                agent_id='good_agent',
                agent_type='ml',
                accuracy=0.7,
                max_drawdown=0.05,
                avg_confidence=0.75
            )
        }
        
        recommendations = await learning_system._generate_learning_recommendations(
            profiles,
            mock_backtest_metrics
        )
        
        # Low accuracy agent should get parameter adjustment recommendation
        assert 'low_accuracy_agent' in recommendations
        low_acc_recs = recommendations['low_accuracy_agent']
        assert any(r['type'] == 'parameter_adjustment' for r in low_acc_recs)
        assert any(r['type'] == 'risk_management' for r in low_acc_recs)
        
        # Good agent should have fewer recommendations
        assert 'good_agent' in recommendations
        good_recs = recommendations['good_agent']
        assert len(good_recs) < len(low_acc_recs)
    
    @pytest.mark.asyncio
    async def test_train_single_agent(self, learning_system, sample_learning_feedback):
        """Test training a single agent model"""
        feedbacks = [sample_learning_feedback] * 20  # Enough samples
        
        # Mock model creation and training
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        learning_system._create_agent_model = MagicMock(return_value=mock_model)
        learning_system._save_model_version = AsyncMock()
        
        with patch('sklearn.model_selection.cross_val_score', return_value=np.array([0.7, 0.72, 0.68, 0.71, 0.69])):
            result = await learning_system._train_single_agent('momentum_agent', feedbacks)
        
        assert result['status'] == 'success'
        assert 'old_accuracy' in result
        assert 'new_accuracy' in result
        assert 'improvement' in result
        assert result['new_accuracy'] == pytest.approx(0.7, 0.01)
        assert mock_model.fit.called
    
    @pytest.mark.asyncio
    async def test_extract_meta_insights(self, learning_system, sample_learning_feedback):
        """Test meta-learning insights extraction"""
        # Create diverse feedback data
        feedbacks = []
        agent_profiles = {}
        
        for agent_id in ['agent1', 'agent2', 'agent3']:
            profile = AgentPerformanceProfile(
                agent_id=agent_id,
                agent_type='technical',
                accuracy=0.6,
                performance_by_regime={
                    'uptrend_normal_vol': {'accuracy': 0.7 if agent_id == 'agent1' else 0.4},
                    'downtrend_high_vol': {'accuracy': 0.4 if agent_id == 'agent1' else 0.7}
                },
                feature_importance={'RSI': 0.3, 'MACD': 0.2}
            )
            agent_profiles[agent_id] = profile
            
            # Create feedbacks for pattern detection
            for i in range(10):
                feedback = LearningFeedback(
                    trade_id=f'{agent_id}_trade_{i}',
                    agent_id=agent_id,
                    signal_timestamp=now_utc(),
                    symbol='AAPL',
                    action='buy',
                    confidence=0.8,
                    predicted_return=0.03,
                    actual_return=0.025,
                    pnl=100,
                    holding_period=timedelta(hours=2),
                    exit_reason='take_profit',
                    market_regime='uptrend_normal_vol' if i % 2 == 0 else 'downtrend_high_vol',
                    volatility_level=0.15,
                    volume_profile={'relative_volume': 1.1},
                    features_at_signal={},
                    indicators_at_signal={},
                    accuracy_contribution=1.0 if i % 2 == 0 else 0.0,
                    sharpe_contribution=1.2,
                    reward=0.02,
                    regret=0.005,
                    surprise=0.005
                )
                feedbacks.append(feedback)
        
        learning_system._store_meta_insights = AsyncMock()
        
        insights = await learning_system._extract_meta_insights(feedbacks, agent_profiles)
        
        assert 'cross_agent_patterns' in insights
        assert 'market_regime_insights' in insights
        assert 'feature_universality' in insights
        assert 'ensemble_opportunities' in insights
        
        # Should find complementary agents (agent1 good in uptrend, agent2/3 good in downtrend)
        assert len(insights['ensemble_opportunities']) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_improvement(self, learning_system):
        """Test improvement calculation"""
        # No history
        improvement = await learning_system._calculate_improvement()
        assert improvement['overall'] == 0.0
        
        # Add performance history
        learning_system.performance_history = [
            {'accuracy': 0.5, 'sharpe': 1.0} for _ in range(15)
        ]
        learning_system.performance_history.extend([
            {'accuracy': 0.6, 'sharpe': 1.2} for _ in range(10)
        ])
        
        improvement = await learning_system._calculate_improvement()
        assert improvement['accuracy'] > 0
        assert improvement['sharpe'] > 0
        assert improvement['overall'] > 0
    
    @pytest.mark.asyncio
    async def test_get_agent_recommendations(self, learning_system):
        """Test getting recommendations for specific agent"""
        # No profile
        result = await learning_system.get_agent_recommendations('unknown_agent')
        assert result['status'] == 'no_profile'
        
        # Add profile with poor performance
        profile = AgentPerformanceProfile(
            agent_id='test_agent',
            agent_type='technical',
            accuracy=0.45,
            sharpe_ratio=0.8,
            total_signals=100,
            model_version=6,
            learning_rate=0.001
        )
        learning_system.agent_profiles['test_agent'] = profile
        
        result = await learning_system.get_agent_recommendations('test_agent')
        assert result['status'] == 'success'
        assert len(result['recommendations']) > 0
        
        # Should recommend parameter tuning for low accuracy
        param_tuning = [r for r in result['recommendations'] if r['type'] == 'parameter_tuning']
        assert len(param_tuning) > 0
        
        # Should recommend learning rate increase for poor performance after many versions
        lr_adjustment = [r for r in result['recommendations'] if r['type'] == 'learning_rate']
        assert len(lr_adjustment) > 0 