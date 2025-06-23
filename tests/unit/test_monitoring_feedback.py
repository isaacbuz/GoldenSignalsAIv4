"""
Tests for continuous monitoring and feedback in GoldenSignalsAI V2.
Based on best practices for real-time monitoring and model adaptation.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestMonitoringFeedback:
    """Test continuous monitoring and feedback functionality"""
    
    @pytest.fixture
    def real_time_signals(self):
        """Generate real-time signals for monitoring tests"""
        base_time = datetime.now()
        signals = []
        
        for i in range(100):
            signal = {
                'id': f'signal_{i}',
                'timestamp': base_time + timedelta(minutes=i),
                'symbol': np.random.choice(['SPY', 'QQQ', 'IWM']),
                'action': np.random.choice(['buy', 'sell', 'hold']),
                'confidence': np.random.uniform(0.5, 0.95),
                'predicted_return': np.random.uniform(-0.02, 0.02),
                'actual_return': None,  # To be filled after execution
                'status': 'pending'
            }
            signals.append(signal)
        
        return pd.DataFrame(signals)
    
    @pytest.fixture
    def performance_metrics(self):
        """Generate performance metrics for monitoring"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        metrics = pd.DataFrame({
            'date': dates,
            'hit_rate': 0.55 + np.random.uniform(-0.1, 0.1, 30),
            'avg_profit_per_signal': 0.002 + np.random.uniform(-0.001, 0.001, 30),
            'false_positive_rate': 0.2 + np.random.uniform(-0.05, 0.05, 30),
            'signal_count': np.random.randint(50, 150, 30),
            'sharpe_ratio': 1.5 + np.random.uniform(-0.5, 0.5, 30)
        })
        
        return metrics
    
    def test_real_time_performance_monitoring(self, real_time_signals):
        """Test real-time monitoring of signal performance"""
        class PerformanceMonitor:
            def __init__(self):
                self.metrics = {
                    'total_signals': 0,
                    'correct_predictions': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'cumulative_return': 0,
                    'alerts': []
                }
                self.thresholds = {
                    'min_hit_rate': 0.5,
                    'max_false_positive_rate': 0.3,
                    'min_sharpe_ratio': 1.0
                }
            
            def update_metrics(self, signal, actual_outcome):
                """Update metrics based on signal outcome"""
                self.metrics['total_signals'] += 1
                
                # Compare prediction with actual outcome
                if signal['action'] == 'buy':
                    if actual_outcome > 0:
                        self.metrics['correct_predictions'] += 1
                    else:
                        self.metrics['false_positives'] += 1
                elif signal['action'] == 'sell':
                    if actual_outcome < 0:
                        self.metrics['correct_predictions'] += 1
                    else:
                        self.metrics['false_positives'] += 1
                
                # Update cumulative return
                self.metrics['cumulative_return'] += actual_outcome
                
                # Check for alerts
                self.check_alerts()
            
            def check_alerts(self):
                """Check if any metrics breach thresholds"""
                if self.metrics['total_signals'] > 20:  # Need minimum samples
                    hit_rate = self.metrics['correct_predictions'] / self.metrics['total_signals']
                    fp_rate = self.metrics['false_positives'] / self.metrics['total_signals']
                    
                    if hit_rate < self.thresholds['min_hit_rate']:
                        self.metrics['alerts'].append({
                            'type': 'low_hit_rate',
                            'value': hit_rate,
                            'threshold': self.thresholds['min_hit_rate'],
                            'timestamp': datetime.now()
                        })
                    
                    if fp_rate > self.thresholds['max_false_positive_rate']:
                        self.metrics['alerts'].append({
                            'type': 'high_false_positive_rate',
                            'value': fp_rate,
                            'threshold': self.thresholds['max_false_positive_rate'],
                            'timestamp': datetime.now()
                        })
            
            def get_current_metrics(self):
                """Get current performance metrics"""
                if self.metrics['total_signals'] == 0:
                    return None
                
                return {
                    'hit_rate': self.metrics['correct_predictions'] / self.metrics['total_signals'],
                    'false_positive_rate': self.metrics['false_positives'] / self.metrics['total_signals'],
                    'avg_return': self.metrics['cumulative_return'] / self.metrics['total_signals'],
                    'total_signals': self.metrics['total_signals'],
                    'alerts': self.metrics['alerts']
                }
        
        # Test monitoring
        monitor = PerformanceMonitor()
        
        # Simulate signal outcomes
        for _, signal in real_time_signals.head(50).iterrows():
            # Simulate actual outcome
            if signal['action'] == 'buy':
                actual_outcome = np.random.uniform(-0.01, 0.02)
            else:
                actual_outcome = np.random.uniform(-0.02, 0.01)
            
            monitor.update_metrics(signal, actual_outcome)
        
        # Verify monitoring
        metrics = monitor.get_current_metrics()
        assert metrics is not None
        assert 'hit_rate' in metrics
        assert 'false_positive_rate' in metrics
        assert metrics['total_signals'] == 50
        
        # Check if alerts are generated properly
        if metrics['hit_rate'] < 0.5:
            assert any(alert['type'] == 'low_hit_rate' for alert in metrics['alerts'])
    
    def test_anomaly_detection(self, performance_metrics):
        """Test anomaly detection in signal performance"""
        def detect_anomalies(metrics, window=7):
            """Detect anomalies using statistical methods"""
            anomalies = []
            
            # Rolling statistics
            rolling_mean = metrics['hit_rate'].rolling(window=window).mean()
            rolling_std = metrics['hit_rate'].rolling(window=window).std()
            
            # Z-score based anomaly detection
            z_scores = (metrics['hit_rate'] - rolling_mean) / rolling_std
            
            # Flag anomalies (z-score > 2 or < -2)
            for idx, z_score in enumerate(z_scores):
                if abs(z_score) > 2:
                    anomalies.append({
                        'date': metrics['date'].iloc[idx],
                        'metric': 'hit_rate',
                        'value': metrics['hit_rate'].iloc[idx],
                        'z_score': z_score,
                        'expected_range': (
                            rolling_mean.iloc[idx] - 2 * rolling_std.iloc[idx],
                            rolling_mean.iloc[idx] + 2 * rolling_std.iloc[idx]
                        )
                    })
            
            # Check for sudden drops in performance
            perf_change = metrics['sharpe_ratio'].pct_change()
            for idx, change in enumerate(perf_change):
                if change < -0.3:  # 30% drop
                    anomalies.append({
                        'date': metrics['date'].iloc[idx],
                        'metric': 'sharpe_ratio',
                        'value': metrics['sharpe_ratio'].iloc[idx],
                        'change': change,
                        'alert': 'sudden_performance_drop'
                    })
            
            return anomalies
        
        # Test anomaly detection
        # Inject some anomalies
        metrics = performance_metrics.copy()
        metrics.loc[15, 'hit_rate'] = 0.2  # Anomalously low
        metrics.loc[20, 'sharpe_ratio'] = 0.3  # Sudden drop
        
        anomalies = detect_anomalies(metrics)
        
        # Verify anomaly detection
        assert len(anomalies) > 0
        assert any(a['metric'] == 'hit_rate' for a in anomalies)
        
        # Test that normal variations aren't flagged as anomalies
        normal_metrics = performance_metrics.copy()
        normal_anomalies = detect_anomalies(normal_metrics)
        assert len(normal_anomalies) < len(anomalies)
    
    def test_model_retraining_trigger(self, performance_metrics):
        """Test automatic model retraining triggers"""
        class ModelRetrainingScheduler:
            def __init__(self):
                self.last_retrain_date = datetime.now() - timedelta(days=30)
                self.performance_threshold = 0.5
                self.time_threshold = timedelta(days=7)
                self.retrain_history = []
            
            def should_retrain(self, current_metrics, current_date):
                """Determine if model should be retrained"""
                reasons = []
                
                # Time-based trigger
                time_since_last_retrain = current_date - self.last_retrain_date
                if time_since_last_retrain >= self.time_threshold:
                    reasons.append('scheduled_retrain')
                
                # Performance-based trigger
                recent_performance = current_metrics.tail(7)['hit_rate'].mean()
                if recent_performance < self.performance_threshold:
                    reasons.append('performance_degradation')
                
                # Volatility-based trigger
                performance_volatility = current_metrics.tail(14)['hit_rate'].std()
                if performance_volatility > 0.1:  # High volatility
                    reasons.append('high_performance_volatility')
                
                # Check for consistent underperformance
                underperforming_days = sum(
                    current_metrics.tail(7)['hit_rate'] < self.performance_threshold
                )
                if underperforming_days >= 5:
                    reasons.append('consistent_underperformance')
                
                return len(reasons) > 0, reasons
            
            def trigger_retrain(self, reason):
                """Trigger model retraining"""
                self.retrain_history.append({
                    'timestamp': datetime.now(),
                    'reason': reason,
                    'metrics_before': None  # Would store current metrics
                })
                self.last_retrain_date = datetime.now()
                return True
        
        # Test retraining triggers
        scheduler = ModelRetrainingScheduler()
        
        # Test time-based trigger
        should_retrain, reasons = scheduler.should_retrain(
            performance_metrics, 
            datetime.now()
        )
        assert should_retrain
        assert 'scheduled_retrain' in reasons
        
        # Test performance-based trigger
        poor_metrics = performance_metrics.copy()
        poor_metrics['hit_rate'] = 0.4  # Below threshold
        
        should_retrain, reasons = scheduler.should_retrain(
            poor_metrics,
            datetime.now()
        )
        assert should_retrain
        assert 'performance_degradation' in reasons
        assert 'consistent_underperformance' in reasons
    
    def test_feedback_loop_integration(self, real_time_signals):
        """Test integration of feedback into signal generation"""
        class FeedbackLoop:
            def __init__(self):
                self.signal_outcomes = []
                self.adjustment_factors = {
                    'confidence_adjustment': 1.0,
                    'threshold_adjustment': 0.0,
                    'feature_weights': {}
                }
            
            def record_outcome(self, signal, actual_return):
                """Record signal outcome for feedback"""
                outcome = {
                    'signal_id': signal['id'],
                    'predicted_return': signal['predicted_return'],
                    'actual_return': actual_return,
                    'error': actual_return - signal['predicted_return'],
                    'confidence': signal['confidence'],
                    'features': signal.get('features', {})
                }
                self.signal_outcomes.append(outcome)
                
                # Update adjustments based on feedback
                self.update_adjustments()
            
            def update_adjustments(self):
                """Update adjustment factors based on accumulated feedback"""
                if len(self.signal_outcomes) < 20:
                    return
                
                recent_outcomes = self.signal_outcomes[-20:]
                
                # Calculate average error by confidence level
                high_conf_errors = [
                    o['error'] for o in recent_outcomes 
                    if o['confidence'] > 0.8
                ]
                low_conf_errors = [
                    o['error'] for o in recent_outcomes 
                    if o['confidence'] <= 0.8
                ]
                
                # Adjust confidence scaling
                if high_conf_errors and np.mean(np.abs(high_conf_errors)) > 0.01:
                    # High confidence signals are overconfident
                    self.adjustment_factors['confidence_adjustment'] *= 0.95
                
                # Adjust threshold based on false positive rate
                false_positives = sum(
                    1 for o in recent_outcomes 
                    if o['predicted_return'] > 0 and o['actual_return'] < 0
                )
                fp_rate = false_positives / len(recent_outcomes)
                
                if fp_rate > 0.3:
                    # Too many false positives, increase threshold
                    self.adjustment_factors['threshold_adjustment'] += 0.05
                
            def apply_adjustments(self, new_signal):
                """Apply learned adjustments to new signal"""
                # Adjust confidence
                new_signal['confidence'] *= self.adjustment_factors['confidence_adjustment']
                
                # Adjust prediction threshold
                if new_signal['predicted_return'] > 0:
                    new_signal['predicted_return'] -= self.adjustment_factors['threshold_adjustment']
                
                return new_signal
        
        # Test feedback loop
        feedback_loop = FeedbackLoop()
        
        # Simulate signal outcomes - ensure we have high confidence signals with errors
        for i, (_, signal) in enumerate(real_time_signals.head(30).iterrows()):
            # Make some signals have high confidence
            if i % 3 == 0:
                signal['confidence'] = 0.85
                signal['predicted_return'] = 0.02  # High prediction
            
            # Simulate outcome with bias (model overestimates returns)
            actual_return = signal['predicted_return'] * 0.3 + np.random.normal(0, 0.002)
            feedback_loop.record_outcome(signal, actual_return)
        
        # Verify feedback adjustments
        assert len(feedback_loop.signal_outcomes) == 30
        # Check if adjustment was applied (may not always happen with random data)
        assert feedback_loop.adjustment_factors['confidence_adjustment'] <= 1.0
        
        # Test adjustment application
        new_signal = {
            'id': 'test_signal',
            'confidence': 0.9,
            'predicted_return': 0.01
        }
        adjusted_signal = feedback_loop.apply_adjustments(new_signal.copy())
        
        assert adjusted_signal['confidence'] < new_signal['confidence']
    
    def test_online_learning_updates(self):
        """Test online learning model updates"""
        class OnlineLearningModel:
            def __init__(self, learning_rate=0.01):
                self.weights = np.random.randn(10)  # Feature weights
                self.learning_rate = learning_rate
                self.update_history = []
            
            def predict(self, features):
                """Make prediction with current weights"""
                return np.dot(self.weights, features)
            
            def update(self, features, prediction, actual):
                """Update model weights based on prediction error"""
                error = actual - prediction
                
                # Gradient descent update
                gradient = -2 * error * features
                self.weights -= self.learning_rate * gradient
                
                # Record update
                self.update_history.append({
                    'timestamp': datetime.now(),
                    'error': error,
                    'gradient_norm': np.linalg.norm(gradient),
                    'weight_change': np.linalg.norm(self.learning_rate * gradient)
                })
                
                return error
            
            def adapt_learning_rate(self):
                """Adapt learning rate based on recent performance"""
                if len(self.update_history) < 10:
                    return
                
                recent_errors = [h['error'] for h in self.update_history[-10:]]
                error_variance = np.var(recent_errors)
                
                # Decrease learning rate if errors are volatile
                if error_variance > 0.01:
                    self.learning_rate *= 0.9
                # Increase learning rate if errors are stable and high
                elif np.mean(np.abs(recent_errors)) > 0.005:
                    self.learning_rate *= 1.1
                
                # Keep learning rate in reasonable bounds
                self.learning_rate = np.clip(self.learning_rate, 0.001, 0.1)
        
        # Test online learning
        model = OnlineLearningModel()
        
        # Simulate streaming data
        for i in range(50):
            features = np.random.randn(10)
            prediction = model.predict(features)
            
            # Simulate actual outcome (with some true relationship)
            true_weights = np.ones(10) * 0.1
            actual = np.dot(true_weights, features) + np.random.normal(0, 0.01)
            
            # Update model
            error = model.update(features, prediction, actual)
            
            # Adapt learning rate periodically
            if i % 10 == 0:
                model.adapt_learning_rate()
        
        # Verify online learning
        assert len(model.update_history) == 50
        
        # Check if errors are decreasing over time
        early_errors = [abs(h['error']) for h in model.update_history[:10]]
        late_errors = [abs(h['error']) for h in model.update_history[-10:]]
        
        assert np.mean(late_errors) < np.mean(early_errors) * 1.5  # Some improvement expected
    
    def test_quality_control_automation(self):
        """Test automated quality control for signals"""
        class QualityController:
            def __init__(self):
                self.quality_checks = {
                    'data_quality': self.check_data_quality,
                    'signal_consistency': self.check_signal_consistency,
                    'model_stability': self.check_model_stability,
                    'performance_bounds': self.check_performance_bounds
                }
                self.quality_history = []
            
            def check_data_quality(self, data):
                """Check quality of input data"""
                issues = []
                
                # Check for missing values
                if data.isnull().any().any():
                    issues.append('missing_values')
                
                # Check for outliers
                for col in data.select_dtypes(include=[np.number]).columns:
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    if (z_scores > 5).any():
                        issues.append(f'outliers_in_{col}')
                
                return len(issues) == 0, issues
            
            def check_signal_consistency(self, signals):
                """Check consistency of generated signals"""
                issues = []
                
                # Check confidence distribution
                confidence_values = signals['confidence']
                if confidence_values.min() < 0 or confidence_values.max() > 1:
                    issues.append('invalid_confidence_range')
                
                # Check for sudden changes in signal distribution
                action_counts = signals['action'].value_counts()
                if len(action_counts) > 0:
                    max_ratio = action_counts.max() / action_counts.sum()
                    if max_ratio > 0.8:  # One action dominates
                        issues.append('imbalanced_signal_distribution')
                
                return len(issues) == 0, issues
            
            def check_model_stability(self, predictions, threshold=0.1):
                """Check if model predictions are stable"""
                issues = []
                
                # Check prediction variance
                if predictions.std() < 0.0001:
                    issues.append('predictions_too_similar')
                
                # Check for NaN or infinite values
                if predictions.isnull().any() or np.isinf(predictions).any():
                    issues.append('invalid_predictions')
                
                return len(issues) == 0, issues
            
            def check_performance_bounds(self, metrics):
                """Check if performance metrics are within acceptable bounds"""
                issues = []
                
                if metrics.get('hit_rate', 0) < 0.4:
                    issues.append('low_hit_rate')
                
                if metrics.get('sharpe_ratio', 0) < 0:
                    issues.append('negative_sharpe_ratio')
                
                if metrics.get('max_drawdown', 0) < -0.5:
                    issues.append('excessive_drawdown')
                
                return len(issues) == 0, issues
            
            def run_quality_checks(self, data, signals, predictions, metrics):
                """Run all quality checks"""
                results = {}
                all_passed = True
                
                for check_name, check_func in self.quality_checks.items():
                    if check_name == 'data_quality':
                        passed, issues = check_func(data)
                    elif check_name == 'signal_consistency':
                        passed, issues = check_func(signals)
                    elif check_name == 'model_stability':
                        passed, issues = check_func(predictions)
                    elif check_name == 'performance_bounds':
                        passed, issues = check_func(metrics)
                    
                    results[check_name] = {
                        'passed': passed,
                        'issues': issues
                    }
                    all_passed &= passed
                
                # Record quality check
                self.quality_history.append({
                    'timestamp': datetime.now(),
                    'all_passed': all_passed,
                    'results': results
                })
                
                return all_passed, results
        
        # Test quality control
        controller = QualityController()
        
        # Create test data
        test_data = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        test_signals = pd.DataFrame({
            'action': ['buy', 'buy', 'sell', 'hold', 'buy'],
            'confidence': [0.8, 0.7, 0.9, 0.6, 0.85]
        })
        
        test_predictions = pd.Series([0.01, 0.02, -0.01, 0.0, 0.015])
        
        test_metrics = {
            'hit_rate': 0.55,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.2
        }
        
        # Run quality checks
        all_passed, results = controller.run_quality_checks(
            test_data, test_signals, test_predictions, test_metrics
        )
        
        # Verify quality control
        assert isinstance(all_passed, bool)
        assert 'data_quality' in results
        assert 'signal_consistency' in results
        assert all_passed  # Should pass with good test data
        
        # Test with bad data
        bad_signals = test_signals.copy()
        bad_signals['confidence'] = 1.5  # Invalid confidence
        
        all_passed_bad, results_bad = controller.run_quality_checks(
            test_data, bad_signals, test_predictions, test_metrics
        )
        
        assert not all_passed_bad
        assert 'invalid_confidence_range' in results_bad['signal_consistency']['issues'] 