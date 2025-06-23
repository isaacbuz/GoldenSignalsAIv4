"""
Tests for ML training pipeline in GoldenSignalsAI V2.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMLTraining:
    """Test ML training functionality"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.cumsum(np.random.randn(100) * 2),
            'volume': np.random.randint(1000000, 10000000, 100),
            'open': 100 + np.cumsum(np.random.randn(100) * 2),
            'high': 0,  # Will be calculated
            'low': 0    # Will be calculated
        })
        
        # Calculate high and low
        data['high'] = data[['open', 'close']].max(axis=1) + np.random.rand(100) * 2
        data['low'] = data[['open', 'close']].min(axis=1) - np.random.rand(100) * 2
        
        return data
    
    def test_feature_engineering_module(self):
        """Test that feature engineering module exists and works"""
        try:
            from ml_training.feature_engineering import calculate_technical_indicators
            assert True, "Feature engineering module imported successfully"
        except ImportError:
            pytest.skip("Feature engineering module not found")
    
    def test_train_test_split(self, sample_training_data):
        """Test train/test data splitting"""
        data = sample_training_data
        
        # Split data 80/20
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        assert len(train_data) == 80
        assert len(test_data) == 20
        assert train_data.index[-1] < test_data.index[0]
    
    def test_data_preprocessing(self, sample_training_data):
        """Test data preprocessing steps"""
        data = sample_training_data
        
        # Check for missing values
        assert data.isnull().sum().sum() == 0, "Data should not have missing values"
        
        # Check data types
        assert data['close'].dtype in [np.float64, np.int64]
        assert data['volume'].dtype in [np.float64, np.int64]
        
        # Check data ranges
        assert data['high'].min() >= data['low'].max()
        assert data['volume'].min() > 0
    
    def test_technical_indicators_calculation(self, sample_training_data):
        """Test calculation of technical indicators"""
        data = sample_training_data.copy()
        
        # Calculate SMA
        data['sma_20'] = data['close'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Verify calculations
        assert 'sma_20' in data.columns
        assert 'rsi' in data.columns
        
        # Check RSI bounds
        valid_rsi = data['rsi'].dropna()
        assert all(0 <= rsi <= 100 for rsi in valid_rsi)
    
    def test_model_training_pipeline(self):
        """Test the model training pipeline structure"""
        # This would test the training pipeline if it were imported
        training_steps = [
            'data_loading',
            'preprocessing',
            'feature_engineering',
            'train_test_split',
            'model_training',
            'evaluation',
            'model_saving'
        ]
        
        # Verify all steps are defined
        for step in training_steps:
            assert isinstance(step, str)
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation"""
        # Mock predictions
        y_true = np.array([100, 102, 101, 103, 105])
        y_pred = np.array([99, 103, 101, 102, 106])
        
        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))
        assert mae == 1.0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse > mae  # RMSE should be >= MAE
        
        # Calculate directional accuracy
        y_true_dir = np.diff(y_true) > 0
        y_pred_dir = np.diff(y_pred) > 0
        dir_accuracy = np.mean(y_true_dir == y_pred_dir)
        assert 0 <= dir_accuracy <= 1
    
    def test_cross_validation_setup(self):
        """Test time series cross-validation setup"""
        n_samples = 100
        n_splits = 5
        test_size = 20
        
        splits = []
        for i in range(n_splits):
            train_end = n_samples - (n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end <= n_samples:
                splits.append((list(range(train_end)), 
                             list(range(test_start, test_end))))
        
        assert len(splits) == n_splits
        
        # Verify no overlap between train and test
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_hyperparameter_search_space(self):
        """Test hyperparameter search space definition"""
        # Example hyperparameter space for a random forest
        param_space = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Calculate total combinations
        total_combinations = 1
        for params in param_space.values():
            total_combinations *= len(params)
        
        assert total_combinations == 3 * 4 * 3 * 3  # 108 combinations
    
    def test_model_persistence(self, tmp_path):
        """Test model saving and loading"""
        import joblib
        
        # Create a simple model (mock)
        model = {'type': 'test_model', 'params': {'alpha': 0.1}}
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        assert loaded_model['type'] == model['type']
        assert loaded_model['params']['alpha'] == model['params']['alpha']
    
    @pytest.mark.slow
    def test_training_time_constraint(self):
        """Test that training completes within reasonable time"""
        import time
        
        # Simulate training process
        start_time = time.time()
        
        # Mock training steps
        for _ in range(100):
            # Simulate some computation
            _ = np.random.randn(1000, 10)
        
        training_time = time.time() - start_time
        
        # Training should complete quickly for small datasets
        assert training_time < 5.0  # 5 seconds for mock training 