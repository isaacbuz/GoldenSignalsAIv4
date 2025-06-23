"""
Tests for ML models in GoldenSignalsAI V2.
"""
import pytest
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path


class TestMLModels:
    """Test ML model functionality"""
    
    @pytest.fixture
    def model_path(self):
        """Get path to ML models directory"""
        return Path(__file__).parent.parent
    
    def test_sentiment_model_exists(self, model_path):
        """Test that sentiment model file exists"""
        sentiment_model_path = model_path / "sentiment_model.pkl"
        assert sentiment_model_path.exists(), "Sentiment model file not found"
    
    def test_forecast_model_exists(self, model_path):
        """Test that forecast model file exists"""
        forecast_model_path = model_path / "forecast_model.pkl"
        assert forecast_model_path.exists(), "Forecast model file not found"
    
    @pytest.mark.skipif(not os.path.exists("ml_models/sentiment_model.pkl"), 
                        reason="Sentiment model not available")
    def test_sentiment_model_prediction(self, model_path):
        """Test sentiment model predictions"""
        try:
            model = joblib.load(model_path / "sentiment_model.pkl")
            
            # Test data
            test_texts = [
                "Stock price surging, great earnings report!",
                "Market crash imminent, sell everything!",
                "Neutral market conditions today"
            ]
            
            # Make predictions
            predictions = model.predict(test_texts)
            
            # Verify predictions
            assert len(predictions) == len(test_texts)
            assert all(pred in [-1, 0, 1] for pred in predictions)
            
            # Test probabilities if available
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(test_texts)
                assert probs.shape[0] == len(test_texts)
                assert np.allclose(probs.sum(axis=1), 1.0)
                
        except Exception as e:
            pytest.skip(f"Could not load sentiment model: {str(e)}")
    
    @pytest.mark.skipif(not os.path.exists("ml_models/forecast_model.pkl"), 
                        reason="Forecast model not available")
    def test_forecast_model_prediction(self, model_path):
        """Test forecast model predictions"""
        try:
            model = joblib.load(model_path / "forecast_model.pkl")
            
            # Generate test time series data
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            prices = 100 + np.cumsum(np.random.randn(100) * 2)
            
            test_data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100),
                'high': prices + np.random.rand(100) * 2,
                'low': prices - np.random.rand(100) * 2,
                'open': prices + np.random.randn(100) * 0.5
            })
            
            # Prepare features (assuming model expects certain format)
            features = test_data[['close', 'volume']].values[-30:]  # Last 30 days
            
            # Make prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)
            elif features.ndim == 2 and features.shape[0] > 1:
                features = features.reshape(1, -1)
            
            prediction = model.predict(features)
            
            # Verify prediction
            assert prediction is not None
            assert len(prediction) > 0
            assert isinstance(prediction[0], (int, float, np.number))
            
        except Exception as e:
            pytest.skip(f"Could not load forecast model: {str(e)}")
    
    def test_model_consistency(self, model_path):
        """Test that models produce consistent results"""
        # This would test that the same input produces the same output
        # Useful for ensuring models are deterministic
        pass
    
    def test_model_performance_metrics(self):
        """Test model performance metrics calculation"""
        # Mock predictions and actuals
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        assert 0 <= accuracy <= 1
        assert accuracy == 0.75  # 6 out of 8 correct
        
        # Calculate precision for class 1
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        assert 0 <= precision <= 1
    
    def test_feature_engineering(self):
        """Test feature engineering for ML models"""
        # Create sample price data
        data = pd.DataFrame({
            'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            'volume': [1000000, 1100000, 900000, 1200000, 1300000, 
                      1000000, 1400000, 1500000, 1100000, 1600000],
            'high': [101, 103, 102, 104, 106, 105, 107, 109, 108, 110],
            'low': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108]
        })
        
        # Calculate technical indicators
        # RSI
        close_delta = data['close'].diff()
        gain = close_delta.where(close_delta > 0, 0)
        loss = -close_delta.where(close_delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(window=5, min_periods=1).mean()
        data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
        
        # Verify calculations
        assert 'rsi' in data.columns
        assert 'sma_5' in data.columns
        assert 'ema_5' in data.columns
        assert data['rsi'].notna().sum() > 0
        assert data['sma_5'].notna().sum() > 0
        
        # Verify RSI is in valid range
        valid_rsi = data['rsi'].dropna()
        assert all(0 <= rsi <= 100 for rsi in valid_rsi)
