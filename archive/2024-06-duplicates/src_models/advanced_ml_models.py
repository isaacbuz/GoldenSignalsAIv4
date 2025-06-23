"""
Advanced ML Models for GoldenSignalsAI
Comprehensive implementation of cutting-edge financial ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelPrediction:
    """Standardized model prediction output"""
    forecast: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    feature_importance: Optional[Dict[str, float]]
    model_name: str
    metadata: Dict[str, Any]


class ARIMAGARCHModel:
    """ARIMA-GARCH hybrid for volatility forecasting"""
    
    def __init__(self, arima_order=(1,1,1), garch_order=(1,1)):
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.arima_model = None
        self.garch_model = None
    
    def fit(self, returns: pd.Series):
        """Fit ARIMA-GARCH model"""
        # Fit ARIMA
        self.arima_model = ARIMA(returns, order=self.arima_order)
        self.arima_fit = self.arima_model.fit()
        
        # Fit GARCH on residuals
        self.garch_model = arch_model(
            self.arima_fit.resid, 
            vol='Garch', 
            p=self.garch_order[0], 
            q=self.garch_order[1]
        )
        self.garch_fit = self.garch_model.fit(disp='off')
    
    def predict(self, horizon: int = 5) -> ModelPrediction:
        """Generate volatility forecast"""
        # ARIMA forecast
        arima_forecast = self.arima_fit.forecast(steps=horizon)
        
        # GARCH volatility forecast
        garch_forecast = self.garch_fit.forecast(horizon=horizon)
        volatility = np.sqrt(garch_forecast.variance.values[-1, :])
        
        # Combine forecasts
        forecast = arima_forecast.values
        upper_bound = forecast + 2 * volatility
        lower_bound = forecast - 2 * volatility
        
        return ModelPrediction(
            forecast=forecast,
            confidence_intervals=(lower_bound, upper_bound),
            feature_importance=None,
            model_name="ARIMA-GARCH",
            metadata={
                "volatility_forecast": volatility,
                "arima_params": self.arima_fit.params.to_dict(),
                "garch_params": self.garch_fit.params.to_dict()
            }
        )


class ProphetModel:
    """Facebook Prophet with custom regressors"""
    
    def __init__(self, seasonality_mode='multiplicative'):
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.regressors = []
    
    def add_regressor(self, name: str):
        """Add external regressor"""
        self.model.add_regressor(name)
        self.regressors.append(name)
    
    def fit(self, df: pd.DataFrame):
        """Fit Prophet model
        df should have columns: ds (datetime), y (target), and any regressors
        """
        self.model.fit(df)
    
    def predict(self, periods: int = 30) -> ModelPrediction:
        """Generate forecast"""
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add regressor values for future dates (would need actual implementation)
        for regressor in self.regressors:
            future[regressor] = 0  # Placeholder
        
        forecast = self.model.predict(future)
        
        return ModelPrediction(
            forecast=forecast['yhat'].tail(periods).values,
            confidence_intervals=(
                forecast['yhat_lower'].tail(periods).values,
                forecast['yhat_upper'].tail(periods).values
            ),
            feature_importance=None,
            model_name="Prophet",
            metadata={
                "trend": forecast['trend'].tail(periods).values,
                "seasonality": forecast[['yearly', 'weekly', 'daily']].tail(periods).to_dict()
            }
        )


class LSTMPricePredictor(nn.Module):
    """LSTM network for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Final prediction
        out = self.fc(attn_out[:, -1, :])
        return out


class TransformerPredictor(nn.Module):
    """Transformer model for financial time series"""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Take last token for prediction
        output = self.output_projection(x[:, -1, :])
        return output


class LightGBMSharpe:
    """LightGBM with custom Sharpe ratio objective"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.model = None
        self.feature_names = None
    
    def sharpe_objective(self, y_true, y_pred):
        """Custom objective to maximize Sharpe ratio"""
        y_true = y_true.get_label()
        
        # Calculate returns
        returns = y_pred - y_true
        
        # Sharpe ratio components
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-6  # Avoid division by zero
        
        # Gradient and Hessian for Sharpe ratio
        n = len(returns)
        grad = (1/std_return) - (mean_return * (returns - mean_return)) / (n * std_return**3)
        hess = np.ones_like(grad) * 0.001  # Simplified Hessian
        
        return grad, hess
    
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        """Train the model"""
        self.feature_names = X.columns.tolist()
        
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': self.sharpe_objective,
            'metric': 'None',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
    
    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Generate predictions"""
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Feature importance
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importance(importance_type='gain')
        ))
        
        return ModelPrediction(
            forecast=predictions,
            confidence_intervals=None,
            feature_importance=importance,
            model_name="LightGBM-Sharpe",
            metadata={"best_iteration": self.model.best_iteration}
        )


class StackedEnsemble:
    """Stacked generalization ensemble"""
    
    def __init__(self):
        # Base models
        self.base_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        # Meta learner
        self.meta_learner = xgb.XGBRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train ensemble"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base models and collect predictions
        base_predictions = []
        
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
            pred = model.predict(X_scaled)
            base_predictions.append(pred)
        
        # Train meta learner on base model predictions
        meta_features = np.column_stack(base_predictions)
        self.meta_learner.fit(meta_features, y)
    
    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Generate ensemble predictions"""
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X_scaled)
            base_predictions.append(pred)
        
        # Meta learner prediction
        meta_features = np.column_stack(base_predictions)
        final_prediction = self.meta_learner.predict(meta_features)
        
        return ModelPrediction(
            forecast=final_prediction,
            confidence_intervals=None,
            feature_importance=None,
            model_name="Stacked-Ensemble",
            metadata={
                "base_predictions": {
                    name: pred.tolist() 
                    for name, pred in zip(self.base_models.keys(), base_predictions)
                }
            }
        )


class HiddenMarkovRegime:
    """Hidden Markov Model for regime detection"""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
    
    def fit(self, returns: pd.Series):
        """Fit HMM to identify market regimes"""
        from hmmlearn import hmm
        
        # Prepare data
        X = returns.values.reshape(-1, 1)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100
        )
        self.model.fit(X)
    
    def predict_regime(self, returns: pd.Series) -> Dict[str, Any]:
        """Predict current market regime"""
        X = returns.values.reshape(-1, 1)
        
        # Predict states
        states = self.model.predict(X)
        current_state = states[-1]
        
        # Get regime characteristics
        means = self.model.means_.flatten()
        covars = self.model.covars_.diagonal()
        
        regime_map = {
            np.argmin(means): "Bear",
            np.argmax(means): "Bull",
        }
        
        # Set remaining state as "Neutral"
        for i in range(self.n_states):
            if i not in regime_map:
                regime_map[i] = "Neutral"
        
        return {
            "current_regime": regime_map[current_state],
            "regime_probabilities": self.model.predict_proba(X[-1:]).flatten(),
            "regime_characteristics": {
                regime_map[i]: {
                    "mean_return": means[i],
                    "volatility": np.sqrt(covars[i])
                }
                for i in range(self.n_states)
            }
        }


class ModelFactory:
    """Factory class to create and manage models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Create model instance"""
        models = {
            'arima_garch': ARIMAGARCHModel,
            'prophet': ProphetModel,
            'lstm': LSTMPricePredictor,
            'transformer': TransformerPredictor,
            'lgb_sharpe': LightGBMSharpe,
            'stacked': StackedEnsemble,
            'hmm_regime': HiddenMarkovRegime
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type](**kwargs)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
    
    # Test ARIMA-GARCH
    arima_garch = ModelFactory.create_model('arima_garch')
    arima_garch.fit(returns)
    prediction = arima_garch.predict(horizon=5)
    print(f"ARIMA-GARCH forecast: {prediction.forecast}")
    
    # Test HMM Regime
    hmm = ModelFactory.create_model('hmm_regime', n_states=3)
    hmm.fit(returns)
    regime = hmm.predict_regime(returns)
    print(f"Current market regime: {regime['current_regime']}") 