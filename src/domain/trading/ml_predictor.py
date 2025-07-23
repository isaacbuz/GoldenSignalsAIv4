"""
Machine learning predictor module leveraging multiple frameworks.
Integrates scikit-learn, XGBoost, LightGBM, and Keras for market predictions.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(
        self,
        lookback_period: int = 20,
        prediction_horizon: int = 5,
        feature_engineering: bool = True,
        ensemble_voting: bool = True
    ):
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.feature_engineering = feature_engineering
        self.ensemble_voting = ensemble_voting
        self.models = {}
        self.scalers = {}
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for ML models."""
        try:
            features = []
            labels = []
            
            # Basic price features
            close_prices = data['close'].values
            volumes = data['volume'].values if 'volume' in data else np.ones_like(close_prices)
            
            for i in range(len(close_prices) - self.lookback_period - self.prediction_horizon):
                # Price sequence
                price_seq = close_prices[i:i+self.lookback_period]
                volume_seq = volumes[i:i+self.lookback_period]
                
                # Calculate returns
                returns = np.diff(price_seq) / price_seq[:-1]
                
                if self.feature_engineering:
                    # Technical indicators
                    sma = np.mean(price_seq)
                    std = np.std(returns)
                    upper_band = sma + 2 * std
                    lower_band = sma - 2 * std
                    rsi = self._calculate_rsi(price_seq)
                    
                    # Volume features
                    vol_sma = np.mean(volume_seq)
                    vol_std = np.std(volume_seq)
                    
                    features.append(np.concatenate([
                        returns,
                        [(price_seq[-1] - sma) / sma],
                        [(price_seq[-1] - upper_band) / price_seq[-1]],
                        [(price_seq[-1] - lower_band) / price_seq[-1]],
                        [rsi],
                        [(volume_seq[-1] - vol_sma) / vol_sma],
                        [vol_std / vol_sma]
                    ]))
                else:
                    features.append(returns)
                
                # Future return (label)
                future_return = (close_prices[i+self.lookback_period+self.prediction_horizon] - 
                               close_prices[i+self.lookback_period]) / close_prices[i+self.lookback_period]
                labels.append(1 if future_return > 0 else 0)
                
            return np.array(features), np.array(labels)
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            return np.array([]), np.array([])
            
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI for feature engineering."""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {str(e)}")
            return 50.0
            
    def train_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Train multiple ML models and create an ensemble."""
        try:
            X, y = self.prepare_features(data)
            if len(X) == 0 or len(y) == 0:
                return {}
                
            # Split data
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, val_idx = list(tscv.split(X))[-1]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[symbol] = scaler
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            # Train LightGBM
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
            lgb_model.fit(X_train_scaled, y_train)
            
            # Train Neural Network
            nn_model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Store models
            self.models[symbol] = {
                'xgb': xgb_model,
                'lgb': lgb_model,
                'nn': nn_model
            }
            
            # Calculate validation metrics
            metrics = {}
            for name, model in self.models[symbol].items():
                if name == 'nn':
                    pred = (model.predict(X_val_scaled) > 0.5).astype(int)
                else:
                    pred = model.predict(X_val_scaled)
                accuracy = np.mean(pred == y_val)
                metrics[f'{name}_accuracy'] = accuracy
                
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {}
            
    def predict(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Generate predictions using the ensemble of models."""
        try:
            if symbol not in self.models or symbol not in self.scalers:
                return {}
                
            # Prepare features
            X, _ = self.prepare_features(data.iloc[-self.lookback_period-1:])
            if len(X) == 0:
                return {}
                
            # Scale features
            X_scaled = self.scalers[symbol].transform(X)
            
            # Get predictions from each model
            predictions = {}
            probabilities = []
            
            for name, model in self.models[symbol].items():
                if name == 'nn':
                    prob = float(model.predict(X_scaled)[-1])
                else:
                    prob = float(model.predict_proba(X_scaled)[-1][1])
                predictions[f'{name}_probability'] = prob
                probabilities.append(prob)
                
            # Ensemble prediction
            if self.ensemble_voting:
                predictions['ensemble_probability'] = np.mean(probabilities)
                predictions['ensemble_vote'] = 1 if predictions['ensemble_probability'] > 0.5 else 0
                
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {}
            
    def save_models(self, path: str, symbol: str):
        """Save trained models to disk."""
        try:
            if symbol not in self.models:
                return
                
            # Save individual models
            for name, model in self.models[symbol].items():
                if name == 'nn':
                    model.save(f"{path}/{symbol}_{name}_model")
                else:
                    joblib.dump(model, f"{path}/{symbol}_{name}_model.joblib")
                    
            # Save scaler
            joblib.dump(self.scalers[symbol], f"{path}/{symbol}_scaler.joblib")
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            
    def load_models(self, path: str, symbol: str):
        """Load trained models from disk."""
        try:
            self.models[symbol] = {}
            
            # Load individual models
            self.models[symbol]['xgb'] = joblib.load(f"{path}/{symbol}_xgb_model.joblib")
            self.models[symbol]['lgb'] = joblib.load(f"{path}/{symbol}_lgb_model.joblib")
            self.models[symbol]['nn'] = keras.models.load_model(f"{path}/{symbol}_nn_model")
            
            # Load scaler
            self.scalers[symbol] = joblib.load(f"{path}/{symbol}_scaler.joblib")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}") 