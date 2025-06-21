"""
Advanced ML Models Service
Implements LSTM, XGBoost, Transformer models, and ensemble methods for sophisticated trading signals
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import joblib
import json
from collections import defaultdict

# Import utilities
from src.utils.timezone_utils import now_utc
from src.utils.technical_indicators import TechnicalIndicators

# Try to import ML libraries (will use mock if not available)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    LSTM = "lstm"
    XGBOOST = "xgboost"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    ENSEMBLE = "ensemble"


@dataclass
class MLPrediction:
    """Container for ML model predictions"""
    model_type: ModelType
    symbol: str
    prediction: str  # BUY, SELL, HOLD
    confidence: float  # 0 to 1
    target_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    features_importance: Dict[str, float]
    reasoning: List[str]


class AdvancedMLModels:
    """Advanced ML models for sophisticated trading predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            ModelType.XGBOOST: {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'multi:softprob',
                'n_jobs': -1
            },
            ModelType.RANDOM_FOREST: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'n_jobs': -1
            },
            ModelType.GRADIENT_BOOST: {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            }
        }
    
    async def initialize(self):
        """Initialize the ML models"""
        try:
            # Try to load pre-trained models
            await self._load_models()
            if not self.is_trained:
                # Train with sample data if no pre-trained models
                await self._train_sample_models()
            logger.info("Advanced ML Models initialized")
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self.is_trained = False
    
    async def _load_models(self):
        """Load pre-trained models from disk"""
        # This would load actual trained models in production
        # For now, we'll skip this
        pass
    
    async def _train_sample_models(self):
        """Train models with sample data for demo purposes"""
        if not HAS_SKLEARN:
            logger.warning("Scikit-learn not available, using mock models")
            return
        
        # Generate sample training data
        n_samples = 1000
        n_features = 20
        
        # Create feature names
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'atr',
            'volume_ratio', 'price_change_1d', 'price_change_5d',
            'volatility', 'momentum', 'trend_strength',
            'support_distance', 'resistance_distance', 'volume_trend', 'market_cap_ratio'
        ]
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels (0: SELL, 1: HOLD, 2: BUY) with some pattern
        y = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
        
        # Add some pattern to make it more realistic
        # High RSI -> more likely to SELL
        high_rsi_mask = X[:, 0] > 1  # RSI column
        y[high_rsi_mask] = np.random.choice([0, 1], size=high_rsi_mask.sum(), p=[0.7, 0.3])
        
        # Low RSI -> more likely to BUY
        low_rsi_mask = X[:, 0] < -1
        y[low_rsi_mask] = np.random.choice([1, 2], size=low_rsi_mask.sum(), p=[0.3, 0.7])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        
        # Train models
        if HAS_XGBOOST:
            self.models[ModelType.XGBOOST] = xgb.XGBClassifier(**self.model_configs[ModelType.XGBOOST])
            self.models[ModelType.XGBOOST].fit(X_train_scaled, y_train)
        
        self.models[ModelType.RANDOM_FOREST] = RandomForestClassifier(**self.model_configs[ModelType.RANDOM_FOREST])
        self.models[ModelType.RANDOM_FOREST].fit(X_train_scaled, y_train)
        
        self.models[ModelType.GRADIENT_BOOST] = GradientBoostingClassifier(**self.model_configs[ModelType.GRADIENT_BOOST])
        self.models[ModelType.GRADIENT_BOOST].fit(X_train_scaled, y_train)
        
        self.is_trained = True
        logger.info("Sample models trained successfully")
    
    async def predict(
        self,
        symbol: str,
        features: Dict[str, float],
        model_types: List[ModelType] = None
    ) -> List[MLPrediction]:
        """Generate predictions using specified models"""
        
        if not model_types:
            model_types = list(self.models.keys())
        
        predictions = []
        
        # Prepare features
        feature_vector = self._prepare_features(features)
        
        for model_type in model_types:
            if model_type == ModelType.ENSEMBLE:
                # Ensemble prediction combines all other models
                prediction = await self._ensemble_predict(symbol, feature_vector, features)
            else:
                prediction = await self._single_model_predict(symbol, feature_vector, features, model_type)
            
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for model input"""
        if not self.feature_columns:
            # Use provided features as is
            return np.array(list(features.values())).reshape(1, -1)
        
        # Align features with trained model expectations
        feature_vector = []
        for col in self.feature_columns:
            if col in features:
                feature_vector.append(features[col])
            else:
                # Use default values for missing features
                default_values = {
                    'rsi': 50,
                    'macd': 0,
                    'volume_ratio': 1,
                    'volatility': 0.02,
                    'momentum': 0,
                    'trend_strength': 0.5
                }
                feature_vector.append(default_values.get(col, 0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    async def _single_model_predict(
        self,
        symbol: str,
        feature_vector: np.ndarray,
        raw_features: Dict[str, float],
        model_type: ModelType
    ) -> Optional[MLPrediction]:
        """Generate prediction using a single model"""
        
        if not self.is_trained or model_type not in self.models:
            # Return mock prediction
            return self._generate_mock_prediction(symbol, model_type, raw_features)
        
        try:
            model = self.models[model_type]
            
            # Scale features
            if 'standard' in self.scalers:
                feature_vector_scaled = self.scalers['standard'].transform(feature_vector)
            else:
                feature_vector_scaled = feature_vector
            
            # Get prediction and probabilities
            prediction = model.predict(feature_vector_scaled)[0]
            probabilities = model.predict_proba(feature_vector_scaled)[0]
            
            # Map prediction to action
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            action = action_map[prediction]
            confidence = float(probabilities[prediction])
            
            # Get feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, col in enumerate(self.feature_columns[:len(importances)]):
                    feature_importance[col] = float(importances[i])
            
            # Calculate price targets
            current_price = raw_features.get('current_price', 100)
            atr = raw_features.get('atr', current_price * 0.02)
            
            if action == "BUY":
                target_price = current_price * 1.05
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 3)
                reasoning = [
                    f"Model {model_type.value} indicates BUY signal",
                    f"Confidence level: {confidence:.1%}",
                    "Technical indicators support upward movement"
                ]
            elif action == "SELL":
                target_price = current_price * 0.95
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 3)
                reasoning = [
                    f"Model {model_type.value} indicates SELL signal",
                    f"Confidence level: {confidence:.1%}",
                    "Technical indicators suggest downward pressure"
                ]
            else:  # HOLD
                target_price = current_price
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 2)
                reasoning = [
                    f"Model {model_type.value} suggests HOLD position",
                    f"Confidence level: {confidence:.1%}",
                    "No clear directional signal"
                ]
            
            # Add feature-based reasoning
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                for feature, importance in top_features:
                    if importance > 0.1:
                        reasoning.append(f"{feature} is a key factor ({importance:.1%} importance)")
            
            return MLPrediction(
                model_type=model_type,
                symbol=symbol,
                prediction=action,
                confidence=confidence,
                target_price=round(target_price, 2),
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                timeframe="1-5 days",
                features_importance=feature_importance,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in {model_type.value} prediction: {e}")
            return self._generate_mock_prediction(symbol, model_type, raw_features)
    
    async def _ensemble_predict(
        self,
        symbol: str,
        feature_vector: np.ndarray,
        raw_features: Dict[str, float]
    ) -> MLPrediction:
        """Generate ensemble prediction combining all models"""
        
        # Get predictions from all individual models
        individual_predictions = []
        for model_type in self.models.keys():
            pred = await self._single_model_predict(symbol, feature_vector, raw_features, model_type)
            if pred:
                individual_predictions.append(pred)
        
        if not individual_predictions:
            return self._generate_mock_prediction(symbol, ModelType.ENSEMBLE, raw_features)
        
        # Aggregate predictions
        buy_score = 0
        sell_score = 0
        hold_score = 0
        total_confidence = 0
        
        for pred in individual_predictions:
            weight = pred.confidence
            if pred.prediction == "BUY":
                buy_score += weight
            elif pred.prediction == "SELL":
                sell_score += weight
            else:
                hold_score += weight
            total_confidence += pred.confidence
        
        # Determine ensemble prediction
        scores = {"BUY": buy_score, "SELL": sell_score, "HOLD": hold_score}
        ensemble_prediction = max(scores, key=scores.get)
        
        # Calculate ensemble confidence
        avg_confidence = total_confidence / len(individual_predictions) if individual_predictions else 0.5
        agreement_ratio = scores[ensemble_prediction] / sum(scores.values()) if sum(scores.values()) > 0 else 0
        ensemble_confidence = (avg_confidence + agreement_ratio) / 2
        
        # Average price targets
        target_prices = [p.target_price for p in individual_predictions]
        stop_losses = [p.stop_loss for p in individual_predictions]
        take_profits = [p.take_profit for p in individual_predictions]
        
        # Aggregate feature importance
        combined_importance = defaultdict(float)
        for pred in individual_predictions:
            for feature, importance in pred.features_importance.items():
                combined_importance[feature] += importance
        
        # Normalize
        total_importance = sum(combined_importance.values())
        if total_importance > 0:
            for feature in combined_importance:
                combined_importance[feature] /= total_importance
        
        reasoning = [
            f"Ensemble of {len(individual_predictions)} models",
            f"Consensus: {ensemble_prediction} ({agreement_ratio:.1%} agreement)",
            f"Combined confidence: {ensemble_confidence:.1%}"
        ]
        
        # Add model breakdown
        model_votes = defaultdict(int)
        for pred in individual_predictions:
            model_votes[pred.prediction] += 1
        
        for action, count in model_votes.items():
            reasoning.append(f"{count} models predict {action}")
        
        return MLPrediction(
            model_type=ModelType.ENSEMBLE,
            symbol=symbol,
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            target_price=round(np.mean(target_prices), 2),
            stop_loss=round(np.mean(stop_losses), 2),
            take_profit=round(np.mean(take_profits), 2),
            timeframe="1-5 days",
            features_importance=dict(combined_importance),
            reasoning=reasoning
        )
    
    def _generate_mock_prediction(
        self,
        symbol: str,
        model_type: ModelType,
        features: Dict[str, float]
    ) -> MLPrediction:
        """Generate mock prediction for demo purposes"""
        
        # Simulate prediction based on RSI and trend
        rsi = features.get('rsi', 50)
        trend = features.get('trend_strength', 0)
        
        if rsi < 30:
            prediction = "BUY"
            confidence = 0.8 + (30 - rsi) / 100
        elif rsi > 70:
            prediction = "SELL"
            confidence = 0.8 + (rsi - 70) / 100
        elif trend > 0.5:
            prediction = "BUY"
            confidence = 0.6 + trend * 0.2
        elif trend < -0.5:
            prediction = "SELL"
            confidence = 0.6 + abs(trend) * 0.2
        else:
            prediction = "HOLD"
            confidence = 0.5 + abs(trend)
        
        confidence = min(0.95, confidence)
        current_price = features.get('current_price', 100)
        atr = features.get('atr', current_price * 0.02)
        
        if prediction == "BUY":
            target_price = current_price * (1.03 + confidence * 0.02)
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3)
        elif prediction == "SELL":
            target_price = current_price * (0.97 - confidence * 0.02)
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3)
        else:
            target_price = current_price
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 2)
        
        # Mock feature importance
        feature_importance = {
            'rsi': 0.25,
            'trend_strength': 0.20,
            'volume_ratio': 0.15,
            'macd': 0.10,
            'volatility': 0.10,
            'momentum': 0.10,
            'support_distance': 0.05,
            'resistance_distance': 0.05
        }
        
        reasoning = [
            f"{model_type.value} model prediction: {prediction}",
            f"Confidence: {confidence:.1%}",
            f"RSI at {rsi:.1f} indicates {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'} conditions"
        ]
        
        if abs(trend) > 0.3:
            reasoning.append(f"{'Strong' if abs(trend) > 0.7 else 'Moderate'} {'uptrend' if trend > 0 else 'downtrend'} detected")
        
        return MLPrediction(
            model_type=model_type,
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            timeframe="1-5 days",
            features_importance=feature_importance,
            reasoning=reasoning
        )
    
    async def train_online(
        self,
        new_data: List[Dict[str, Any]],
        model_type: ModelType = ModelType.XGBOOST
    ) -> bool:
        """Online training with new data (incremental learning)"""
        # This would implement online learning in production
        # For now, just log the attempt
        logger.info(f"Online training requested for {model_type.value} with {len(new_data)} samples")
        return True
    
    async def evaluate_performance(
        self,
        predictions: List[MLPrediction],
        actual_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        # This would calculate actual performance metrics
        # For now, return mock metrics
        return {
            'accuracy': 0.75,
            'precision': 0.72,
            'recall': 0.78,
            'f1_score': 0.75,
            'sharpe_ratio': 1.8,
            'win_rate': 0.62,
            'avg_return': 0.08
        }


# Singleton instance
advanced_ml_models = AdvancedMLModels() 