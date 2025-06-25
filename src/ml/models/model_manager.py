from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from agents.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import os
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def handle_errors(func):
    """Decorator for error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class ModelManager:
    """Model management functionality integrated from AlphaPy"""
    
    VALID_MODEL_TYPES = {"classifier", "regressor"}
    
    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = model_dir
        self.models: Dict[str, BaseEstimator] = {}
        os.makedirs(model_dir, exist_ok=True)
        logger.info("Initialized ModelManager with directory: %s", model_dir)
        
    @handle_errors
    def train_model(
        self,
        model_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "classifier",
        params: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """Train a new model with given parameters"""
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type. Must be one of {self.VALID_MODEL_TYPES}")
            
        if params is None:
            params = {}
            
        logger.info("Training %s model with ID: %s", model_type, model_id)
        logger.debug("Model parameters: %s", params)
            
        if model_type == "classifier":
            model = RandomForestClassifier(**params)
        else:
            model = GradientBoostingRegressor(**params)
            
        model.fit(X_train, y_train)
        self.models[model_id] = model
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
        joblib.dump(model, model_path)
        logger.info("Model saved to: %s", model_path)
        
        return model
    
    @handle_errors
    def predict(
        self,
        model_id: str,
        X: pd.DataFrame,
        probability: bool = False
    ) -> np.ndarray:
        """Make predictions using a trained model"""
        model = self.get_model(model_id)
        
        if probability and not hasattr(model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
            
        logger.info("Making predictions with model: %s", model_id)
        if probability and hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
        else:
            predictions = model.predict(X)
            
        logger.debug("Generated %d predictions", len(predictions))
        return predictions
    
    @handle_errors
    def evaluate_model(
        self,
        model_id: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        model = self.get_model(model_id)
        predictions = model.predict(X_test)
        
        metrics = {}
        if hasattr(model, 'predict_proba'):
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            logger.info("Classification metrics for model %s: accuracy=%.3f",
                       model_id, metrics['accuracy'])
        else:
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, predictions)
            logger.info("Regression metrics for model %s: rmse=%.3f, r2=%.3f",
                       model_id, metrics['rmse'], metrics['r2'])
            
        return metrics
    
    @handle_errors
    def get_model(self, model_id: str) -> BaseEstimator:
        """Retrieve a model by ID"""
        if model_id not in self.models:
            model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
            if os.path.exists(model_path):
                logger.info("Loading model from: %s", model_path)
                self.models[model_id] = joblib.load(model_path)
            else:
                logger.error("Model not found: %s", model_id)
                raise ValueError(f"Model {model_id} not found")
        return self.models[model_id]
        
    def list_models(self) -> List[str]:
        """List all available models"""
        model_files = [f for f in os.listdir(self.model_dir) 
                      if f.endswith('.joblib')]
        return [os.path.splitext(f)[0] for f in model_files] 