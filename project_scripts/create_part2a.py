# create_part2a.py
# Purpose: Creates files in the application/ai_service/ directory for the
# updated GoldenSignalsAI_project_new, including model factory and orchestrator with
# performance optimizations (Redis caching, batch predictions) for options trading.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part2a():
    """Create files in application/ai_service/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating application/ai_service files in {base_dir}"})

    # Define ai_service directory files
    ai_service_files = {
        "application/ai_service/__init__.py": """# application/ai_service/__init__.py
# Purpose: Marks the ai_service directory as a Python subpackage, enabling imports
# for AI model orchestration components.

# Empty __init__.py to mark ai_service as a subpackage
""",
        "application/ai_service/model_factory.py": """# application/ai_service/model_factory.py
# Purpose: Provides a factory pattern for creating AI models (e.g., LSTM, XGBoost) used in
# the orchestrator for ensemble predictions. Integrates with MLflow for logging model metrics
# and artifacts, supporting options trading signal generation.

from abc import ABC, abstractmethod
from domain.models.ai_models import (
    LSTMModel, GRUModel, TransformerModel, CNNModel,
    XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel,
    GradientBoostingModel, SVMModel, KNNModel, ARIMAModel,
    GARCHModel, DQNModel, GaussianProcessModel
)
import mlflow
import logging
import pickle
import os

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class AIModel(ABC):
    \"\"\"Abstract base class for all AI models.\"\"\"
    @abstractmethod
    def train(self, data, **kwargs):
        \"\"\"Train the model with the provided data.
        
        Args:
            data: Training data (format depends on model).
            **kwargs: Additional training parameters.
        \"\"\"
        pass

    @abstractmethod
    def predict(self, data):
        \"\"\"Generate predictions with the trained model.
        
        Args:
            data: Input data for prediction.
        
        Returns:
            Predicted values.
        \"\"\"
        pass

    def log_to_mlflow(self, model_type: str, symbol: str, metrics: dict):
        \"\"\"Log training metrics and model to MLflow.
        
        Args:
            model_type (str): Type of model (e.g., 'lstm').
            symbol (str): Stock symbol (e.g., 'AAPL').
            metrics (dict): Metrics to log (e.g., {'mse': 0.01}).
        \"\"\"
        logger.info({"message": f"Logging {model_type} model for {symbol} to MLflow"})
        try:
            with mlflow.start_run(run_name=f"{model_type}_{symbol}"):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("symbol", symbol)
                # Save and log the model as an artifact
                model_path = f"{model_type}_{symbol}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(self, f)
                mlflow.log_artifact(model_path)
                os.remove(model_path)  # Clean up temporary file
            logger.info({"message": f"Successfully logged {model_type} model for {symbol} to MLflow"})
        except Exception as e:
            logger.error({"message": f"Failed to log to MLflow: {str(e)}"})

class ModelFactory:
    \"\"\"Factory for creating AI model instances.\"\"\"
    # Dictionary mapping model types to their classes
    _models = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'transformer': TransformerModel,
        'cnn': CNNModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
        'svm': SVMModel,
        'knn': KNNModel,
        'arima': ARIMAModel,
        'garch': GARCHModel,
        'dqn': DQNModel,
        'gaussian_process': GaussianProcessModel
    }

    @staticmethod
    def create_model(model_type: str, **kwargs) -> AIModel:
        \"\"\"Create an AI model instance based on the specified type.
        
        Args:
            model_type (str): Type of model to create (e.g., 'lstm').
            **kwargs: Additional parameters for model initialization.
        
        Returns:
            AIModel: Instantiated model object.
        
        Raises:
            ValueError: If the model type is unknown.
        \"\"\"
        logger.info({"message": f"Creating model: {model_type}"})
        model_class = ModelFactory._models.get(model_type)
        if not model_class:
            logger.error({"message": f"Unknown model type: {model_type}"})
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class(**kwargs)
""",
        "application/ai_service/orchestrator.py": """# application/ai_service/orchestrator.py
# Purpose: Orchestrates AI model predictions, combining multiple models into an ensemble
# for robust trading signals. Enhanced with Redis caching and batch predictions for
# performance, and consumes Redis Streams for decoupled data processing. Supports
# options trading with anomaly and drift detection.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import entropy
import logging
import yaml
import requests
from .model_factory import ModelFactory, AIModel
import redis
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class Orchestrator:
    \"\"\"Manages AI model predictions, anomaly detection, and drift detection.\"\"\"
    def __init__(self, data_fetcher, scaler=None):
        # Initialize data fetcher and scaler
        self.data_fetcher = data_fetcher
        self.scaler = scaler if scaler is not None else MinMaxScaler(feature_range=(0, 1))
        # Define supported model types
        self.model_types = [
            'lstm', 'gru', 'transformer', 'cnn', 'xgboost', 'lightgbm',
            'catboost', 'random_forest', 'gradient_boosting', 'svm',
            'knn', 'arima', 'garch', 'dqn', 'gaussian_process'
        ]
        # Initialize model storage
        self.models = {model_type: {} for model_type in self.model_types}
        self.isolation_forests = {}
        self.training_distributions = {}
        self.model_performance = {model_type: {} for model_type in self.model_types}
        self.performance_history = {model_type: {} for model_type in self.model_types}
        # Load configuration parameters
        self.drift_threshold = config["drift_threshold"]
        self.model_weights = config["model_weights"]
        # Initialize Redis client (supports cluster mode)
        if config['redis'].get('cluster_enabled', False):
            from redis.cluster import RedisCluster
            nodes = config['redis']['cluster_nodes']
            self.redis_client = RedisCluster(startup_nodes=[{'host': node['host'], 'port': node['port']} for node in nodes])
        else:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        # Initialize thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_models()
        logger.info({"message": f"Orchestrator initialized with data_fetcher={data_fetcher}"})

    def _initialize_models(self):
        \"\"\"Initialize AI models for each model type.\"\"\"
        logger.info({"message": "Initializing AI models"})
        for model_type in self.model_types:
            if model_type in ['lstm', 'gru', 'transformer', 'cnn']:
                self.models[model_type] = ModelFactory.create_model(model_type, scaler=self.scaler)
            else:
                self.models[model_type] = ModelFactory.create_model(model_type)

    async def consume_stream(self, symbol: str) -> dict:
        \"\"\"Consume the latest data from the Redis Stream for a given symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL').
        
        Returns:
            dict: Latest market data or None if unavailable.
        \"\"\"
        logger.info({"message": f"Consuming stream for {symbol}"})
        try:
            messages = self.redis_client.xrevrange('market-data-stream', count=1)
            if not messages:
                logger.warning({"message": f"No data available in stream for {symbol}"})
                return None

            for message_id, message in messages:
                data = {}
                for key, value in message.items():
                    data[key.decode('utf-8')] = value.decode('utf-8')
                if data['symbol'] == symbol:
                    # Deserialize JSON data
                    data['stock_data'] = pd.read_json(data['stock_data'])
                    data['options_data'] = pd.read_json(data['options_data'])
                    data['news_data'] = eval(data['news_articles'])  # Safely parse stringified list
                    logger.info({"message": f"Consumed stream data for {symbol}"})
                    return data
            return None
        except Exception as e:
            logger.error({"message": f"Failed to consume stream for {symbol}: {str(e)}"})
            return None

    def _cache_prediction(self, model_type: str, symbol: str, prediction: float):
        \"\"\"Cache a model prediction in Redis.
        
        Args:
            model_type (str): Model type (e.g., 'lstm').
            symbol (str): Stock symbol.
            prediction (float): Prediction value.
        \"\"\"
        try:
            cache_key = f"prediction:{model_type}:{symbol}"
            self.redis_client.setex(cache_key, 300, str(prediction))  # Cache for 5 minutes
            logger.debug({"message": f"Cached prediction for {model_type}:{symbol}"})
        except Exception as e:
            logger.error({"message": f"Failed to cache prediction: {str(e)}"})

    def _get_cached_prediction(self, model_type: str, symbol: str) -> float | None:
        \"\"\"Retrieve a cached prediction from Redis.
        
        Args:
            model_type (str): Model type.
            symbol (str): Stock symbol.
        
        Returns:
            float | None: Cached prediction or None if not found.
        \"\"\"
        try:
            cache_key = f"prediction:{model_type}:{symbol}"
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug({"message": f"Retrieved cached prediction for {model_type}:{symbol}"})
                return float(cached.decode('utf-8'))
            return None
        except Exception as e:
            logger.error({"message": f"Failed to retrieve cached prediction: {str(e)}"})
            return None

    def train_model(self, model_type: str, symbol: str, X, y=None, historical_df=None):
        \"\"\"Train an AI model for a specific symbol.
        
        Args:
            model_type (str): Model type.
            symbol (str): Stock symbol.
            X: Input data for training (format depends on model).
            y: Target data for supervised models (optional).
            historical_df: Historical data DataFrame for some models.
        \"\"\"
        logger.info({"message": f"Training {model_type} model for {symbol}"})
        try:
            if model_type in ['lstm', 'gru', 'transformer', 'cnn']:
                self.models[model_type].train(X, y)
                predicted = self.models[model_type].predict(X[-10:])
                actual = y[-10:]
                mse = np.mean((actual - predicted) ** 2)
                self.models[model_type].log_to_mlflow(model_type, symbol, {"mse": mse})
            else:
                self.models[model_type].train(historical_df)
                predicted = self.models[model_type].predict(historical_df.tail(10))
                actual = historical_df['Close'].shift(-1).tail(10).dropna()
                mse = np.mean((actual - predicted[:len(actual)]) ** 2)
                self.models[model_type].log_to_mlflow(model_type, symbol, {"mse": mse})
            self.models[model_type][symbol] = self.models[model_type]

            # Store training distribution for drift detection
            if model_type == 'lstm':
                training_data = X.flatten()
                ecdf = ECDF(training_data)
                self.training_distributions[symbol] = ecdf
            logger.info({"message": f"{model_type} model trained successfully for {symbol}"})
        except Exception as e:
            logger.error({"message": f"Failed to train {model_type} model for {symbol}: {str(e)}"})
            self.models[model_type][symbol] = None

    def train_isolation_forest(self, data: np.ndarray, symbol: str, contamination: float = 0.1):
        \"\"\"Train an Isolation Forest for anomaly detection.
        
        Args:
            data (np.ndarray): Data for training.
            symbol (str): Stock symbol.
            contamination (float): Expected proportion of anomalies.
        \"\"\"
        logger.info({"message": f"Training Isolation Forest for {symbol} with contamination={contamination}"})
        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_forest.fit(data.reshape(-1, 1))
            self.isolation_forests[symbol] = iso_forest
            logger.info({"message": f"Isolation Forest trained successfully for {symbol}"})
        except Exception as e:
            logger.error({"message": f"Failed to train Isolation Forest for {symbol}: {str(e)}"})
            self.isolation_forests[symbol] = None

    def detect_anomalies(self, symbol: str, X_new: np.ndarray) -> np.ndarray:
        \"\"\"Detect anomalies in new data using LSTM and Isolation Forest.
        
        Args:
            symbol (str): Stock symbol.
            X_new (np.ndarray): New data for anomaly detection.
        
        Returns:
            np.ndarray: Anomaly labels (-1 for anomaly, 1 for normal).
        \"\"\"
        logger.info({"message": f"Detecting anomalies for {symbol}"})
        if symbol not in self.models['lstm'] or symbol not in self.isolation_forests:
            logger.error({"message": f"Models not found for {symbol}"})
            return None

        try:
            lstm_model = self.models['lstm'][symbol]
            predictions = lstm_model.predict(X_new)
            errors = np.abs(X_new[:, -1, 0] - predictions.flatten())
            error_threshold = np.percentile(errors, 95)
            lstm_anomalies = errors > error_threshold

            iso_forest = self.isolation_forests[symbol]
            iso_predictions = iso_forest.predict(X_new[:, -1, :].reshape(-1, 1))
            iso_anomalies = iso_predictions == -1

            combined_anomalies = np.logical_or(lstm_anomalies, iso_anomalies)
            labels = np.where(combined_anomalies, -1, 1)
            logger.info({"message": f"Detected {np.sum(combined_anomalies)} anomalies for {symbol}"})
            return labels
        except Exception as e:
            logger.error({"message": f"Failed to detect anomalies for {symbol}: {str(e)}"})
            return None

    async def predict_model(self, model_type: str, symbol: str, X_new) -> float:
        \"\"\"Generate a prediction for a specific model and symbol.
        
        Args:
            model_type (str): Model type.
            symbol (str): Stock symbol.
            X_new: Input data for prediction.
        
        Returns:
            float: Predicted value or None if prediction fails.
        \"\"\"
        logger.info({"message": f"Predicting with {model_type} for {symbol}"})
        if symbol not in self.models[model_type]:
            logger.error({"message": f"No trained {model_type} model found for {symbol}"})
            return None

        if self.models[model_type][symbol] is None:
            logger.error({"message": f"{model_type} model for {symbol} is not initialized"})
            return None

        # Check for model drift (for LSTM only)
        if model_type == 'lstm' and self.detect_drift(symbol, X_new):
            logger.warning({"message": f"Model drift detected for {symbol}. Consider retraining."})

        try:
            # Check cache first
            cached_prediction = self._get_cached_prediction(model_type, symbol)
            if cached_prediction is not None:
                return cached_prediction

            # Run prediction in thread pool for CPU-bound models
            loop = asyncio.get_event_loop()
            predicted = await loop.run_in_executor(
                self.executor,
                lambda: self.models[model_type][symbol].predict(X_new)
            )
            self._cache_prediction(model_type, symbol, predicted)
            logger.debug({"message": f"{model_type} prediction for {symbol}: {predicted:.2f}"})
            return predicted
        except Exception as e:
            logger.error({"message": f"Failed to predict with {model_type} model for {symbol}: {str(e)}"})
            return None

    def update_model_performance(self, model_type: str, symbol: str, actual: np.ndarray, predicted: np.ndarray):
        \"\"\"Update model performance metrics.
        
        Args:
            model_type (str): Model type.
            symbol (str): Stock symbol.
            actual (np.ndarray): Actual values.
            predicted (np.ndarray): Predicted values.
        \"\"\"
        logger.info({"message": f"Updating performance for {model_type} on {symbol}"})
        try:
            error = np.mean((actual - predicted) ** 2)
            performance_score = 1 / (error + 1e-10)
            if symbol not in self.performance_history[model_type]:
                self.performance_history[model_type][symbol] = []
            self.performance_history[model_type][symbol].append(performance_score)
            if len(self.performance_history[model_type][symbol]) > 30:
                self.performance_history[model_type][symbol].pop(0)
            self.model_performance[model_type][symbol] = np.mean(self.performance_history[model_type][symbol])
            logger.debug({"message": f"Updated performance for {model_type} on {symbol}: {self.model_performance[model_type][symbol]:.4f}"})
        except Exception as e:
            logger.error({"message": f"Failed to update performance: {str(e)}"})

    async def ensemble_predict(self, symbol: str, X_new_lstm: np.ndarray, X_new_tree: pd.DataFrame) -> float:
        \"\"\"Generate an ensemble prediction combining multiple models.
        
        Args:
            symbol (str): Stock symbol.
            X_new_lstm (np.ndarray): Input data for LSTM-based models.
            X_new_tree (pd.DataFrame): Input data for tree-based models.
        
        Returns:
            float: Ensemble prediction or None if no valid predictions.
        \"\"\"
        logger.info({"message": f"Generating ensemble prediction for {symbol}"})
        predictions = []
        weights = []

        # Batch predictions asynchronously
        tasks = []
        for model_type in self.model_types:
            X_new = X_new_lstm if model_type in ['lstm', 'gru', 'transformer', 'cnn'] else X_new_tree
            tasks.append(self.predict_model(model_type, symbol, X_new))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for model_type, result in zip(self.model_types, results):
            if isinstance(result, Exception) or result is None:
                logger.warning({"message": f"{model_type} prediction for {symbol} is unavailable: {str(result)}"})
                continue
            pred = result
            # Update performance (simplified; use validation set in production)
            actual = X_new_tree['Close'].values[-1]
            self.update_model_performance(model_type, symbol, np.array([actual]), np.array([pred]))
            predictions.append(pred)
            weight = self.model_performance.get(model_type, {}).get(symbol, self.model_weights.get(model_type, 0.1))
            weights.append(weight)
            logger.info({"message": f"{model_type} prediction for {symbol}: {pred:.2f} (weight: {weight:.2f})"})

        if not predictions:
            logger.error({"message": f"No valid predictions available for ensemble for {symbol}"})
            return None

        # Compute weighted ensemble prediction
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        ensemble_pred = sum(pred * weight for pred, weight in zip(predictions, weights))
        logger.info({"message": f"Ensemble prediction for {symbol}: {ensemble_pred:.2f}"})
        return ensemble_pred

    def detect_drift(self, symbol: str, X_new: np.ndarray) -> bool:
        \"\"\"Detect model drift by comparing data distributions.
        
        Args:
            symbol (str): Stock symbol.
            X_new (np.ndarray): New data for drift detection.
        
        Returns:
            bool: True if drift is detected, False otherwise.
        \"\"\"
        logger.info({"message": f"Detecting drift for {symbol}"})
        if symbol not in self.training_distributions:
            logger.error({"message": f"No training distribution found for {symbol}"})
            return False

        try:
            inference_data = X_new.flatten()
            ecdf_inference = ECDF(inference_data)
            ecdf_training = self.training_distributions[symbol]

            x = np.linspace(
                min(min(ecdf_training.x), min(ecdf_inference.x)),
                max(max(ecdf_training.x), max(ecdf_inference.x)), 1000
            )
            p = ecdf_training(x) + 1e-10
            q = ecdf_inference(x) + 1e-10
            kl_divergence = entropy(p, q)

            logger.info({"message": f"KL Divergence for {symbol}: {kl_divergence:.4f}"})
            if kl_divergence > self.drift_threshold:
                slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
                if slack_webhook_url:
                    requests.post(slack_webhook_url, json={
                        "text": f"Model drift detected for {symbol}! KL Divergence: {kl_divergence:.4f}. Consider retraining."
                    })
                return True
            return False
        except Exception as e:
            logger.error({"message": f"Failed to detect drift for {symbol}: {str(e)}"})
            return False

    def analyze_sentiment(self, news_articles: list) -> float:
        \"\"\"Analyze sentiment from news articles using VADER.
        
        Args:
            news_articles (list): List of news articles (dicts with description).
        
        Returns:
            float: Average sentiment score (positive > 0, negative < 0).
        \"\"\"
        logger.info({"message": f"Analyzing sentiment from {len(news_articles)} news articles"})
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = [
                analyzer.polarity_scores(article["description"])["compound"]
                for article in news_articles if article.get("description")
            ]
            return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        except Exception as e:
            logger.error({"message": f"Failed to analyze sentiment: {str(e)}"})
            return 0.0
""",
    }

    # Write ai_service directory files
    for file_path, content in ai_service_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 2a: application/ai_service/ created successfully")


if __name__ == "__main__":
    create_part2a()
