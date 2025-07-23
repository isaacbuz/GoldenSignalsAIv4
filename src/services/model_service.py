import numpy as np
import tensorflow as tf

# Import the agentic foundation model abstraction layer
from integration.external_model_service import ExternalModelService

from src.ml.models.factory import ModelFactory


class Validator:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models:
            prediction = model.predict(X)
            predictions.append(prediction)
        return np.array(predictions)

class ModelService:
    """
    ModelService orchestrates all model-related operations in GoldenSignalsAI.
    It manages both custom (local) models and external foundation models via agentic workflows.
    
    - Custom models: LSTM, XGBoost, LightGBM, RL, etc.
    - Foundation models: Claude, Llama, Titan, Cohere, Grok, etc., via ExternalModelService.
    
    This design allows you to combine, benchmark, or ensemble custom and external models for maximum robustness and flexibility.
    """
    def __init__(self):
        # Custom model infrastructure
        self.model_factory = ModelFactory()
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            tf.config.set_visible_devices(self.gpus[0], 'GPU')
        self.validator = Validator([
            self.model_factory.create_model("lstm"),
            self.model_factory.create_model("xgboost"),
            self.model_factory.create_model("lightgbm"),
            self.model_factory.create_model("sentiment")
        ])
        # Agentic foundation model abstraction layer
        # Configure providers as needed, or use default selection logic
        self.external_models = ExternalModelService(config={
            "sentiment_provider": "grok",  # Example: use Grok for sentiment
            "explanation_provider": "claude",
            "embedding_provider": "titan",
            "vision_provider": "llama"
        })

    async def train_lstm(self, X, y, symbol):
        lstm_model = self.model_factory.create_model("lstm")
        with tf.device('/GPU:0'):
            lstm_model.fit(X, y)
        return True

    async def predict_lstm(self, symbol, X, scaler):
        lstm_model = self.model_factory.create_model("lstm")
        with tf.device('/GPU:0'):
            prediction = lstm_model.predict(X)
        return scaler.inverse_transform(prediction)[0]

    async def train_xgboost(self, df, symbol):
        xgboost_model = self.model_factory.create_model("xgboost")
        X, y = self._prepare_data(df)
        xgboost_model.fit(X, y)
        return xgboost_model.predict(X[-1:])[0]

    async def train_lightgbm(self, df, symbol):
        lightgbm_model = self.model_factory.create_model("lightgbm")
        X, y = self._prepare_data(df)
        lightgbm_model.fit(X, y)
        return lightgbm_model.predict(X[-1:])[0]

    async def analyze_sentiment(self, news_articles, use_external: bool = False, provider: str = None):
        """
        Analyze sentiment of news articles using either the custom model or an external foundation model (async).
        If use_external is True, route the request to the agentic foundation model layer (with caching, fallback, and async).
        """
        if use_external:
            text = news_articles if isinstance(news_articles, str) else ", ".join(news_articles)
            return await self.external_models.analyze_sentiment(text, provider=provider)
        else:
            sentiment_model = self.model_factory.create_model("sentiment")
            return sentiment_model.analyze(news_articles)

    async def ensemble_sentiment(self, news_articles) -> dict:
        """
        Agentic ensemble: aggregate sentiment from all available external foundation models (async, parallel, weighted).
        Returns a consensus sentiment and confidence.
        """
        text = news_articles if isinstance(news_articles, str) else ", ".join(news_articles)
        return await self.external_models.ensemble_sentiment(text)

    async def generate_explanation(self, input_text: str, provider: str = None) -> str:
        """
        Generate a natural language explanation for a trade or strategy using an external foundation model (async, cached).
        Provider can be specified or chosen agentically.
        """
        return await self.external_models.generate_explanation(input_text, provider=provider)

    async def get_embeddings(self, text: str, provider: str = None) -> list:
        """
        Get semantic embeddings for a given text using an external foundation model (async, cached).
        Useful for semantic search, clustering, or similarity tasks.
        """
        return await self.external_models.get_embeddings(text, provider=provider)

    async def vision_analysis(self, image, provider: str = None) -> dict:
        """
        Analyze an image (e.g., chart, screenshot, document) using a vision-capable foundation model (async, cached).
        Returns extracted insights or detected patterns.
        """
        return await self.external_models.vision_analysis(image, provider=provider)

    async def train_rl(self, df, symbol):
        rl_model = self.model_factory.create_model("rl")
        return rl_model.train(df)

    def _prepare_data(self, df):
        X = df[['open', 'high', 'low', 'volume']].values
        y = df['close'].shift(-1).values[:-1]
        X = X[:-1]
        return X, y
