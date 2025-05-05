import tensorflow as tf
import numpy as np
from GoldenSignalsAI.domain.models.factory import ModelFactory

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
    def __init__(self):
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

    async def analyze_sentiment(self, news_articles):
        sentiment_model = self.model_factory.create_model("sentiment")
        return sentiment_model.analyze(news_articles)

    async def train_rl(self, df, symbol):
        rl_model = self.model_factory.create_model("rl")
        return rl_model.train(df)

    def _prepare_data(self, df):
        X = df[['open', 'high', 'low', 'volume']].values
        y = df['close'].shift(-1).values[:-1]
        X = X[:-1]
        return X, y
