import os
import pandas as pd
import logging
import asyncio
from GoldenSignalsAI.application.services.data_service import DataService
from GoldenSignalsAI.application.services.model_service import ModelService
from GoldenSignalsAI.application.services.strategy_service import StrategyService
from GoldenSignalsAI.application.services.alert_service import AlertService
from GoldenSignalsAI.application.events.event_publisher import EventPublisher
from GoldenSignalsAI.application.services.decision_logger import DecisionLogger
from GoldenSignalsAI.infrastructure.config.env_config import configure_hardware

logger = logging.getLogger(__name__)
DEVICE, USE_NUMBA = configure_hardware()

class Orchestrator:
    def __init__(self):
        self.data_service = DataService(use_numba=USE_NUMBA)
        self.model_service = ModelService()
        self.strategy_service = StrategyService()
        self.alert_service = AlertService()
        self.event_publisher = EventPublisher()
        self.logger = DecisionLogger()

    async def train_and_predict(self, symbols):
        logger.info(f"Starting training and prediction for {symbols}")
        if not isinstance(symbols, (list, tuple)) or not all(isinstance(s, str) and s.strip() for s in symbols):
            logger.error("Invalid symbols: Must be a list of non-empty strings")
            return False

        async def process_symbol(symbol):
            logger.info(f"Processing symbol {symbol}")
            historical_df, news_articles, realtime_df = await self.data_service.fetch_all_data(symbol)
            if historical_df is None:
                return False
            X, y, scaler = await self.data_service.preprocess_data(historical_df)
            if X is None or y is None:
                return False
            if not await self.model_service.train_lstm(X, y, symbol):
                return False
            xgboost_pred = await self.model_service.train_xgboost(historical_df, symbol)
            lightgbm_pred = await self.model_service.train_lightgbm(historical_df, symbol)
            sentiment_score = await self.model_service.analyze_sentiment(news_articles)
            rl_model = await self.model_service.train_rl(historical_df, symbol)
            predicted_changes = []
            lstm_pred = await self.model_service.predict_lstm(symbol, X[-1], scaler)
            if lstm_pred:
                last_close = historical_df['close'].iloc[-1]
                predicted_changes.append((lstm_pred - last_close) / last_close)
            if xgboost_pred:
                predicted_changes.append(xgboost_pred)
            if lightgbm_pred:
                predicted_changes.append(lightgbm_pred)
            if sentiment_score:
                predicted_changes.append(sentiment_score)
            avg_pred_change = sum(predicted_changes) / len(predicted_changes) if predicted_changes else 0
            backtest_result = await self.strategy_service.backtest(symbol, historical_df, [avg_pred_change] * len(historical_df))
            logger.info(f"Backtest result for {symbol}: {backtest_result}")
            if realtime_df is not None:
                latest_price = realtime_df['close'].iloc[-1]
                threshold = 100.0
                await self.alert_service.send_price_alert(symbol, latest_price, threshold)
            return True

        tasks = [process_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return all(results)

if __name__ == "__main__":
    orchestrator = Orchestrator()
    symbols = ["AAPL"]
    asyncio.run(orchestrator.train_and_predict(symbols))
