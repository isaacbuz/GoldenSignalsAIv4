"""
orchestrator.py
Purpose: Implements the Orchestrator for GoldenSignalsAI, coordinating data fetching, model training, strategy execution, alerting, and event publishing for autonomous trading workflows.
"""

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
# === Grok AI Agents ===
from agents.grok.grok_sentiment import GrokSentimentAgent
from agents.grok.grok_strategy import GrokStrategyAgent
from agents.grok.grok_backtest import GrokBacktestCritic
# === Meta/ML Agents ===
from application.services.meta_signal_agent import MetaSignalAgent
from application.services.gpt_model_copilot import GPTModelCopilot
from application.services.forecasting_agent import ForecastingAgent
from application.services.strategy_selector import StrategySelector
# === Advanced Model Agents ===
from agents.finbert_sentiment_agent import FinBERTSentimentAgent
from agents.lstm_forecast_agent import LSTMForecastAgent
from agents.ml_classifier_agent import MLClassifierAgent
from agents.rsi_macd_agent import RSIMACDAgent

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
        # === Grok AI Agents ===
        grok_api_key = os.getenv("GROK_API_KEY")
        self.grok_sentiment = GrokSentimentAgent(grok_api_key)
        self.grok_strategy = GrokStrategyAgent(grok_api_key)
        self.grok_critic = GrokBacktestCritic(grok_api_key)
        # === Advanced Model Agents ===
        self.finbert_agent = FinBERTSentimentAgent()
        self.lstm_agent = LSTMForecastAgent()
        self.ml_agent = MLClassifierAgent()
        self.rsi_macd_agent = RSIMACDAgent()

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
                        # === Use GrokSentimentAgent for enhanced sentiment ===
            sentiment_score = self.grok_sentiment.get_sentiment_score(symbol)
            # Optionally combine with legacy sentiment:
            # legacy_sentiment = await self.model_service.analyze_sentiment(news_articles)
            # sentiment_score = 0.7 * sentiment_score + 0.3 * legacy_sentiment if legacy_sentiment else sentiment_score
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

            # === Advanced Model Agent Usage ===
            # FinBERT Sentiment
            finbert_sentiment = self.finbert_agent.analyze_texts(["Sample news headline for " + symbol])
            # LSTM Forecast
            lstm_forecast = self.lstm_agent.predict(historical_df['close']) if not historical_df.empty else None
            # ML Classifier (RandomForest/XGBoost)
            ml_signal = self.ml_agent.predict_signal(historical_df[['close']].tail(5)) if len(historical_df) >= 5 else None
            # RSI + MACD Agent
            rsi_macd_signal = self.rsi_macd_agent.compute_signal(historical_df) if not historical_df.empty else None

            logger.info(f"FinBERT sentiment: {finbert_sentiment}")
            logger.info(f"LSTM forecast: {lstm_forecast}")
            logger.info(f"ML classifier signal: {ml_signal}")
            logger.info(f"RSI+MACD signal: {rsi_macd_signal}")

            backtest_result = await self.strategy_service.backtest(symbol, historical_df, [avg_pred_change] * len(historical_df))
            logger.info(f"Backtest result for {symbol}: {backtest_result}")

            # === Use GrokBacktestCritic for feedback ===
            grok_feedback = self.grok_critic.critique(
                logic="...",  # Supply actual strategy logic here
                win_rate=backtest_result.get('win_rate', 0),
                avg_return=backtest_result.get('avg_return', 0)
            )
            logger.info(f"Grok Backtest Critic feedback for {symbol}: {grok_feedback}")

            # === Use GrokStrategyAgent to generate strategies (on demand or as fallback) ===
            # new_logic = self.grok_strategy.generate_logic(symbol, timeframe="1h")
            # logger.info(f"Grok-generated strategy for {symbol}: {new_logic}")

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
