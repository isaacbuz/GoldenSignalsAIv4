"""
orchestrator.py
Purpose: Implements the Orchestrator for GoldenSignalsAI, coordinating data fetching, model training, strategy execution, alerting, and event publishing for autonomous trading workflows.
"""

import os
import pandas as pd
import logging
import asyncio
import yaml

# Load per-symbol thresholds from config
THRESHOLD_CONFIG = {}
def get_threshold(symbol, default=100.0):
    global THRESHOLD_CONFIG
    if not THRESHOLD_CONFIG:
        try:
            with open('config/thresholds.yaml', 'r') as f:
                THRESHOLD_CONFIG = yaml.safe_load(f) or {}
        except Exception:
            THRESHOLD_CONFIG = {}
    return THRESHOLD_CONFIG.get(symbol, default)

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
from src.services.meta_signal_agent import MetaSignalAgent
from src.services.gpt_model_copilot import GPTModelCopilot
from src.services.forecasting_agent import ForecastingAgent
from src.services.strategy_selector import StrategySelector
# === Advanced Model Agents ===
from agents.finbert_sentiment_agent import FinBERTSentimentAgent
from agents.lstm_forecast_agent import LSTMForecastAgent
from agents.ml_classifier_agent import MLClassifierAgent
from agents.rsi_macd_agent import RSIMACDAgent
from strategies.advanced_strategies import AdvancedStrategies
from src.domain.trading.strategies.backtest_strategy import BacktestStrategy
from src.ml.models.factory import ModelFactory
from src.infrastructure.error_handler import ErrorHandler, ModelInferenceError, DataFetchError

logger = logging.getLogger(__name__)
DEVICE, USE_NUMBA = configure_hardware()

class Orchestrator:
    """
    Orchestrator for GoldenSignalsAI: config-driven, registry-based, robust error handling.
    """
    def __init__(self,
                 symbols: List[str],
                 model_names: List[str],
                 strategy_name: str,
                 config_path: str = 'config/parameters.yaml',
                 thresholds_path: str = 'config/thresholds.yaml'):
        self.symbols = symbols
        self.model_names = model_names
        self.strategy_name = strategy_name
        self.data_service = DataService()
        self.model_service = ModelService()
        self.producer = EventPublisher()
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(thresholds_path, 'r') as f:
            self.thresholds = yaml.safe_load(f)

    def get_model_params(self, name):
        return self.config.get('models', {}).get(name, {})

    def get_strategy_params(self, name):
        return self.config.get('strategies', {}).get(name, {})

    def get_threshold(self, symbol):
        t = self.thresholds.get(symbol, self.thresholds.get('default', {}))
        return t.get('base', 1.0), t.get('atr_mult', 2.0)

    async def run(self):
        logger.info(f"Starting training and prediction for {self.symbols}")
        if not isinstance(self.symbols, (list, tuple)) or not all(isinstance(s, str) and s.strip() for s in self.symbols):
            logger.error("Invalid symbols: Must be a list of non-empty strings")
            return False

        async def process_symbol(symbol: str):
            try:
                df = await self.data_service.fetch_ohlcv(symbol)
                if df is None or df.empty:
                    raise DataFetchError(f"No data for symbol {symbol}")
                # Compute ATR-based and config threshold
                base, atr_mult = self.get_threshold(symbol)
                atr = AdvancedStrategies.run_strategy('atr', df, period=self.config.get('strategies', {}).get('atr_breakout', {}).get('period', 14))
                dynamic_threshold = base + (atr.iloc[-1] * atr_mult)

                # Preprocess data
                X_train, y_train, X_pred = self.data_service.preprocess(df)

                # Ensemble predictions using model registry and config
                preds = []
                for name in self.model_names:
                    params = self.get_model_params(name)
                    model = ModelFactory.get_model(name, config=params)
                    model_path = f'models/{symbol}/{name}.model'
                    if os.path.exists(model_path):
                        model.load(model_path)
                    else:
                        model.fit(X_train, y_train)
                        model.save(model_path)
                    p = model.predict(X_pred)
                    preds.append(p)
                ensemble_signal = np.mean(preds, axis=0)

                # Strategy selection and backtest
                strat_params = self.get_strategy_params(self.strategy_name)
                df['signal'] = AdvancedStrategies.run_strategy(self.strategy_name, df, **strat_params)
                bt = BacktestStrategy(price_df=df, signal_series=df['signal'])
                result = bt.run()
                logger.info(f"{symbol} backtest metrics: {result.metrics}")

                # Alert if signal magnitude exceeds threshold
                latest_signal = ensemble_signal[-1] if hasattr(ensemble_signal, '__getitem__') else ensemble_signal
                if abs(latest_signal) > dynamic_threshold:
                    message = {
                        'symbol': symbol,
                        'signal': float(latest_signal),
                        'threshold': float(dynamic_threshold)
                    }
                    await self.producer.send(message)
                    logger.info(f"Alert sent for {symbol}: {message}")

            except Exception as e:
                ErrorHandler.handle_error(e, context={'symbol': symbol})

        tasks = [process_symbol(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks)
        return all(results)

if __name__ == '__main__':
    import os
    syms = os.getenv('SYMBOLS', 'AAPL,GOOG,MSFT').split(',')
    models = os.getenv('MODELS', 'xgboost,lightgbm').split(',')
    strategy = os.getenv('STRATEGY', 'moving_average_crossover')
    orch = Orchestrator(symbols=syms, model_names=models, strategy_name=strategy)
    asyncio.run(orch.run())
