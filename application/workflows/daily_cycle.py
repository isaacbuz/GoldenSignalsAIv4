from prefect import flow, task
from GoldenSignalsAI.application.ai_service.orchestrator import Orchestrator
from GoldenSignalsAI.application.ai_service.autonomous_engine import AutonomousEngine, Action
from GoldenSignalsAI.application.services.auto_executor import AutoExecutor
from GoldenSignalsAI.domain.trading.strategies.signal_engine import SignalEngine
from GoldenSignalsAI.application.strategies.strategy_tuner import StrategyTuner
from datetime import datetime
import pandas as pd
import numpy as np

@task
async def fetch_data(orchestrator, symbol):
    df, news_articles, realtime_df = await orchestrator.data_service.fetch_all_data(symbol)
    return df, news_articles, realtime_df

@task
async def fetch_multi_timeframe_data(orchestrator, symbol):
    data = await orchestrator.data_service.fetch_multi_timeframe_data(symbol)
    return data

@task
async def check_model_drift(orchestrator, df, symbol):
    X, y, scaler = await orchestrator.data_service.preprocess_data(df)
    lstm_pred = await orchestrator.model_service.predict_lstm(symbol, X[-1], scaler)
    actual = df['close'].iloc[-1]
    error = abs(lstm_pred - actual) / actual
    if error > 0.1:
        await orchestrator.model_service.train_lstm(X, y, symbol)
    return error

@task
async def tune_strategy(data, symbol, historical_returns):
    tuner = StrategyTuner(data["1h"], symbol, historical_returns)
    best_weights = tuner.optimize(n_trials=50)
    return best_weights

@task
async def generate_signal(data, symbol, risk_profile, weights, engine):
    decision = await engine.analyze_and_decide(symbol, data, risk_profile)
    return decision

@task
async def execute_trade(decision, orchestrator, executor):
    if decision.action != Action.HOLD:
        await executor._execute_trade(decision, orchestrator)
        await orchestrator.logger.log_decision_process(decision.symbol, decision)

@flow(name="daily-trading-cycle")
async def daily_trading_cycle(symbols: list = ["TSLA"], risk_profile: str = "balanced"):
    orchestrator = Orchestrator()
    executor = AutoExecutor()
    engine = AutonomousEngine(orchestrator.model_service.validator)
    historical_returns = []

    for symbol in symbols:
        data = await fetch_multi_timeframe_data(orchestrator, symbol)
        if not data:
            logger.error(f"No data fetched for {symbol}, skipping...")
            continue
        historical_df = await fetch_data(orchestrator, symbol)
        if historical_df[0] is not None:
            drift_error = await check_model_drift(orchestrator, historical_df[0], symbol)
            logger.info(f"Model drift error for {symbol}: {drift_error}")
        weights = await tune_strategy(data, symbol, historical_returns)
        decision = await generate_signal(data, symbol, risk_profile, weights, engine)
        await execute_trade(decision, orchestrator, executor)

if __name__ == "__main__":
    asyncio.run(daily_trading_cycle())
