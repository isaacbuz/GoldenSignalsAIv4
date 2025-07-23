import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from src.legacy_backend_agents.ml.lstm_price_forecast_agent import LSTMPriceForecastAgent
from src.legacy_backend_agents.ml.prophet_price_forecast_agent import ProphetPriceForecastAgent
from src.legacy_backend_agents.ml.xgboost_price_forecast_agent import XGBoostPriceForecastAgent


def retrain_all():
    logging.info("[Retrain] Starting model retraining...")
    LSTMPriceForecastAgent().train()
    XGBoostPriceForecastAgent().train()
    ProphetPriceForecastAgent().train()
    logging.info("[Retrain] All models retrained successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_all, 'cron', hour=2)  # Retrain daily at 2am
    scheduler.start()
    print("Retraining scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
