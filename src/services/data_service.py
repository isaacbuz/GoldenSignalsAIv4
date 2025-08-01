import pandas as pd

from src.application.events.event_publisher import EventPublisher
from src.data.fetchers.database_fetcher import fetch_stock_data
from src.data.fetchers.news_fetcher import fetch_news_articles
from src.data.fetchers.realtime_fetcher import fetch_realtime_data
from src.data.preprocessors.stock_preprocessor import StockPreprocessor


class DataService:
    def __init__(self, use_numba=True):
        self.use_numba = use_numba
        self.preprocessor = StockPreprocessor()
        self.event_publisher = EventPublisher()

    async def fetch_all_data(self, symbol):
        historical_df = await fetch_stock_data(symbol)
        news_articles = await fetch_news_articles(symbol)
        realtime_df = await fetch_realtime_data(symbol)

        if realtime_df is not None:
            event = {
                "type": "PriceUpdateEvent",
                "symbol": symbol,
                "price": realtime_df["close"].iloc[-1],
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            await self.event_publisher.publish("price_updates", event)

        return historical_df, news_articles, realtime_df

    async def fetch_multi_timeframe_data(self, symbol):
        timeframes = ["1m", "5m", "15m", "1h"]
        data = {}
        for tf in timeframes:
            df = await fetch_stock_data(symbol, timeframe=tf)
            if df is not None:
                data[tf] = df
        return data

    async def preprocess_data(self, df):
        if df is None:
            return None, None, None
        X, y, scaler = self.preprocessor.preprocess(df, use_numba=self.use_numba)
        return X, y, scaler
