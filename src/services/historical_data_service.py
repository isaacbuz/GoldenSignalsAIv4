"""
Historical Data Service - 30 Year Data Management
Handles ingestion, storage, and retrieval of historical market data
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import requests
import yfinance as yf
from psycopg2.extras import execute_batch
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class HistoricalDataService:
    """
    Service for managing 30 years of historical market data
    Includes stock prices, sentiment, news, and economic indicators
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL", "postgresql://user:pass@localhost/goldensignals"
        )
        self.engine = create_engine(self.db_url, poolclass=QueuePool, pool_size=10, max_overflow=20)
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")

    async def initialize_database(self):
        """Create database schema for historical data"""
        schema_sql = """
        -- Enable TimescaleDB extension if available
        CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

        -- Historical prices table (partitioned by year)
        CREATE TABLE IF NOT EXISTS historical_prices (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            adjusted_close DECIMAL(10,2),
            volume BIGINT,
            dividends DECIMAL(10,4) DEFAULT 0,
            splits DECIMAL(10,4) DEFAULT 1,
            returns DECIMAL(10,6),
            log_returns DECIMAL(10,6),
            PRIMARY KEY (symbol, date)
        );

        -- Create index for fast lookups
        CREATE INDEX IF NOT EXISTS idx_prices_symbol_date
        ON historical_prices(symbol, date DESC);

        -- Technical indicators table
        CREATE TABLE IF NOT EXISTS technical_indicators (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            rsi_14 DECIMAL(5,2),
            macd DECIMAL(10,4),
            macd_signal DECIMAL(10,4),
            macd_histogram DECIMAL(10,4),
            sma_20 DECIMAL(10,2),
            sma_50 DECIMAL(10,2),
            sma_200 DECIMAL(10,2),
            ema_12 DECIMAL(10,2),
            ema_26 DECIMAL(10,2),
            bollinger_upper DECIMAL(10,2),
            bollinger_middle DECIMAL(10,2),
            bollinger_lower DECIMAL(10,2),
            atr_14 DECIMAL(10,2),
            adx_14 DECIMAL(5,2),
            volume_sma_20 BIGINT,
            obv BIGINT,
            PRIMARY KEY (symbol, date)
        );

        -- Market sentiment table
        CREATE TABLE IF NOT EXISTS market_sentiment (
            date DATE PRIMARY KEY,
            consumer_sentiment DECIMAL(5,2),
            consumer_confidence DECIMAL(5,2),
            investor_sentiment DECIMAL(5,2),
            vix DECIMAL(5,2),
            put_call_ratio DECIMAL(5,3),
            advance_decline_ratio DECIMAL(5,3),
            bull_bear_spread DECIMAL(5,2),
            fear_greed_index INTEGER,
            market_breadth DECIMAL(5,2)
        );

        -- Economic indicators table
        CREATE TABLE IF NOT EXISTS economic_indicators (
            date DATE PRIMARY KEY,
            gdp_growth DECIMAL(5,2),
            gdp_nominal DECIMAL(15,2),
            inflation_rate DECIMAL(5,2),
            core_inflation DECIMAL(5,2),
            unemployment_rate DECIMAL(5,2),
            interest_rate DECIMAL(5,2),
            fed_funds_rate DECIMAL(5,2),
            ten_year_yield DECIMAL(5,2),
            dollar_index DECIMAL(10,2),
            oil_price DECIMAL(10,2),
            gold_price DECIMAL(10,2),
            m2_money_supply DECIMAL(15,2)
        );

        -- News sentiment table
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            date TIMESTAMP NOT NULL,
            symbol VARCHAR(10),
            headline TEXT,
            summary TEXT,
            source VARCHAR(100),
            url TEXT,
            sentiment_score DECIMAL(3,2),
            relevance_score DECIMAL(3,2),
            entity_mentions TEXT,
            topics TEXT[]
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_news_date ON news_sentiment(date DESC);
        CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_sentiment(symbol);

        -- Market regimes table
        CREATE TABLE IF NOT EXISTS market_regimes (
            date DATE NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            regime VARCHAR(50),
            regime_confidence DECIMAL(3,2),
            volatility_regime VARCHAR(50),
            trend_strength DECIMAL(5,2),
            PRIMARY KEY (symbol, date)
        );

        -- Data quality tracking
        CREATE TABLE IF NOT EXISTS data_quality (
            table_name VARCHAR(50),
            symbol VARCHAR(10),
            date_range_start DATE,
            date_range_end DATE,
            record_count INTEGER,
            completeness DECIMAL(5,2),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (table_name, symbol)
        );
        """

        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()

        logger.info("Historical database schema initialized")

    async def load_stock_history(
        self, symbol: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load up to 30 years of stock history"""
        try:
            # Default to 30 years ago if no start date
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365 * 30)).strftime("%Y-%m-%d")

            # Try Yahoo Finance first
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, interval="1d", auto_adjust=False)

            if hist.empty and self.polygon_api_key:
                # Fallback to Polygon.io
                hist = await self._load_from_polygon(symbol, start_date)

            if not hist.empty:
                # Calculate additional metrics
                hist["Returns"] = hist["Close"].pct_change()
                hist["Log_Returns"] = np.log(hist["Close"] / hist["Close"].shift(1))
                hist["Volume_Ratio"] = hist["Volume"] / hist["Volume"].rolling(20).mean()

                # Store in database
                await self._store_price_data(symbol, hist)

                # Calculate and store technical indicators
                await self._calculate_and_store_indicators(symbol, hist)

                logger.info(f"Loaded {len(hist)} days of history for {symbol}")
                return hist

        except Exception as e:
            logger.error(f"Error loading history for {symbol}: {e}")

        return pd.DataFrame()

    async def _load_from_polygon(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Load data from Polygon.io API"""
        if not self.polygon_api_key:
            return pd.DataFrame()

        end_date = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"

        params = {"apiKey": self.polygon_api_key, "adjusted": "true", "sort": "asc", "limit": 50000}

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                df = pd.DataFrame(data["results"])
                df["date"] = pd.to_datetime(df["t"], unit="ms")
                df.set_index("date", inplace=True)
                df.rename(
                    columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"},
                    inplace=True,
                )
                return df

        return pd.DataFrame()

    async def load_economic_indicators(self, start_date: Optional[str] = None):
        """Load economic indicators from FRED"""
        if not self.fred_api_key:
            logger.warning("FRED API key not configured")
            return

        indicators = {
            "GDP": "GDP",
            "INFLATION": "CPIAUCSL",
            "UNEMPLOYMENT": "UNRATE",
            "FED_FUNDS": "DFF",
            "TEN_YEAR": "DGS10",
            "DOLLAR_INDEX": "DTWEXBGS",
            "M2": "M2SL",
            "VIX": "VIXCLS",
        }

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 30)).strftime("%Y-%m-%d")

        economic_data = {}

        for name, series_id in indicators.items():
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": start_date,
                "frequency": "d",  # Daily data where available
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if "observations" in data:
                    df = pd.DataFrame(data["observations"])
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df.set_index("date", inplace=True)
                    economic_data[name] = df["value"]

        # Combine all indicators
        if economic_data:
            combined_df = pd.DataFrame(economic_data)
            await self._store_economic_data(combined_df)
            logger.info(f"Loaded economic indicators from {start_date}")

    async def _calculate_and_store_indicators(self, symbol: str, price_data: pd.DataFrame):
        """Calculate technical indicators"""
        indicators = pd.DataFrame(index=price_data.index)

        # RSI
        indicators["rsi_14"] = self._calculate_rsi(price_data["Close"])

        # MACD
        exp12 = price_data["Close"].ewm(span=12).mean()
        exp26 = price_data["Close"].ewm(span=26).mean()
        indicators["macd"] = exp12 - exp26
        indicators["macd_signal"] = indicators["macd"].ewm(span=9).mean()
        indicators["macd_histogram"] = indicators["macd"] - indicators["macd_signal"]

        # Moving averages
        indicators["sma_20"] = price_data["Close"].rolling(20).mean()
        indicators["sma_50"] = price_data["Close"].rolling(50).mean()
        indicators["sma_200"] = price_data["Close"].rolling(200).mean()
        indicators["ema_12"] = price_data["Close"].ewm(span=12).mean()
        indicators["ema_26"] = price_data["Close"].ewm(span=26).mean()

        # Bollinger Bands
        sma20 = price_data["Close"].rolling(20).mean()
        std20 = price_data["Close"].rolling(20).std()
        indicators["bollinger_upper"] = sma20 + (std20 * 2)
        indicators["bollinger_middle"] = sma20
        indicators["bollinger_lower"] = sma20 - (std20 * 2)

        # ATR
        high_low = price_data["High"] - price_data["Low"]
        high_close = np.abs(price_data["High"] - price_data["Close"].shift())
        low_close = np.abs(price_data["Low"] - price_data["Close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr_14"] = true_range.rolling(14).mean()

        # Volume indicators
        indicators["volume_sma_20"] = price_data["Volume"].rolling(20).mean()
        indicators["obv"] = (np.sign(price_data["Close"].diff()) * price_data["Volume"]).cumsum()

        # Add symbol column
        indicators["symbol"] = symbol

        # Store in database
        await self._store_indicators(symbol, indicators)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def _store_price_data(self, symbol: str, data: pd.DataFrame):
        """Store price data in database"""
        data_to_store = data.copy()
        data_to_store["symbol"] = symbol
        data_to_store.reset_index(inplace=True)
        data_to_store.rename(columns={"Date": "date"}, inplace=True)

        # Column mapping
        columns = {
            "symbol": "symbol",
            "date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "splits",
            "Returns": "returns",
            "Log_Returns": "log_returns",
        }

        data_to_store = data_to_store[[k for k in columns.keys() if k in data_to_store.columns]]
        data_to_store.rename(columns=columns, inplace=True)

        # Use COPY for efficient bulk insert
        with self.engine.connect() as conn:
            data_to_store.to_sql(
                "historical_prices", conn, if_exists="append", index=False, method="multi"
            )

    async def query_similar_market_conditions(
        self, symbol: str, current_conditions: Dict[str, float], lookback_days: int = 30
    ) -> List[Dict]:
        """Find similar historical market conditions"""

        # Build query based on current conditions
        query = """
        WITH current_metrics AS (
            SELECT
                %(rsi)s as current_rsi,
                %(volume_ratio)s as current_volume_ratio,
                %(volatility)s as current_volatility,
                %(trend)s as current_trend
        ),
        historical_matches AS (
            SELECT
                p.date,
                p.symbol,
                p.close,
                p.returns,
                ti.rsi_14,
                p.volume / NULLIF(ti.volume_sma_20, 0) as volume_ratio,
                ABS(ti.rsi_14 - (SELECT current_rsi FROM current_metrics)) as rsi_diff,
                r.regime,
                r.volatility_regime
            FROM historical_prices p
            JOIN technical_indicators ti ON p.symbol = ti.symbol AND p.date = ti.date
            LEFT JOIN market_regimes r ON p.symbol = r.symbol AND p.date = r.date
            WHERE p.symbol = %(symbol)s
            AND p.date < CURRENT_DATE - INTERVAL '%(lookback)s days'
        )
        SELECT
            date,
            close as price,
            returns,
            rsi_14,
            volume_ratio,
            regime,
            volatility_regime,
            -- Calculate what happened next
            LEAD(returns, 1) OVER (ORDER BY date) as next_day_return,
            LEAD(returns, 5) OVER (ORDER BY date) as next_week_return,
            LEAD(returns, 20) OVER (ORDER BY date) as next_month_return
        FROM historical_matches
        WHERE rsi_diff < 5  -- Similar RSI
        ORDER BY rsi_diff
        LIMIT 20;
        """

        with self.engine.connect() as conn:
            result = conn.execute(
                text(query),
                {
                    "symbol": symbol,
                    "rsi": current_conditions.get("rsi", 50),
                    "volume_ratio": current_conditions.get("volume_ratio", 1),
                    "volatility": current_conditions.get("volatility", 0.02),
                    "trend": current_conditions.get("trend", 0),
                    "lookback": lookback_days,
                },
            )

            similar_conditions = []
            for row in result:
                similar_conditions.append(
                    {
                        "date": row.date,
                        "price": float(row.price),
                        "rsi": float(row.rsi_14) if row.rsi_14 else None,
                        "volume_ratio": float(row.volume_ratio) if row.volume_ratio else None,
                        "regime": row.regime,
                        "next_day_return": float(row.next_day_return)
                        if row.next_day_return
                        else None,
                        "next_week_return": float(row.next_week_return)
                        if row.next_week_return
                        else None,
                        "next_month_return": float(row.next_month_return)
                        if row.next_month_return
                        else None,
                    }
                )

        return similar_conditions

    async def get_regime_analysis(self, symbol: str, date: Optional[str] = None) -> Dict:
        """Analyze market regime for a symbol"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        query = """
        WITH price_trends AS (
            SELECT
                date,
                close,
                sma_20,
                sma_50,
                sma_200,
                CASE
                    WHEN close > sma_20 AND sma_20 > sma_50 AND sma_50 > sma_200 THEN 'Strong Uptrend'
                    WHEN close > sma_50 AND sma_50 > sma_200 THEN 'Uptrend'
                    WHEN close < sma_20 AND sma_20 < sma_50 AND sma_50 < sma_200 THEN 'Strong Downtrend'
                    WHEN close < sma_50 AND sma_50 < sma_200 THEN 'Downtrend'
                    ELSE 'Ranging'
                END as trend_regime,
                atr_14 / close as volatility_normalized
            FROM historical_prices p
            JOIN technical_indicators ti ON p.symbol = ti.symbol AND p.date = ti.date
            WHERE p.symbol = %(symbol)s
            AND p.date <= %(date)s
            ORDER BY date DESC
            LIMIT 20
        )
        SELECT
            trend_regime,
            AVG(volatility_normalized) as avg_volatility,
            COUNT(*) as regime_days
        FROM price_trends
        GROUP BY trend_regime
        ORDER BY regime_days DESC
        LIMIT 1;
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"symbol": symbol, "date": date}).fetchone()

            if result:
                return {
                    "regime": result.trend_regime,
                    "volatility": float(result.avg_volatility),
                    "confidence": min(result.regime_days / 20.0, 1.0),
                }

        return {"regime": "Unknown", "volatility": 0, "confidence": 0}

    async def get_performance_stats(self, symbol: str, period_years: int = 30) -> Dict:
        """Get comprehensive performance statistics"""
        query = """
        WITH performance_data AS (
            SELECT
                returns,
                date,
                EXTRACT(YEAR FROM date) as year,
                close
            FROM historical_prices
            WHERE symbol = %(symbol)s
            AND date >= CURRENT_DATE - INTERVAL '%(years)s years'
        )
        SELECT
            COUNT(*) as total_days,
            AVG(returns) * 252 as annual_return,
            STDDEV(returns) * SQRT(252) as annual_volatility,
            (AVG(returns) / NULLIF(STDDEV(returns), 0)) * SQRT(252) as sharpe_ratio,
            MIN(close) as min_price,
            MAX(close) as max_price,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY returns) as var_95,
            COUNT(DISTINCT year) as years_of_data
        FROM performance_data;
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"symbol": symbol, "years": period_years}).fetchone()

            if result:
                return {
                    "total_trading_days": result.total_days,
                    "annual_return": float(result.annual_return) if result.annual_return else 0,
                    "annual_volatility": float(result.annual_volatility)
                    if result.annual_volatility
                    else 0,
                    "sharpe_ratio": float(result.sharpe_ratio) if result.sharpe_ratio else 0,
                    "min_price": float(result.min_price) if result.min_price else 0,
                    "max_price": float(result.max_price) if result.max_price else 0,
                    "value_at_risk_95": float(result.var_95) if result.var_95 else 0,
                    "years_of_data": result.years_of_data,
                }

        return {}


# Singleton instance
historical_data_service = HistoricalDataService()


async def get_historical_context(symbol: str, current_price: float) -> Dict:
    """Get historical context for current market conditions"""
    # Get current technical indicators
    current_conditions = {
        "rsi": 50,  # Would calculate from recent data
        "volume_ratio": 1.0,
        "volatility": 0.02,
        "trend": 0.05,
    }

    # Find similar historical setups
    similar_setups = await historical_data_service.query_similar_market_conditions(
        symbol, current_conditions
    )

    # Get current regime
    regime = await historical_data_service.get_regime_analysis(symbol)

    # Get long-term stats
    stats = await historical_data_service.get_performance_stats(symbol)

    return {
        "similar_historical_setups": similar_setups,
        "current_regime": regime,
        "long_term_stats": stats,
        "recommendation": _generate_recommendation(similar_setups, regime, stats),
    }


def _generate_recommendation(similar_setups: List[Dict], regime: Dict, stats: Dict) -> str:
    """Generate recommendation based on historical analysis"""
    if not similar_setups:
        return "Insufficient historical data for comparison"

    # Analyze outcomes of similar setups
    positive_outcomes = sum(1 for s in similar_setups if s.get("next_week_return", 0) > 0)
    avg_return = np.mean(
        [s.get("next_week_return", 0) for s in similar_setups if s.get("next_week_return")]
    )

    win_rate = positive_outcomes / len(similar_setups)

    if win_rate > 0.65 and avg_return > 0.02:
        return f"Historical data shows {win_rate:.0%} win rate with {avg_return:.1%} avg return in similar conditions"
    elif win_rate < 0.35:
        return f"Historical data shows only {win_rate:.0%} win rate in similar conditions - exercise caution"
    else:
        return f"Mixed historical outcomes ({win_rate:.0%} win rate) in similar conditions"
