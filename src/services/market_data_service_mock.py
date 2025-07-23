#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI V3 - Mock Market Data Service
For demonstration and testing without API rate limits
"""

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# Import the real service classes
from market_data_service import (
    DataUnavailableReason,
    MarketDataCache,
    MarketDataError,
    MarketHours,
    MarketTick,
    SignalData,
    TechnicalIndicators,
)

logger = logging.getLogger(__name__)

class MockMarketDataService:
    """Mock market data service for testing without API calls"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'IWM']
        self.indicators = TechnicalIndicators()
        self.cache = MarketDataCache()
        
        # Mock prices
        self.mock_prices = {
            'AAPL': 175.50,
            'GOOGL': 140.25,
            'MSFT': 380.75,
            'TSLA': 250.30,
            'AMZN': 155.40,
            'NVDA': 890.25,
            'META': 485.60,
            'SPY': 475.80,
            'QQQ': 440.30,
            'IWM': 195.20
        }
    
    def check_market_hours(self) -> MarketHours:
        """Check if market is currently open"""
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()
        
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        is_weekday = now.weekday() < 5
        is_within_hours = market_open <= current_time <= market_close
        is_open = is_weekday and is_within_hours
        
        # Calculate next market open
        if not is_open:
            if current_time < market_open and is_weekday:
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                reason = "Pre-market hours"
            elif current_time > market_close and is_weekday:
                next_open = now + timedelta(days=1)
                if next_open.weekday() >= 5:
                    days_until_monday = 7 - next_open.weekday()
                    next_open += timedelta(days=days_until_monday)
                next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
                reason = "After-hours trading"
            else:
                days_until_monday = 7 - now.weekday() if now.weekday() >= 5 else 1
                next_open = now + timedelta(days=days_until_monday)
                next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
                reason = "Weekend - market closed"
        else:
            next_open = now
            reason = "Market is open"
        
        return MarketHours(
            is_open=is_open,
            current_time=now,
            market_open=market_open,
            market_close=market_close,
            next_open=next_open,
            reason=reason,
            timezone="US/Eastern"
        )
    
    def fetch_real_time_data(self, symbol: str) -> Tuple[Optional[MarketTick], Optional[MarketDataError]]:
        """Fetch mock market data"""
        market_hours = self.check_market_hours()
        
        # Check for invalid symbol
        if symbol not in self.mock_prices and symbol != "INVALID_SYMBOL":
            error = MarketDataError(
                symbol=symbol,
                reason=DataUnavailableReason.INVALID_SYMBOL,
                message=f"Symbol {symbol} not found",
                is_recoverable=False,
                suggested_action="Check symbol spelling",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            return None, error
        
        # Simulate market closed scenario sometimes
        if not market_hours.is_open and random.random() > 0.3:
            # Try cache first
            cached_tick = self.cache.get_tick(symbol)
            if cached_tick:
                logger.info(f"Using cached data for {symbol} (market closed)")
                return cached_tick, None
            
            error = MarketDataError(
                symbol=symbol,
                reason=DataUnavailableReason.MARKET_CLOSED,
                message=f"Market is closed. {market_hours.reason}. Next open: {market_hours.next_open}",
                is_recoverable=True,
                suggested_action="Use cached data or wait for market open",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            return None, error
        
        # Generate mock tick data
        base_price = self.mock_prices.get(symbol, 100.0)
        
        # Add some randomness
        change_percent = random.uniform(-2, 2)
        price = base_price * (1 + change_percent / 100)
        
        tick = MarketTick(
            symbol=symbol,
            price=float(price),
            volume=random.randint(1000000, 50000000),
            bid=float(price * 0.999),
            ask=float(price * 1.001),
            spread=float(price * 0.002),
            timestamp=datetime.now(timezone.utc).isoformat(),
            change=float(price - base_price),
            change_percent=float(change_percent)
        )
        
        # Cache the data
        self.cache.save_tick(symbol, tick)
        
        return tick, None
    
    def get_historical_data(self, symbol: str, period: str = "3mo") -> Tuple[pd.DataFrame, Optional[MarketDataError]]:
        """Generate mock historical data"""
        market_hours = self.check_market_hours()
        
        # Check cache first
        cached_data = self.cache.get_historical(symbol)
        if cached_data is not None and not cached_data.empty:
            return cached_data, None
        
        # Check for invalid symbol
        if symbol not in self.mock_prices and symbol != "INVALID_SYMBOL":
            error = MarketDataError(
                symbol=symbol,
                reason=DataUnavailableReason.INVALID_SYMBOL,
                message=f"Symbol {symbol} not found",
                is_recoverable=False,
                suggested_action="Check symbol spelling",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            return pd.DataFrame(), error
        
        # Generate mock data
        base_price = self.mock_prices.get(symbol, 100.0)
        days = 90  # 3 months of data
        
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=days, freq='D')
        
        # Generate price series with trend and volatility
        trend = random.uniform(-0.001, 0.002)  # Daily trend
        volatility = random.uniform(0.01, 0.03)  # Daily volatility
        
        prices = []
        current_price = base_price
        
        for i in range(days):
            # Random walk with trend
            daily_return = trend + random.gauss(0, volatility)
            current_price *= (1 + daily_return)
            
            # OHLCV data
            open_price = current_price * (1 + random.uniform(-0.005, 0.005))
            high = current_price * (1 + random.uniform(0, 0.02))
            low = current_price * (1 + random.uniform(-0.02, 0))
            close = current_price
            volume = random.randint(10000000, 100000000)
            
            prices.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        data = pd.DataFrame(prices, index=dates)
        
        # Calculate technical indicators
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (2 * bb_std)
        data['BB_Lower'] = data['BB_Middle'] - (2 * bb_std)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Additional features
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        data['Volatility'] = data['Price_Change'].rolling(20).std()
        data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
        
        clean_data = data.dropna()
        
        # Cache the data
        if not clean_data.empty:
            self.cache.save_historical(symbol, clean_data)
        
        return clean_data, None
    
    def generate_signal(self, symbol: str) -> Optional[SignalData]:
        """Generate mock trading signal"""
        market_hours = self.check_market_hours()
        is_after_hours = not market_hours.is_open
        
        # Get historical data
        hist_data, error = self.get_historical_data(symbol)
        
        if hist_data.empty or len(hist_data) < 10:
            if is_after_hours and error:
                logger.info(f"Generating HOLD signal for {symbol} - {error.message}")
                return SignalData(
                    symbol=symbol,
                    signal_type='HOLD',
                    confidence=0.5,
                    price_target=0.0,
                    stop_loss=0.0,
                    risk_score=0.5,
                    indicators={},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    is_after_hours=True
                )
            return None
        
        # Mock signal generation based on technical indicators
        latest = hist_data.iloc[-1]
        
        # Simple signal logic based on RSI and MACD
        rsi = latest['RSI']
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        current_price = latest['Close']
        
        # Determine signal
        if rsi < 30 and macd > macd_signal:
            signal_type = 'BUY'
            confidence = 0.75
            price_target = current_price * 1.05
            stop_loss = current_price * 0.97
        elif rsi > 70 and macd < macd_signal:
            signal_type = 'SELL'
            confidence = 0.75
            price_target = current_price * 0.95
            stop_loss = current_price * 1.03
        else:
            signal_type = 'HOLD'
            confidence = 0.6
            price_target = current_price
            stop_loss = current_price
        
        # Adjust confidence for after-hours
        if is_after_hours:
            confidence *= 0.8
        
        # Risk score based on volatility
        risk_score = min(latest['Volatility'] * 10, 1.0)
        
        # Calculate indicators
        indicators = self.indicators.calculate_indicators(hist_data)
        
        if is_after_hours:
            indicators['after_hours_data'] = True
            indicators['market_status'] = market_hours.reason
        
        return SignalData(
            symbol=symbol,
            signal_type=signal_type,
            confidence=float(confidence),
            price_target=float(price_target),
            stop_loss=float(stop_loss),
            risk_score=float(risk_score),
            indicators=indicators,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_after_hours=is_after_hours
        )
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get mock market summary"""
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbols': {},
            'market_status': 'OPEN' if self.check_market_hours().is_open else 'CLOSED',
            'total_symbols': len(self.symbols)
        }
        
        for symbol in self.symbols:
            tick, error = self.fetch_real_time_data(symbol)
            if tick:
                summary['symbols'][symbol] = {
                    'price': tick.price,
                    'change': tick.change,
                    'change_percent': tick.change_percent,
                    'volume': tick.volume
                }
        
        return summary 