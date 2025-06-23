"""
Market Data Models for GoldenSignalsAI V3
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator


class OHLCV(BaseModel):
    """Open, High, Low, Close, Volume data point"""
    
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    
    @validator("high")
    def high_must_be_highest(cls, v, values) -> bool:
        """Validate high is the highest price"""
        if "open" in values and "low" in values and "close" in values:
            prices = [values["open"], values["low"], values["close"], v]
            if v != max(prices):
                raise ValueError("High must be the highest price")
        return v
    
    @validator("low")
    def low_must_be_lowest(cls, v, values) -> bool:
        """Validate low is the lowest price"""
        if "open" in values and "close" in values:
            prices = [values["open"], values["close"], v]
            if v != min(prices):
                raise ValueError("Low must be the lowest price")
        return v


class VolumeProfile(BaseModel):
    """Volume profile data for a trading session"""
    
    price_levels: List[float] = Field(..., description="Price levels")
    volumes: List[int] = Field(..., description="Volume at each price level")
    poc: float = Field(..., description="Point of Control (highest volume)")
    value_area_high: float = Field(..., description="Value Area High")
    value_area_low: float = Field(..., description="Value Area Low")
    
    @validator("volumes")
    def volumes_match_prices(cls, v, values) -> bool:
        """Ensure volumes list matches price levels"""
        if "price_levels" in values and len(v) != len(values["price_levels"]):
            raise ValueError("Volumes must match price levels length")
        return v


class OrderBook(BaseModel):
    """Level 2 order book data"""
    
    timestamp: datetime
    bids: List[List[float]] = Field(..., description="Bid prices and sizes")
    asks: List[List[float]] = Field(..., description="Ask prices and sizes")
    
    def get_bid_ask_spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            best_bid = max(bid[0] for bid in self.bids)
            best_ask = min(ask[0] for ask in self.asks)
            return best_ask - best_bid
        return 0.0
    
    def get_market_depth(self, levels: int = 5) -> Dict[str, float]:
        """Calculate market depth for specified levels"""
        bid_depth = sum(bid[1] for bid in sorted(self.bids, reverse=True)[:levels])
        ask_depth = sum(ask[1] for ask in sorted(self.asks)[:levels])
        return {"bid_depth": bid_depth, "ask_depth": ask_depth}


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    
    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum indicators
    rsi: Optional[float] = Field(None, ge=0, le=100)
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stochastic_k: Optional[float] = Field(None, ge=0, le=100)
    stochastic_d: Optional[float] = Field(None, ge=0, le=100)
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = Field(None, ge=0)
    
    # Volume indicators
    volume_sma: Optional[float] = None
    obv: Optional[float] = None  # On Balance Volume
    vwap: Optional[float] = None  # Volume Weighted Average Price
    
    # Support/Resistance
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)


class MarketConditions(BaseModel):
    """Current market regime and conditions"""
    
    volatility_regime: Optional[str] = Field(None, pattern="^(low|normal|high|extreme)$")
    trend_direction: Optional[str] = Field(None, pattern="^(up|down|sideways)$")
    trend_strength: Optional[float] = Field(None, ge=0, le=1)
    
    # Market structure
    market_phase: Optional[str] = Field(
        None, 
        pattern="^(accumulation|markup|distribution|markdown)$"
    )
    
    # Risk metrics
    vix: Optional[float] = Field(None, ge=0)  # Volatility Index
    correlation_to_spy: Optional[float] = Field(None, ge=-1, le=1)
    
    # Sentiment indicators
    put_call_ratio: Optional[float] = Field(None, ge=0)
    fear_greed_index: Optional[float] = Field(None, ge=0, le=100)


class NewsItem(BaseModel):
    """Individual news item"""
    
    timestamp: datetime
    headline: str
    content: Optional[str] = None
    source: str
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    impact_score: Optional[float] = Field(None, ge=0, le=1)
    categories: List[str] = Field(default_factory=list)


class MarketData(BaseModel):
    """
    Comprehensive market data container for a specific symbol
    """
    
    # Basic identification
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core price data
    current_price: float = Field(..., gt=0)
    previous_close: Optional[float] = Field(None, gt=0)
    day_change: Optional[float] = None
    day_change_percent: Optional[float] = None
    
    # Historical OHLCV data
    ohlcv_1m: List[OHLCV] = Field(default_factory=list, description="1-minute bars")
    ohlcv_5m: List[OHLCV] = Field(default_factory=list, description="5-minute bars")
    ohlcv_1h: List[OHLCV] = Field(default_factory=list, description="1-hour bars")
    ohlcv_1d: List[OHLCV] = Field(default_factory=list, description="Daily bars")
    
    # Market microstructure
    order_book: Optional[OrderBook] = None
    volume_profile: Optional[VolumeProfile] = None
    
    # Technical analysis
    indicators: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    
    # Market conditions and regime
    conditions: MarketConditions = Field(default_factory=MarketConditions)
    
    # News and sentiment
    recent_news: List[NewsItem] = Field(default_factory=list)
    
    # Options data (if applicable)
    options_data: Optional[Dict] = None
    
    # Metadata
    data_quality: float = Field(default=1.0, ge=0, le=1, description="Data quality score")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("day_change", pre=True, always=True)
    def calculate_day_change(cls, v, values) -> float:
        """Auto-calculate day change if not provided"""
        if v is None and "current_price" in values and "previous_close" in values:
            if values["previous_close"]:
                return values["current_price"] - values["previous_close"]
        return v
    
    @validator("day_change_percent", pre=True, always=True)
    def calculate_day_change_percent(cls, v, values) -> float:
        """Auto-calculate day change percentage if not provided"""
        if v is None and "day_change" in values and "previous_close" in values:
            if values["previous_close"] and values["day_change"] is not None:
                return (values["day_change"] / values["previous_close"]) * 100
        return v
    
    def get_latest_ohlcv(self, timeframe: str = "1m") -> Optional[OHLCV]:
        """Get the most recent OHLCV data for specified timeframe"""
        data_map = {
            "1m": self.ohlcv_1m,
            "5m": self.ohlcv_5m,
            "1h": self.ohlcv_1h,
            "1d": self.ohlcv_1d
        }
        
        if timeframe in data_map and data_map[timeframe]:
            return max(data_map[timeframe], key=lambda x: x.timestamp)
        return None
    
    def calculate_volatility(self, periods: int = 20, timeframe: str = "1d") -> Optional[float]:
        """Calculate historical volatility"""
        ohlcv_data = {
            "1m": self.ohlcv_1m,
            "5m": self.ohlcv_5m,
            "1h": self.ohlcv_1h,
            "1d": self.ohlcv_1d
        }.get(timeframe, [])
        
        if len(ohlcv_data) >= periods:
            returns = []
            for i in range(1, min(periods + 1, len(ohlcv_data))):
                prev_close = ohlcv_data[-i-1].close
                curr_close = ohlcv_data[-i].close
                returns.append((curr_close - prev_close) / prev_close)
            
            if returns:
                return float(np.std(returns) * np.sqrt(252))  # Annualized volatility
        
        return None
    
    def get_price_trend(self, periods: int = 20) -> Optional[str]:
        """Determine price trend over specified periods"""
        if len(self.ohlcv_1d) >= periods:
            recent_closes = [bar.close for bar in self.ohlcv_1d[-periods:]]
            
            # Simple trend detection using linear regression slope
            x = np.arange(len(recent_closes))
            slope = np.polyfit(x, recent_closes, 1)[0]
            
            if slope > 0.001:  # Threshold for uptrend
                return "up"
            elif slope < -0.001:  # Threshold for downtrend
                return "down"
            else:
                return "sideways"
        
        return None
    
    def is_data_stale(self, max_age_minutes: int = 5) -> bool:
        """Check if market data is stale"""
        age = (datetime.utcnow() - self.last_updated).total_seconds() / 60
        return age > max_age_minutes 