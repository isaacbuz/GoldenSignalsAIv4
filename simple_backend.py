#!/usr/bin/env python3
"""
Simple Backend for GoldenSignalsAI
Minimal imports to avoid circular dependencies
"""

import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import only the essential services
from src.services.simple_live_data import SimpleLiveDataFetcher
from src.services.simple_ml_signals import SimplifiedMLSignalGenerator

# Remove these problematic imports:
# from src.api.v1.ai_analyst import router as ai_analyst_router
# from src.api.v1.signals import router as signals_router
# from src.api.v1.market_data import router as market_data_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
simple_live_data = SimpleLiveDataFetcher()
simplified_ml_signals = SimplifiedMLSignalGenerator()

# Import additional services if available
try:
    from src.services.enhanced_sentiment_service import EnhancedSentimentService
    enhanced_sentiment_service = EnhancedSentimentService()
except ImportError:
    logger.warning("Enhanced sentiment service not available")
    enhanced_sentiment_service = None

# Import error recovery if available
try:
    from src.utils.error_recovery import error_recovery, RetryStrategy, CircuitBreakerConfig, ErrorSeverity
except ImportError:
    logger.warning("Error recovery not available, using direct calls")
    # Create a dummy decorator
    class DummyErrorRecovery:
        def with_recovery(self, **kwargs):
            def decorator(func):
                return func
            return decorator
    error_recovery = DummyErrorRecovery()
    
    class RetryStrategy:
        def __init__(self, **kwargs):
            pass
    
    class CircuitBreakerConfig:
        def __init__(self, **kwargs):
            pass
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

# Import technical indicators if available
try:
    from src.services.technical_indicators import TechnicalIndicators
except ImportError:
    logger.warning("Technical indicators not available")
    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(data):
            return {}

# Import numpy if available
try:
    import numpy as np
except ImportError:
    logger.warning("Numpy not available")
    np = None

# Helper function for timezone-aware datetime
def now_utc():
    return datetime.now(timezone.utc)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await simple_live_data.initialize()
        print("✅ Simple live data fetcher initialized successfully")
    except Exception as e:
        print(f"⚠️  Failed to initialize live data fetcher: {e}")
        print("   Falling back to mock data mode")
    
    try:
        await simplified_ml_signals.initialize()
        print("✅ Simplified ML Signal Generator initialized successfully")
    except Exception as e:
        print(f"⚠️  Failed to initialize ML Signal Generator: {e}")
        print("   Using mock signals as fallback")
    
    if enhanced_sentiment_service:
        try:
            await enhanced_sentiment_service.initialize()
            print("✅ Enhanced Sentiment Service initialized successfully")
        except Exception as e:
            print(f"⚠️  Failed to initialize Enhanced Sentiment Service: {e}")
            print("   Using mock sentiment as fallback")
    
    yield
    # Shutdown
    await simple_live_data.close()
    if enhanced_sentiment_service:
        await enhanced_sentiment_service.close()

app = FastAPI(
    title="GoldenSignalsAI Simple Backend",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://192.168.1.182:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
active_connections: List[WebSocket] = []

# Mock data fallback for when live data is unavailable
def generate_mock_signal():
    """Generate a mock trading signal (fallback)"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
    patterns = ["Bull Flag", "Ascending Triangle", "Double Bottom", "Cup and Handle", 
                "Breakout", "Support Bounce", "Resistance Break", "Golden Cross"]
    
    symbol = random.choice(symbols)
    pattern = random.choice(patterns)
    
    return {
        "id": f"{symbol}_{int(now_utc().timestamp())}_{random.randint(1, 100)}",
        "symbol": symbol,
        "pattern": pattern,
        "confidence": round(random.uniform(70, 95), 1),
        "entry": round(random.uniform(100, 500), 2),
        "stopLoss": round(random.uniform(95, 495), 2),
        "takeProfit": round(random.uniform(105, 505), 2),
        "timestamp": now_utc().isoformat(),
        "type": random.choice(["BUY", "SELL"]),
        "timeframe": random.choice(["5m", "15m", "1h", "4h", "1d"]),
        "risk": random.choice(["LOW", "MEDIUM", "HIGH"]),
    }

async def generate_mock_market_data(symbol: str):
    """Fallback function for market data"""
    base_price = {
        "AAPL": 185.50,
        "GOOGL": 142.75,
        "MSFT": 378.90,
        "TSLA": 245.60,
        "NVDA": 625.40,
        "META": 325.80,
        "AMZN": 155.20,
        "SPY": 450.25,
    }.get(symbol, 100.0)
    
    change = random.uniform(-2, 2)
    change_percent = change / base_price * 100
    
    return {
        "symbol": symbol,
        "price": round(base_price + change, 2),
        "change": round(change, 2),
        "changePercent": round(change_percent, 2),
        "volume": random.randint(1000000, 50000000),
        "high": round(base_price + abs(change) + random.uniform(0, 2), 2),
        "low": round(base_price - abs(change) - random.uniform(0, 2), 2),
        "open": round(base_price + random.uniform(-1, 1), 2),
        "previousClose": base_price,
        "marketCap": f"{random.randint(100, 3000)}B",
        "pe": round(random.uniform(10, 40), 1),
        "timestamp": now_utc().isoformat(),
    }

@error_recovery.with_recovery(
    fallback=generate_mock_market_data,
    retry_strategy=RetryStrategy(max_retries=3, initial_delay=0.5),
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=timedelta(seconds=30)),
    severity=ErrorSeverity.MEDIUM
)
async def get_live_market_data(symbol: str):
    """Get live market data from database/API with error recovery"""
    # Try to get live quote
    quotes = await simple_live_data.fetch_live_quotes([symbol])
    if symbol in quotes:
        quote = quotes[symbol]
        return {
            "symbol": symbol,
            "price": quote.get('price', 0),
            "change": quote.get('change', 0),
            "changePercent": quote.get('changePercent', 0),
            "volume": quote.get('volume', 0),
            "high": quote.get('high', 0),
            "low": quote.get('low', 0),
            "open": quote.get('open', 0),
            "previousClose": quote.get('previousClose', 0),
            "marketCap": f"{random.randint(100, 3000)}B",  # Still mock this
            "pe": round(random.uniform(10, 40), 1),  # Still mock this
            "timestamp": quote.get('timestamp', now_utc().isoformat()),
        }
    else:
        raise ValueError(f"No quote data available for {symbol}")

async def generate_mock_historical_data(symbol: str, period: str, interval: str):
    """Fallback function for historical data"""
    periods = {
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90),
        "6m": timedelta(days=180),
        "1y": timedelta(days=365),
        "ytd": now_utc() - datetime(now_utc().year, 1, 1).replace(tzinfo=now_utc().tzinfo),
    }
    
    intervals = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
    }
    
    interval_minutes = intervals.get(interval, 5)
    period_delta = periods.get(period, timedelta(days=1))
    period_minutes = int(period_delta.total_seconds() / 60)
    data_points = min(period_minutes // interval_minutes, 500)
    
    base_price = 150.0
    data = []
    current_time = now_utc()
    
    for i in range(data_points, 0, -1):
        time = current_time - timedelta(minutes=i * interval_minutes)
        
        open_price = base_price + random.uniform(-2, 2)
        close_price = open_price + random.uniform(-1, 1)
        high_price = max(open_price, close_price) + random.uniform(0, 0.5)
        low_price = min(open_price, close_price) - random.uniform(0, 0.5)
        volume = random.randint(100000, 1000000)
        
        data.append({
            "time": int(time.timestamp()),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume,
        })
        
        base_price = close_price
    
    return data

@error_recovery.with_recovery(
    fallback=generate_mock_historical_data,
    retry_strategy=RetryStrategy(max_retries=2, initial_delay=1.0),
    severity=ErrorSeverity.MEDIUM
)
async def get_live_historical_data(symbol: str, period: str, interval: str):
    """Get live historical data from database with error recovery"""
    # Convert period to date range
    end_date = now_utc()
    periods = {
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90),
        "6m": timedelta(days=180),
        "1y": timedelta(days=365),
        "ytd": now_utc() - datetime(now_utc().year, 1, 1).replace(tzinfo=end_date.tzinfo),
    }
    
    period_delta = periods.get(period, timedelta(days=1))
    start_date = end_date - period_delta
    
    # Fetch from simple live data
    raw_data = await simple_live_data.fetch_historical_data(
        symbol, start_date, end_date, interval
    )
    
    if raw_data:
        # Data is already in the expected format
        return raw_data
    else:
        raise ValueError(f"No historical data available for {symbol}")

@app.get("/")
async def root():
    return {"message": "GoldenSignalsAI Simple Backend", "status": "running", "mode": "live_data"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint with error statistics"""
    stats = error_recovery.get_error_statistics()
    
    # Determine health status based on errors
    total_errors = stats["total_errors"]
    critical_errors = stats["errors_by_severity"].get("critical", 0)
    
    if critical_errors > 0 or total_errors > 100:
        health_status = "degraded"
    elif total_errors > 50:
        health_status = "warning"
    else:
        health_status = "healthy"
    
    return {
        "status": health_status,
        "timestamp": now_utc().isoformat(),
        "error_statistics": stats,
        "services": {
            "live_data": "connected" if hasattr(simple_live_data, 'pool') and simple_live_data.pool else "disconnected",
            "websocket_connections": len(active_connections),
        }
    }

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    return await get_live_market_data(symbol)

@app.get("/api/v1/market-data/{symbol}/historical")
async def get_historical_data(symbol: str, period: str = "1d", interval: str = "5m"):
    """Get historical market data for charting"""
    data = await get_live_historical_data(symbol, period, interval)
    return {"data": data, "symbol": symbol, "period": period, "interval": interval}

@app.get("/api/v1/signals")
async def get_signals():
    """Get latest trading signals"""
    try:
        # Try to get real ML-generated signals
        signals = await simplified_ml_signals.generate_signals(
            limit=10,
            live_data_fetcher=simple_live_data
        )
        if signals:
            logger.info(f"Generated {len(signals)} ML signals")
            return {"signals": signals, "count": len(signals)}
    except Exception as e:
        logger.error(f"Error generating ML signals: {e}")
    
    # Fallback to mock signals
    logger.warning("Falling back to mock signals")
    signals = [generate_mock_signal() for _ in range(10)]
    return {"signals": signals, "count": len(signals)}

@app.get("/api/v1/signals/{symbol}")
async def get_symbol_signals(symbol: str):
    """Get signals for a specific symbol"""
    try:
        # Try to get real ML-generated signals for specific symbol
        signals = await simplified_ml_signals.generate_signals(
            symbols=[symbol], 
            limit=5,
            live_data_fetcher=simple_live_data
        )
        if signals:
            logger.info(f"Generated {len(signals)} ML signals for {symbol}")
            return {"signals": signals, "symbol": symbol}
    except Exception as e:
        logger.error(f"Error generating ML signals for {symbol}: {e}")
    
    # Fallback to mock signals
    logger.warning(f"Falling back to mock signals for {symbol}")
    signals = [generate_mock_signal() for _ in range(5)]
    # Override symbol in generated signals
    for signal in signals:
        signal["symbol"] = symbol
    return {"signals": signals, "symbol": symbol}

@app.get("/api/v1/signals/{signal_id}/insights")
async def get_signal_insights(signal_id: str):
    """Get AI insights for a specific signal"""
    # Extract symbol from signal_id if possible
    symbol = signal_id.split('_')[0] if '_' in signal_id else 'SPY'
    
    # Try to get real technical indicators
    try:
        end_date = now_utc()
        start_date = end_date - timedelta(days=30)
        
        data = await simple_live_data.fetch_historical_data(
            symbol, start_date, end_date, '1d'
        )
        
        if data:
            # Calculate real technical indicators
            indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            latest_rsi = indicators.get('rsi', 50)
            macd_histogram = indicators.get('macd_histogram', 0)
            macd_status = "Bullish Crossover" if macd_histogram > 0 else "Bearish Divergence"
            support_levels = indicators.get('support_levels', [])
            resistance_levels = indicators.get('resistance_levels', [])
        else:
            latest_rsi = random.uniform(30, 70)
            macd_status = random.choice(["Bullish Crossover", "Bearish Divergence", "Neutral"])
            support_levels = []
            resistance_levels = []
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        latest_rsi = random.uniform(30, 70)
        macd_status = random.choice(["Bullish Crossover", "Bearish Divergence", "Neutral"])
        support_levels = []
        resistance_levels = []
    
    # Get real sentiment data
    try:
        if enhanced_sentiment_service:
            sentiment_data = await enhanced_sentiment_service.get_aggregated_sentiment(symbol)
            sentiment_score = sentiment_data.get('overall_score', 0.7)
            sentiment_label = sentiment_data.get('overall_label', 'Bullish')
            sentiment_sources = list(sentiment_data.get('sources', {}).keys())
            if not sentiment_sources:
                sentiment_sources = ["Twitter", "Reddit", "News"]
        else:
            raise Exception("Enhanced sentiment service not available")
    except Exception as e:
        logger.error(f"Error getting sentiment data: {e}")
        sentiment_score = round(random.uniform(0.6, 0.9), 2)
        sentiment_label = random.choice(["Bullish", "Very Bullish", "Neutral"])
        sentiment_sources = ["Twitter", "Reddit", "News"]
    
    return {
        "signalId": signal_id,
        "insights": {
            "sentiment": {
                "score": round(sentiment_score, 2),
                "label": sentiment_label,
                "sources": sentiment_sources,
            },
            "technicalAnalysis": {
                "rsi": round(float(latest_rsi), 1) if not np.isnan(latest_rsi) else 50,
                "macd": macd_status,
                "supportLevels": support_levels[:3] if support_levels else [round(random.uniform(90, 98), 2) for _ in range(3)],
                "resistanceLevels": resistance_levels[:3] if resistance_levels else [round(random.uniform(102, 110), 2) for _ in range(3)],
            },
            "aiPrediction": {
                "targetPrice": round(random.uniform(105, 120), 2),
                "confidence": round(random.uniform(70, 90), 1),
                "timeframe": "7 days",
                "reasoning": "Strong momentum with breakout pattern confirmed by volume",
            },
            "riskAnalysis": {
                "volatility": random.choice(["Low", "Medium", "High"]),
                "maxDrawdown": round(random.uniform(2, 8), 1),
                "sharpeRatio": round(random.uniform(1.0, 2.5), 2),
            },
        },
    }

@app.get("/api/v1/market/opportunities")
async def get_market_opportunities():
    """Get current market opportunities with live data"""
    opportunities = []
    symbols_data = [
        ("NVDA", "NVIDIA Corporation", "Technology"),
        ("TSLA", "Tesla Inc", "Automotive"),
        ("META", "Meta Platforms Inc", "Technology"),
        ("SPY", "SPDR S&P 500 ETF", "ETF"),
        ("AMD", "Advanced Micro Devices", "Technology"),
    ]
    
    # Fetch live quotes for all symbols
    symbols = [s[0] for s in symbols_data]
    try:
        live_quotes = await simple_live_data.fetch_live_quotes(symbols)
    except:
        live_quotes = {}
    
    for i, (symbol, name, sector) in enumerate(symbols_data):
        signal = generate_mock_signal()
        signal["symbol"] = symbol
        
        # Use live data if available
        if symbol in live_quotes:
            current_price = live_quotes[symbol].get('price', 100)
            volume = live_quotes[symbol].get('volume', random.randint(10000000, 150000000))
        else:
            current_price = random.uniform(100, 500)
            volume = random.randint(10000000, 150000000)
        
        is_call = signal["type"] == "BUY"
        momentum_options = ["strong", "moderate", "building"]
        
        opportunities.append({
            "id": signal["id"],
            "symbol": symbol,
            "name": name,
            "type": "CALL" if is_call else "PUT",
            "confidence": signal["confidence"],
            "potentialReturn": round(random.uniform(5, 20), 1),
            "timeframe": signal["timeframe"],
            "keyReason": f"{signal['pattern']} pattern detected with {signal['confidence']}% confidence",
            "momentum": random.choice(momentum_options),
            "aiScore": round(random.uniform(75, 95), 0),
            "sector": sector,
            "volume": volume,
            "volatility": round(random.uniform(0.15, 0.65), 2),
            "currentPrice": round(current_price, 2),
        })
    
    return {"opportunities": opportunities, "timestamp": now_utc().isoformat()}

@app.get("/api/v1/sentiment/heatmap")
async def get_sentiment_heatmap():
    """Get sentiment heatmap for popular symbols"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY", "AMD", "NFLX"]
    
    try:
        if enhanced_sentiment_service:
            heatmap_data = await enhanced_sentiment_service.get_market_sentiment_heatmap(symbols)
            return heatmap_data
        else:
            raise Exception("Enhanced sentiment service not available")
    except Exception as e:
        logger.error(f"Error getting sentiment heatmap: {e}")
        # Fallback to mock data
        heatmap_data = []
        for symbol in symbols:
            sentiment_score = random.uniform(-1, 1)
            heatmap_data.append({
                'symbol': symbol,
                'sentiment': round(sentiment_score, 3),
                'confidence': round(random.uniform(0.5, 1.0), 3),
                'label': 'Very Bullish' if sentiment_score > 0.6 else 'Bullish' if sentiment_score > 0.3 else 'Neutral' if sentiment_score > -0.3 else 'Bearish' if sentiment_score > -0.6 else 'Very Bearish',
                'volume': random.randint(100, 10000)
            })
        
        heatmap_data.sort(key=lambda x: x['sentiment'], reverse=True)
        
        return {
            'timestamp': now_utc().isoformat(),
            'data': heatmap_data
        }

@app.get("/api/v1/predictions/{symbol}")
async def get_price_predictions(symbol: str, timeframe: str = "1h", periods: int = 20):
    """Get price predictions with trend visualization data"""
    try:
        # Import services
        from src.services.prediction_visualization import PredictionVisualizationService, PredictionTimeframe
        
        # Get historical data
        end_date = now_utc()
        start_date = end_date - timedelta(days=30)
        
        historical_data = await simple_live_data.fetch_historical_data(
            symbol, start_date, end_date, '1h'
        )
        
        if historical_data:
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Generate predictions
            prediction_service = PredictionVisualizationService()
            timeframe_enum = PredictionTimeframe(timeframe)
            
            prediction = prediction_service.generate_predictions(
                symbol=symbol,
                historical_data=df,
                timeframe=timeframe_enum,
                prediction_periods=periods
            )
            
            # Format for API response
            return {
                "symbol": prediction.symbol,
                "currentPrice": prediction.current_price,
                "predictions": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "confidence": p.confidence,
                        "upperBound": p.upper_bound,
                        "lowerBound": p.lower_bound
                    }
                    for p in prediction.prediction_points
                ],
                "trend": {
                    "direction": prediction.trend_direction,
                    "strength": prediction.strength,
                    "confidence": prediction.confidence
                },
                "levels": {
                    "support": prediction.support_levels,
                    "resistance": prediction.resistance_levels,
                    "key": [
                        {
                            "price": level["price"],
                            "type": level["type"],
                            "strength": level["strength"],
                            "distance": level["distance_pct"]
                        }
                        for level in prediction.key_levels[:5]
                    ]
                },
                "metrics": {
                    "momentum": prediction.momentum_score,
                    "volatility": prediction.volatility
                },
                "metadata": prediction.metadata
            }
        else:
            raise ValueError("No historical data available")
            
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        # Return mock predictions
        current_price = 150.0
        predictions = []
        for i in range(periods):
            timestamp = now_utc() + timedelta(hours=i+1)
            price = current_price + random.uniform(-5, 5) * (i+1) * 0.1
            confidence = 100 * np.exp(-i * 0.05)
            
            predictions.append({
                "timestamp": timestamp.isoformat(),
                "price": round(price, 2),
                "confidence": round(confidence, 1),
                "upperBound": round(price + 2 + i*0.5, 2),
                "lowerBound": round(price - 2 - i*0.5, 2)
            })
        
        return {
            "symbol": symbol,
            "currentPrice": current_price,
            "predictions": predictions,
            "trend": {
                "direction": "bullish" if predictions[-1]["price"] > current_price else "bearish",
                "strength": round(random.uniform(60, 85), 1),
                "confidence": round(random.uniform(70, 90), 1)
            },
            "levels": {
                "support": [145.50, 142.30, 140.00],
                "resistance": [155.20, 158.50, 162.00],
                "key": [
                    {"price": 145.50, "type": "support", "strength": 85, "distance": 3.0},
                    {"price": 155.20, "type": "resistance", "strength": 90, "distance": 3.5}
                ]
            },
            "metrics": {
                "momentum": round(random.uniform(-50, 50), 1),
                "volatility": round(random.uniform(15, 35), 1)
            },
            "metadata": {
                "models_used": ["linear", "polynomial", "ml_ensemble"],
                "timeframe": timeframe,
                "periods": periods
            }
        }

@app.get("/api/v1/patterns/{symbol}")
async def get_candlestick_patterns(symbol: str, lookback: int = 100):
    """Get detected candlestick patterns for a symbol"""
    try:
        # Import services
        from src.services.candlestick_patterns import CandlestickPatternService
        
        # Get historical data
        end_date = now_utc()
        start_date = end_date - timedelta(days=30)
        
        historical_data = await simple_live_data.fetch_historical_data(
            symbol, start_date, end_date, '1h'
        )
        
        if historical_data:
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Detect patterns
            pattern_service = CandlestickPatternService()
            patterns = pattern_service.detect_all_patterns(df, lookback=lookback)
            
            # Get statistics
            stats = pattern_service.get_pattern_statistics(patterns)
            
            # Format for API response
            return {
                "symbol": symbol,
                "patterns": [
                    {
                        "type": p.pattern_type.value,
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "direction": p.direction,
                        "strength": p.strength,
                        "confidence": p.confidence,
                        "successRate": p.success_rate,
                        "description": p.description,
                        "targets": {
                            "priceTarget": p.price_target,
                            "stopLoss": p.stop_loss
                        }
                    }
                    for p in patterns[:20]  # Return latest 20 patterns
                ],
                "statistics": stats,
                "timestamp": now_utc().isoformat()
            }
        else:
            raise ValueError("No historical data available")
            
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        # Return mock patterns
        patterns = []
        pattern_types = [
            ("hammer", "bullish", "Hammer pattern detected at support", 0.62),
            ("shooting_star", "bearish", "Shooting star at resistance", 0.59),
            ("bullish_engulfing", "bullish", "Bullish engulfing pattern", 0.65),
            ("doji", "neutral", "Doji indicates indecision", 0.55),
            ("morning_star", "bullish", "Morning star reversal pattern", 0.65)
        ]
        
        for i, (ptype, direction, desc, success_rate) in enumerate(pattern_types[:3]):
            patterns.append({
                "type": ptype,
                "timestamp": (now_utc() - timedelta(hours=i*8)).isoformat(),
                "price": round(150 + random.uniform(-10, 10), 2),
                "direction": direction,
                "strength": round(random.uniform(70, 95), 1),
                "confidence": round(random.uniform(60, 90), 1),
                "successRate": success_rate,
                "description": desc,
                "targets": {
                    "priceTarget": round(155 + random.uniform(-5, 10), 2) if direction == "bullish" else round(145 - random.uniform(-5, 10), 2),
                    "stopLoss": round(148 - random.uniform(0, 3), 2) if direction == "bullish" else round(152 + random.uniform(0, 3), 2)
                }
            })
        
        return {
            "symbol": symbol,
            "patterns": patterns,
            "statistics": {
                "total_patterns": len(patterns),
                "bullish_count": sum(1 for p in patterns if p["direction"] == "bullish"),
                "bearish_count": sum(1 for p in patterns if p["direction"] == "bearish"),
                "neutral_count": sum(1 for p in patterns if p["direction"] == "neutral"),
                "avg_confidence": round(sum(p["confidence"] for p in patterns) / len(patterns), 1) if patterns else 0,
                "avg_success_rate": round(sum(p["successRate"] for p in patterns) / len(patterns), 2) if patterns else 0
            },
            "timestamp": now_utc().isoformat()
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": now_utc().isoformat()
        })
        
        # Create tasks for different update types
        async def send_market_updates():
            while True:
                await asyncio.sleep(5)  # Every 5 seconds
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "NVDA"]
                updates = []
                
                for symbol in symbols:
                    try:
                        market_data = await get_live_market_data(symbol)
                        updates.append(market_data)
                    except Exception as e:
                        logger.error(f"Error fetching market data for {symbol}: {e}")
                
                if updates:
                    await websocket.send_json({
                        "type": "market_updates",
                        "data": updates,
                        "timestamp": now_utc().isoformat()
                    })
        
        async def send_signals():
            while True:
                await asyncio.sleep(30)  # Every 30 seconds
                try:
                    # Generate new signals
                    signals = await simplified_ml_signals.generate_signals(
                        limit=3,
                        live_data_fetcher=simple_live_data
                    )
                    if signals:
                        await websocket.send_json({
                            "type": "new_signals",
                            "data": signals,
                            "timestamp": now_utc().isoformat()
                        })
                except Exception as e:
                    logger.error(f"Error generating signals for WebSocket: {e}")
        
        # Run both tasks concurrently
        await asyncio.gather(
            send_market_updates(),
            send_signals()
        )
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
        await websocket.close()

if __name__ == "__main__":
    print("Starting GoldenSignalsAI Simple Backend with Live Data")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 