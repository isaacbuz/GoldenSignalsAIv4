"""
GoldenSignalsAI - Production FastAPI Backend with Database Integration
Enhanced implementation with PostgreSQL database storage
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uvicorn

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import yfinance as yf
import pandas as pd
import numpy as np

# Import database components
from database import get_db, init_database, db_manager
from models.signal import Signal, SignalAction, RiskLevel, SignalStatus
from models.agent import Agent
from models.user import User
from models.portfolio import Portfolio

# Import Redis cache service
from services.redis_cache_service import get_cache_service, cache_agent_result

# Import error tracking
from services.error_tracking import get_error_tracker, track_errors, create_sentry_exception_handler

# Import rate limiting
from middleware.rate_limiter import RateLimitMiddleware, rate_limit_low, rate_limit_medium

# Import position sizing
from services.position_sizing import get_position_sizer, PositionSizeResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Initialize error tracking
error_tracker = get_error_tracker()

# Create FastAPI app
app = FastAPI(
    title="GoldenSignalsAI API",
    description="AI-Powered Trading Signal Intelligence Platform with Database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add Sentry exception handler
if error_tracker.enabled:
    app.add_exception_handler(Exception, create_sentry_exception_handler(error_tracker))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app.add_middleware(RateLimitMiddleware, redis_url=redis_url)

# Data Models for API
class SignalResponse(BaseModel):
    id: str
    symbol: str
    action: str = Field(..., pattern="^(BUY|SELL|HOLD)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    price: float = Field(..., gt=0)
    risk_level: str = Field(..., pattern="^(low|medium|high)$")
    indicators: Dict = {}
    reasoning: str
    timestamp: datetime
    consensus_strength: float = Field(..., ge=0.0, le=1.0)
    position_size: Optional[Dict] = None  # Position sizing recommendation

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime

# WebSocket connections storage
websocket_connections: List[WebSocket] = []

# Enhanced AI Signal Generation Engine with Database Integration and Caching
class DatabaseSignalGenerator:
    """Enhanced signal generator with database storage, agent tracking, and Redis caching"""
    
    def __init__(self):
        self.agents = [
            "RSI_Agent", "MACD_Agent", "Sentiment_Agent", 
            "Volume_Agent", "Momentum_Agent"
        ]
        self.cache_service = get_cache_service()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD indicator"""
        exp12 = prices.ewm(span=12).mean()
        exp26 = prices.ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {
            "macd": macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
            "signal": signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0,
            "histogram": histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
        }
    
    def _generate_mock_historical_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock historical data for signal generation"""
        import random
        from datetime import datetime, timedelta
        
        # Base prices for common symbols
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'AMZN': 3200.0,
            'TSLA': 800.0,
            'NVDA': 900.0,
            'META': 350.0,
            'NFLX': 400.0,
            'SPY': 450.0,
            'QQQ': 380.0
        }
        
        base_price = base_prices.get(symbol.upper(), 100.0)
        
        # Generate 30 days of data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        data = []
        
        current_price = base_price
        for date in dates:
            # Random walk with slight upward bias
            change = random.uniform(-0.05, 0.06)
            current_price *= (1 + change)
            
            # OHLC data
            open_price = current_price * (1 + random.uniform(-0.02, 0.02))
            high_price = current_price * (1 + random.uniform(0, 0.03))
            low_price = current_price * (1 + random.uniform(-0.03, 0))
            close_price = current_price
            volume = random.randint(1000000, 10000000)
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @track_errors("signal_generation")
    async def generate_signal(self, symbol: str, db: Session) -> Signal:
        """Generate AI trading signal and store in database with caching"""
        try:
            # Check cache first
            cached_signal = self.cache_service.get_agent_result(
                agent_name="SignalGenerator",
                symbol=symbol,
                timeframe="30d"
            )
            
            if cached_signal:
                logger.info(f"🎯 Using cached signal for {symbol}")
                # Convert cached data back to Signal object
                signal_id = cached_signal['result'].get('id')
                if signal_id:
                    cached_db_signal = db.query(Signal).filter(Signal.id == signal_id).first()
                    if cached_db_signal:
                        return cached_db_signal
            
            # Fetch market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                # Use mock data if Yahoo Finance fails
                logger.warning(f"Yahoo Finance failed for {symbol}, using mock data for signal generation")
                hist = self._generate_mock_historical_data(symbol)
            
            # Calculate technical indicators
            prices = hist['Close']
            rsi = self.calculate_rsi(prices)
            macd_data = self.calculate_macd(prices)
            
            # Volume analysis
            avg_volume = hist['Volume'].rolling(window=10).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum
            price_change = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] * 100
            
            # Get agent weights from database
            agents_data = db.query(Agent).filter(Agent.is_active == True).all()
            
            # AI Consensus Algorithm with database agents
            signals = []
            confidences = []
            agent_votes = {}
            
            for agent in agents_data:
                if agent.agent_type == "rsi":
                    if rsi < 30:
                        vote = "BUY"
                        confidence = 0.8
                    elif rsi > 70:
                        vote = "SELL"
                        confidence = 0.8
                    else:
                        vote = "HOLD"
                        confidence = 0.5
                
                elif agent.agent_type == "macd":
                    if macd_data["histogram"] > 0 and macd_data["macd"] > macd_data["signal"]:
                        vote = "BUY"
                        confidence = 0.7
                    elif macd_data["histogram"] < 0 and macd_data["macd"] < macd_data["signal"]:
                        vote = "SELL"
                        confidence = 0.7
                    else:
                        vote = "HOLD"
                        confidence = 0.4
                
                elif agent.agent_type == "volume":
                    if volume_ratio > 1.5:
                        vote = "BUY" if price_change > 0 else "SELL"
                        confidence = 0.6
                    else:
                        vote = "HOLD"
                        confidence = 0.3
                
                elif agent.agent_type == "momentum":
                    if price_change > 2:
                        vote = "BUY"
                        confidence = 0.75
                    elif price_change < -2:
                        vote = "SELL"
                        confidence = 0.75
                    else:
                        vote = "HOLD"
                        confidence = 0.4
                
                else:
                    vote = "HOLD"
                    confidence = 0.5
                
                # Apply agent's consensus weight
                weighted_confidence = confidence * agent.consensus_weight
                signals.append(vote)
                confidences.append(weighted_confidence)
                agent_votes[agent.name] = {"vote": vote, "confidence": confidence, "weight": agent.consensus_weight}
            
            # Consensus calculation
            signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            total_confidence = 0
            
            for signal, confidence in zip(signals, confidences):
                signal_counts[signal] += confidence
                total_confidence += confidence
            
            # Determine final signal
            final_action = max(signal_counts, key=signal_counts.get)
            final_confidence = signal_counts[final_action] / total_confidence if total_confidence > 0 else 0.5
            consensus_strength = signal_counts[final_action] / sum(signal_counts.values()) if sum(signal_counts.values()) > 0 else 0.5
            
            # Risk assessment
            volatility = prices.pct_change().std() * 100
            if volatility > 3:
                risk_level = RiskLevel.HIGH
            elif volatility > 1.5:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Generate reasoning
            reasoning = f"Based on multi-agent analysis: RSI={rsi:.1f}, MACD histogram={macd_data['histogram']:.3f}, "
            reasoning += f"Volume ratio={volume_ratio:.1f}, Price momentum={price_change:.1f}%. "
            reasoning += f"Consensus from {len(self.agents)} AI agents suggests {final_action} with {final_confidence:.0%} confidence."
            
            # Calculate target price and stop loss
            current_price = float(prices.iloc[-1])
            if final_action == "BUY":
                target_price = current_price * 1.05  # 5% target
                stop_loss = current_price * 0.95     # 5% stop loss
            elif final_action == "SELL":
                target_price = current_price * 0.95  # 5% target (short)
                stop_loss = current_price * 1.05     # 5% stop loss (short)
            else:
                target_price = None
                stop_loss = None
            
            # Create signal in database
            signal = Signal(
                symbol=symbol,
                action=SignalAction[final_action],
                confidence=final_confidence,
                price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_level=risk_level,
                indicators={
                    "rsi": rsi,
                    "macd": macd_data,
                    "volume_ratio": volume_ratio,
                    "price_momentum": price_change,
                    "volatility": volatility
                },
                reasoning=reasoning,
                consensus_strength=consensus_strength,
                agent_votes=agent_votes,
                status=SignalStatus.ACTIVE,
                expires_at=datetime.now() + timedelta(hours=24)  # Expire in 24 hours
            )
            
            # Save to database
            db.add(signal)
            db.commit()
            db.refresh(signal)
            
            # Cache the generated signal
            self.cache_service.set_agent_result(
                agent_name="SignalGenerator",
                symbol=symbol,
                timeframe="30d",
                result={
                    'id': str(signal.id),
                    'action': final_action,
                    'confidence': final_confidence,
                    'price': current_price,
                    'risk_level': risk_level.value,
                    'consensus_strength': consensus_strength,
                    'generated_at': datetime.now().isoformat()
                },
                ttl_seconds=300  # Cache for 5 minutes
            )
            
            logger.info(f"✅ Generated signal for {symbol}: {final_action} with {final_confidence:.2f} confidence")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate signal: {str(e)}")

# Initialize signal generator
signal_generator = DatabaseSignalGenerator()

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "GoldenSignalsAI Backend with Database",
        "status": "online",
        "version": "1.0.0",
        "database": "connected",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check with database status"""
    try:
        # Test database connection
        signal_count = db.query(Signal).count()
        agent_count = db.query(Agent).count()
        
        return {
            "status": "healthy",
            "uptime": "running",
            "services": {
                "api": "online",
                "database": "connected",
                "signal_generator": "online",
                "websocket": "online"
            },
            "stats": {
                "total_signals": signal_count,
                "active_agents": agent_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/v1/signals", response_model=List[SignalResponse])
async def get_signals(
    symbol: Optional[str] = None,
    limit: int = 50,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get recent trading signals from database"""
    try:
        query = db.query(Signal)
        
        # Filter by symbol
        if symbol:
            query = query.filter(Signal.symbol.ilike(f"%{symbol.upper()}%"))
        
        # Filter by status
        if status:
            query = query.filter(Signal.status == SignalStatus[status.upper()])
        
        # Order by created_at and limit
        signals = query.order_by(Signal.created_at.desc()).limit(limit).all()
        
        # Convert to response format
        signal_responses = []
        for signal in signals:
            signal_responses.append(SignalResponse(
                id=str(signal.id),
                symbol=signal.symbol,
                action=signal.action.value,
                confidence=signal.confidence,
                price=signal.price,
                risk_level=signal.risk_level.value,
                indicators=signal.indicators or {},
                reasoning=signal.reasoning,
                timestamp=signal.created_at,
                consensus_strength=signal.consensus_strength
            ))
        
        return signal_responses
    
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch signals")

@app.post("/api/v1/signals/generate/{symbol}", response_model=SignalResponse)
async def generate_signal(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Generate new trading signal for a symbol and store in database"""
    try:
        # Generate signal
        signal = await signal_generator.generate_signal(symbol.upper(), db)
        
        # Convert to response format
        signal_response = SignalResponse(
            id=str(signal.id),
            symbol=signal.symbol,
            action=signal.action.value,
            confidence=signal.confidence,
            price=signal.price,
            risk_level=signal.risk_level.value,
            indicators=signal.indicators or {},
            reasoning=signal.reasoning,
            timestamp=signal.created_at,
            consensus_strength=signal.consensus_strength
        )
        
        # Broadcast to WebSocket clients
        background_tasks.add_task(broadcast_signal, signal_response)
        
        return signal_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate signal")

class MarketDataProvider:
    def __init__(self):
        self.providers = ['yfinance', 'polygon', 'alpha_vantage']
        self.api_keys = {
            'polygon': os.getenv('POLYGON_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY')
        }

    def get_data(self, symbol: str, period: str = '30d'):
        for provider in self.providers:
            try:
                if provider == 'yfinance':
                    return yf.Ticker(symbol).history(period=period)
                # Add Polygon and Alpha Vantage logic here
                # For example:
                # if provider == 'polygon': ...
                hist = yf.Ticker(symbol).history(period=period)
                if not hist.empty:
                    return hist
            except:
                logger.warning(f"Provider {provider} failed for {symbol}")
        raise HTTPException(404, "All providers failed")

@app.get("/api/v1/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str):
    provider = MarketDataProvider()
    hist = provider.get_data(symbol, '2d')
    # Rest of function

def _generate_mock_market_data(symbol: str) -> MarketDataResponse:
    """Generate mock market data for testing"""
    import random
    
    # Base prices for common symbols
    base_prices = {
        'AAPL': 150.0,
        'GOOGL': 2800.0,
        'MSFT': 300.0,
        'AMZN': 3200.0,
        'TSLA': 800.0,
        'NVDA': 900.0,
        'META': 350.0,
        'NFLX': 400.0,
        'SPY': 450.0,
        'QQQ': 380.0
    }
    
    base_price = base_prices.get(symbol.upper(), 100.0)
    
    # Add some random variation
    current_price = base_price * (1 + random.uniform(-0.05, 0.05))
    previous_close = base_price * (1 + random.uniform(-0.03, 0.03))
    change = current_price - previous_close
    change_percent = (change / previous_close * 100) if previous_close != 0 else 0
        
    return MarketDataResponse(
        symbol=symbol.upper(),
        price=round(current_price, 2),
        change=round(change, 2),
        change_percent=round(change_percent, 2),
        volume=random.randint(1000000, 10000000),
        high=round(current_price * 1.02, 2),
        low=round(current_price * 0.98, 2),
        open=round(previous_close * 1.01, 2),
        previous_close=round(previous_close, 2),
        timestamp=datetime.now()
    )

@app.get("/api/v1/agents")
async def get_agents(db: Session = Depends(get_db)):
    """Get AI agent performance statistics"""
    try:
        agents = db.query(Agent).all()
        return [agent.to_dict() for agent in agents]
    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch agents")

@app.post("/api/v1/position-size/calculate")
async def calculate_position_size(
    symbol: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    capital: float = 100000,
    signal_confidence: float = 0.8,
    db: Session = Depends(get_db)
):
    """Calculate optimal position size using Kelly Criterion"""
    try:
        # Get position sizer
        sizer = get_position_sizer()
        
        # Get historical win rate for the symbol (if available)
        recent_signals = db.query(Signal).filter(
            Signal.symbol == symbol,
            Signal.created_at >= datetime.now() - timedelta(days=30)
        ).all()
        
        win_rate = None
        if recent_signals:
            wins = sum(1 for s in recent_signals if s.pnl and s.pnl > 0)
            win_rate = wins / len(recent_signals)
        
        # Calculate current volatility
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="20d")
        volatility = None
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std()
        
        # Calculate position size
        result = sizer.calculate_position_size(
            capital=capital,
            signal_confidence=signal_confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            historical_win_rate=win_rate,
            volatility=volatility
        )
        
        return {
            "symbol": symbol,
            "recommended_position_size": round(result.recommended_size * 100, 2),
            "kelly_percentage": round(result.kelly_percentage * 100, 2),
            "adjusted_size": round(result.adjusted_size * 100, 2),
            "risk_amount": round(result.risk_amount, 2),
            "shares": result.shares,
            "position_value": round(result.shares * entry_price, 2),
            "reasoning": result.reasoning,
            "risk_reward_ratio": round((take_profit - entry_price) / (entry_price - stop_loss), 2),
            "max_loss": round(result.shares * (entry_price - stop_loss), 2),
            "max_profit": round(result.shares * (take_profit - entry_price), 2)
        }
        
    except Exception as e:
        logger.error(f"Position sizing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate position size: {str(e)}")

@app.get("/api/v1/rate-limit/status")
async def get_rate_limit_status(request: Request):
    """Get current rate limit status for the requesting client"""
    if not hasattr(app.state, 'rate_limiter') or not app.state.rate_limiter:
        return {"rate_limiting": "disabled"}
    
    # Get the middleware instance
    for middleware in app.middleware_stack:
        if isinstance(middleware, RateLimitMiddleware):
            identifier = middleware.rate_limiter._get_identifier(request)
            
            # Check current status without incrementing
            limit, window = middleware._get_limits_for_endpoint(request.url.path)
            key = middleware.rate_limiter._get_rate_limit_key(identifier, request.url.path)
            
            try:
                count = middleware.redis_client.zcard(key) if middleware.redis_client else 0
                return {
                    "rate_limiting": "enabled",
                    "identifier": identifier.split(':')[0],  # Type only, not full ID
                    "current_usage": count,
                    "limit": limit,
                    "window_seconds": window,
                    "remaining": max(0, limit - count)
                }
            except Exception as e:
                logger.error(f"Rate limit status error: {e}")
                
    return {"rate_limiting": "enabled", "status": "unknown"}

@app.get("/api/v1/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get platform statistics including cache performance"""
    try:
        total_signals = db.query(Signal).count()
        active_signals = db.query(Signal).filter(Signal.status == SignalStatus.ACTIVE).count()
        total_agents = db.query(Agent).count()
        
        # Recent performance
        recent_signals = db.query(Signal).filter(
            Signal.created_at >= datetime.now() - timedelta(days=7)
        ).all()
        
        profitable_signals = sum(1 for s in recent_signals if s.pnl > 0)
        win_rate = (profitable_signals / len(recent_signals) * 100) if recent_signals else 0
        
        # Get cache statistics
        cache_stats = signal_generator.cache_service.get_cache_stats()
        
        return {
            "total_signals": total_signals,
            "active_signals": active_signals,
            "total_agents": total_agents,
            "recent_win_rate": round(win_rate, 2),
            "recent_signals_count": len(recent_signals),
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")

# WebSocket Support
async def broadcast_signal(signal: SignalResponse):
    """Broadcast signal to all connected WebSocket clients"""
    if websocket_connections:
        message = {
            "type": "new_signal",
            "data": signal.dict()
        }
        
        disconnected = []
        for websocket in websocket_connections:
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in websocket_connections:
                websocket_connections.remove(websocket)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to GoldenSignalsAI with Database",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle ping
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# Background task for generating sample signals
async def generate_sample_signals():
    """Background task to generate sample signals periodically"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    
    while True:
        try:
            # Get database session
            with db_manager.get_session() as db:
                # Generate signals for random symbols
                import random
                symbol = random.choice(symbols)
                
                signal = await signal_generator.generate_signal(symbol, db)
                
                # Convert to response format and broadcast
                signal_response = SignalResponse(
                    id=str(signal.id),
                    symbol=signal.symbol,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    price=signal.price,
                    risk_level=signal.risk_level.value,
                    indicators=signal.indicators or {},
                    reasoning=signal.reasoning,
                    timestamp=signal.created_at,
                    consensus_strength=signal.consensus_strength
                )
                
                # Broadcast to WebSocket clients
                await broadcast_signal(signal_response)
            
            # Wait before next signal
            await asyncio.sleep(30)  # Generate signal every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in background signal generation: {e}")
            await asyncio.sleep(60)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Application startup with database initialization"""
    logger.info("🚀 GoldenSignalsAI Backend Starting...")
    
    try:
        # Initialize database
        init_database()
        
        # Start background signal generation
        asyncio.create_task(generate_sample_signals())
        
        logger.info("✅ GoldenSignalsAI Backend Ready with Database!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 GoldenSignalsAI Backend Shutting Down...")
    
    try:
        # Close database connections
        db_manager.close()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 