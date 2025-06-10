#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI V3 - Live Market Data Service
Real-time market data streaming and processing for institutional trading

Features:
- Real-time price feeds via Yahoo Finance
- WebSocket streaming for frontend
- Technical indicator calculation
- Signal generation pipeline
- Risk assessment integration
- Performance monitoring
"""

import asyncio
import json
import websockets
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    timestamp: str
    change: float
    change_percent: float
    
@dataclass
class SignalData:
    """Trading signal data"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price_target: float
    stop_loss: float
    risk_score: float
    indicators: Dict[str, float]
    timestamp: str

class TechnicalIndicators:
    """Real-time technical indicator calculations"""
    
    @staticmethod
    def calculate_indicators(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators with NaN handling"""
        if len(data) < 10:  # Reduced from 50 to 10
            return {}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        def safe_float(value, default=0.0):
            """Convert to float, handling NaN values"""
            try:
                if pd.isna(value) or np.isnan(value):
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default
        
        # Moving averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        macd_line = ema_12 - ema_26
        signal_line = pd.Series(macd_line).ewm(span=9).mean().iloc[-1]
        macd_histogram = macd_line - signal_line
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
        
        # Volatility
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        
        # Volume indicators
        volume_sma = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1
        
        return {
            'sma_20': safe_float(sma_20),
            'sma_50': safe_float(sma_50),
            'ema_12': safe_float(ema_12),
            'ema_26': safe_float(ema_26),
            'rsi': safe_float(rsi),
            'macd': safe_float(macd_line),
            'macd_signal': safe_float(signal_line),
            'macd_histogram': safe_float(macd_histogram),
            'bb_upper': safe_float(bb_upper),
            'bb_middle': safe_float(bb_middle),
            'bb_lower': safe_float(bb_lower),
            'bb_width': safe_float(bb_width),
            'volatility': safe_float(volatility),
            'volume_ratio': safe_float(volume_ratio),
            'current_price': safe_float(close.iloc[-1])
        }

class MLModelLoader:
    """Load and use trained ML models"""
    
    def __init__(self, model_dir: str = None):
        # Auto-detect correct path based on current working directory
        if model_dir is None:
            import os
            if os.path.exists("../../ml_training/models"):
                model_dir = "../../ml_training/models"
            elif os.path.exists("../ml_training/models"):
                model_dir = "../ml_training/models"
            elif os.path.exists("ml_training/models"):
                model_dir = "ml_training/models"
            else:
                model_dir = "../../ml_training/models"  # fallback
        
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load forecast model
            with open(f"{self.model_dir}/forecast_model.pkl", 'rb') as f:
                self.models['forecast'] = pickle.load(f)
            with open(f"{self.model_dir}/forecast_scaler.pkl", 'rb') as f:
                self.scalers['forecast'] = pickle.load(f)
            
            # Load signal classifier
            with open(f"{self.model_dir}/signal_classifier.pkl", 'rb') as f:
                self.models['signal'] = pickle.load(f)
            with open(f"{self.model_dir}/signal_classifier_scaler.pkl", 'rb') as f:
                self.scalers['signal'] = pickle.load(f)
            
            # Load risk model
            with open(f"{self.model_dir}/risk_model.pkl", 'rb') as f:
                self.models['risk'] = pickle.load(f)
            with open(f"{self.model_dir}/risk_scaler.pkl", 'rb') as f:
                self.scalers['risk'] = pickle.load(f)
            
            logger.info("✅ ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.models = {}
            self.scalers = {}
    
    def predict_price_movement(self, features: np.ndarray) -> float:
        """Predict future price movement"""
        if 'forecast' not in self.models:
            return 0.0
        
        try:
            scaled_features = self.scalers['forecast'].transform(features.reshape(1, -1))
            prediction = self.models['forecast'].predict(scaled_features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0
    
    def classify_signal(self, features: np.ndarray) -> Dict[str, float]:
        """Classify trading signal"""
        if 'signal' not in self.models:
            return {'neutral': 1.0, 'bull': 0.0, 'bear': 0.0}
        
        try:
            scaled_features = self.scalers['signal'].transform(features.reshape(1, -1))
            proba = self.models['signal'].predict_proba(scaled_features)[0]
            
            # Map probabilities to signal types
            return {
                'neutral': float(proba[0]),
                'bull': float(proba[1]) if len(proba) > 1 else 0.0,
                'bear': float(proba[2]) if len(proba) > 2 else 0.0
            }
        except Exception as e:
            logger.error(f"Signal classification error: {e}")
            return {'neutral': 1.0, 'bull': 0.0, 'bear': 0.0}
    
    def assess_risk(self, features: np.ndarray) -> float:
        """Assess risk score"""
        if 'risk' not in self.models:
            return 0.5
        
        try:
            scaled_features = self.scalers['risk'].transform(features.reshape(1, -1))
            risk_score = self.models['risk'].predict(scaled_features)[0]
            return float(np.clip(risk_score, 0, 1))
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return 0.5

class MarketDataService:
    """Main market data service"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'IWM']
        self.data_cache = {}
        self.indicators = TechnicalIndicators()
        self.ml_models = MLModelLoader()
        self.connected_clients = set()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Feature columns (must match training data)
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'Price_Change', 'High_Low_Pct', 'Volume_Ratio',
            'Volatility', 'ATR'
        ]
    
    def fetch_real_time_data(self, symbol: str) -> Optional[MarketTick]:
        """Fetch real-time market data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
            
            current = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
            
            change = current['Close'] - previous['Close']
            change_percent = (change / previous['Close']) * 100
            
            # Get bid/ask from info (may not always be available)
            bid = info.get('bid', current['Close'] * 0.999)
            ask = info.get('ask', current['Close'] * 1.001)
            
            return MarketTick(
                symbol=symbol,
                price=float(current['Close']),
                volume=int(current['Volume']),
                bid=float(bid),
                ask=float(ask),
                spread=float(ask - bid),
                timestamp=datetime.now().isoformat(),
                change=float(change),
                change_percent=float(change_percent)
            )
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Get historical data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate all technical indicators
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
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signal(self, symbol: str) -> Optional[SignalData]:
        """Generate trading signal using ML models"""
        try:
            # Get historical data
            hist_data = self.get_historical_data(symbol)
            if hist_data.empty or len(hist_data) < 10:  # Reduced from 50 to 10
                return None
            
            # Extract features from latest data
            latest = hist_data.iloc[-1]
            features = []
            
            for col in self.feature_columns:
                if col in hist_data.columns:
                    features.append(latest[col])
                else:
                    features.append(0.0)  # Default value for missing features
            
            features = np.array(features)
            
            # Get ML predictions
            price_prediction = self.ml_models.predict_price_movement(features)
            signal_proba = self.ml_models.classify_signal(features)
            risk_score = self.ml_models.assess_risk(features)
            
            # Determine signal type
            current_price = latest['Close']
            
            if signal_proba['bull'] > 0.6:
                signal_type = 'BUY'
                confidence = signal_proba['bull']
                price_target = current_price * (1 + abs(price_prediction))
                stop_loss = current_price * 0.95
            elif signal_proba['bear'] > 0.6:
                signal_type = 'SELL'
                confidence = signal_proba['bear']
                price_target = current_price * (1 - abs(price_prediction))
                stop_loss = current_price * 1.05
            else:
                signal_type = 'HOLD'
                confidence = signal_proba['neutral']
                price_target = current_price
                stop_loss = current_price
            
            # Calculate technical indicators for display
            indicators = self.indicators.calculate_indicators(hist_data)
            
            return SignalData(
                symbol=symbol,
                signal_type=signal_type,
                confidence=float(confidence),
                price_target=float(price_target),
                stop_loss=float(stop_loss),
                risk_score=float(risk_score),
                indicators=indicators,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        logger.info(f"🔌 New WebSocket connection: {websocket.remote_address}")
        self.connected_clients.add(websocket)
        
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            logger.info(f"🔌 WebSocket disconnected: {websocket.remote_address}")
    
    async def broadcast_data(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.connected_clients.discard(client)
    
    def data_collection_loop(self):
        """Main data collection loop"""
        logger.info("🔄 Starting market data collection...")
        
        while self.running:
            try:
                # Collect market ticks
                ticks = []
                signals = []
                
                for symbol in self.symbols:
                    # Get real-time tick
                    tick = self.fetch_real_time_data(symbol)
                    if tick:
                        ticks.append(asdict(tick))
                    
                    # Generate signal (less frequently)
                    if len(self.data_cache.get(symbol, [])) % 10 == 0:  # Every 10th iteration
                        signal = self.generate_signal(symbol)
                        if signal:
                            signals.append(asdict(signal))
                
                # Broadcast data
                if ticks or signals:
                    asyncio.run(self.broadcast_data({
                        'type': 'market_update',
                        'ticks': ticks,
                        'signals': signals,
                        'timestamp': datetime.now().isoformat()
                    }))
                
                # Wait before next update
                time.sleep(10)  # 10-second intervals
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(5)
    
    def start_service(self, host: str = "localhost", port: int = 8765):
        """Start the market data service"""
        logger.info("🚀 Starting Market Data Service...")
        self.running = True
        
        # Start data collection in background thread
        data_thread = threading.Thread(target=self.data_collection_loop, daemon=True)
        data_thread.start()
        
        # Start WebSocket server
        logger.info(f"🌐 WebSocket server starting on ws://{host}:{port}")
        start_server = websockets.serve(self.websocket_handler, host, port)
        
        try:
            asyncio.get_event_loop().run_until_complete(start_server)
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down market data service...")
            self.running = False
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get current market summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'market_status': 'OPEN',  # Simplified
            'total_symbols': len(self.symbols)
        }
        
        for symbol in self.symbols:
            tick = self.fetch_real_time_data(symbol)
            if tick:
                summary['symbols'][symbol] = {
                    'price': tick.price,
                    'change': tick.change,
                    'change_percent': tick.change_percent,
                    'volume': tick.volume
                }
        
        return summary

def main():
    """Main function for testing"""
    print("🔥 GoldenSignalsAI V3 - Market Data Service")
    print("=" * 50)
    
    service = MarketDataService()
    
    # Test market summary
    summary = service.get_market_summary()
    print(f"📊 Market Summary: {len(summary['symbols'])} symbols")
    
    # Start service
    try:
        service.start_service()
    except KeyboardInterrupt:
        print("\n👋 Service stopped")

if __name__ == "__main__":
    main() 