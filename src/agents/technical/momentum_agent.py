"""
Momentum Trading Agent for GoldenSignalsAI
Analyzes price momentum and generates trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import talib

from src.agents.core.unified_base_agent import UnifiedBaseAgent, AgentType, MessagePriority, AgentMessage


class MomentumAgent(UnifiedBaseAgent):
    """Agent specialized in momentum-based trading strategies"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.TECHNICAL, config)
        
        # Momentum indicators configuration
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.momentum_threshold = config.get('momentum_threshold', 0.02)
        
        # Additional momentum indicators
        self.stoch_period = config.get('stoch_period', 14)
        self.stoch_smooth_k = config.get('stoch_smooth_k', 3)
        self.stoch_smooth_d = config.get('stoch_smooth_d', 3)
        self.adx_period = config.get('adx_period', 14)
        
        # Signal generation parameters
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.strong_trend_adx = config.get('strong_trend_adx', 25)
        self.volume_confirmation = config.get('volume_confirmation', True)
        
        # Risk management
        self.position_size = config.get('position_size', 0.1)
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 2.0)
        self.take_profit_ratio = config.get('take_profit_ratio', 2.0)
        
        # Cache for calculations
        self.indicator_cache: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[Dict[str, Any]] = []
        
    def _register_capabilities(self):
        """Register momentum agent capabilities"""
        self.capabilities = {
            'analyze_momentum': {
                'description': 'Analyze price momentum indicators',
                'input': {'symbol': 'str', 'data': 'pd.DataFrame'},
                'output': {'indicators': 'Dict[str, float]', 'strength': 'float'}
            },
            'generate_momentum_signals': {
                'description': 'Generate momentum-based trading signals',
                'input': {'symbol': 'str', 'data': 'pd.DataFrame'},
                'output': {'signals': 'List[Dict]'}
            },
            'screen_momentum': {
                'description': 'Screen for momentum opportunities',
                'input': {'symbols': 'List[str]', 'market_data': 'Dict[str, pd.DataFrame]'},
                'output': {'opportunities': 'List[Dict]'}
            },
            'detect_breakouts': {
                'description': 'Detect momentum breakouts',
                'input': {'symbol': 'str', 'data': 'pd.DataFrame'},
                'output': {'breakouts': 'List[Dict]'}
            }
        }
    
    def _register_message_handlers(self):
        """Register message handlers"""
        self.message_handlers = {
            'analyze_momentum': self.handle_analyze_momentum,
            'generate_signals': self.handle_generate_signals,
            'screen_momentum': self.handle_screen_momentum,
            'detect_breakouts': self.handle_detect_breakouts,
            'backtest_strategy': self.handle_backtest_strategy
        }
    
    async def handle_analyze_momentum(self, message: AgentMessage) -> Dict[str, Any]:
        """Analyze momentum indicators for a symbol"""
        symbol = message.payload.get('symbol')
        data = message.payload.get('data')
        
        if not isinstance(data, pd.DataFrame):
            return {'error': 'Invalid data format'}
        
        try:
            indicators = self._calculate_momentum_indicators(data)
            strength = self._calculate_momentum_strength(indicators)
            
            # Cache results
            self.indicator_cache[symbol] = {
                'indicators': indicators,
                'strength': strength,
                'timestamp': datetime.now()
            }
            
            return {
                'symbol': symbol,
                'indicators': indicators,
                'strength': strength,
                'interpretation': self._interpret_momentum(indicators, strength)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {'error': str(e)}
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all momentum indicators"""
        indicators = {}
        
        # Ensure we have required columns
        if not all(col in data.columns for col in ['close', 'high', 'low', 'volume']):
            raise ValueError("Missing required columns in data")
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # RSI
        indicators['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        indicators['macd'] = macd[-1]
        indicators['macd_signal'] = signal[-1]
        indicators['macd_histogram'] = hist[-1]
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_period,
            slowk_period=self.stoch_smooth_k,
            slowd_period=self.stoch_smooth_d
        )
        indicators['stoch_k'] = slowk[-1]
        indicators['stoch_d'] = slowd[-1]
        
        # ADX (Average Directional Index)
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)[-1]
        indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)[-1]
        indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)[-1]
        
        # Rate of Change
        indicators['roc'] = talib.ROC(close, timeperiod=10)[-1]
        
        # Momentum
        indicators['mom'] = talib.MOM(close, timeperiod=10)[-1]
        
        # Volume indicators
        indicators['obv'] = talib.OBV(close, volume)[-1]
        indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
        indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
        
        # Price action
        indicators['price_change'] = (close[-1] - close[-2]) / close[-2]
        indicators['high_low_ratio'] = (high[-1] - low[-1]) / close[-1]
        
        return indicators
    
    def _calculate_momentum_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate overall momentum strength score"""
        strength = 0.0
        weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'adx': 0.2,
            'stoch': 0.15,
            'volume': 0.1,
            'roc': 0.1
        }
        
        # RSI component
        if indicators['rsi'] > 50:
            strength += weights['rsi'] * (indicators['rsi'] - 50) / 50
        else:
            strength -= weights['rsi'] * (50 - indicators['rsi']) / 50
        
        # MACD component
        if indicators['macd'] > indicators['macd_signal']:
            macd_strength = min(abs(indicators['macd_histogram']) / 0.01, 1.0)
            strength += weights['macd'] * macd_strength
        else:
            macd_strength = min(abs(indicators['macd_histogram']) / 0.01, 1.0)
            strength -= weights['macd'] * macd_strength
        
        # ADX component (trend strength)
        if indicators['adx'] > self.strong_trend_adx:
            adx_strength = min((indicators['adx'] - self.strong_trend_adx) / 25, 1.0)
            if indicators['plus_di'] > indicators['minus_di']:
                strength += weights['adx'] * adx_strength
            else:
                strength -= weights['adx'] * adx_strength
        
        # Stochastic component
        stoch_avg = (indicators['stoch_k'] + indicators['stoch_d']) / 2
        if stoch_avg > 50:
            strength += weights['stoch'] * (stoch_avg - 50) / 50
        else:
            strength -= weights['stoch'] * (50 - stoch_avg) / 50
        
        # Volume component
        if indicators['volume_ratio'] > 1.0:
            volume_strength = min((indicators['volume_ratio'] - 1.0) / 1.0, 1.0)
            strength += weights['volume'] * volume_strength
        
        # ROC component
        if indicators['roc'] > 0:
            roc_strength = min(indicators['roc'] / 5.0, 1.0)
            strength += weights['roc'] * roc_strength
        else:
            roc_strength = min(abs(indicators['roc']) / 5.0, 1.0)
            strength -= weights['roc'] * roc_strength
        
        # Normalize to [-1, 1]
        return max(min(strength, 1.0), -1.0)
    
    def _interpret_momentum(self, indicators: Dict[str, float], strength: float) -> str:
        """Interpret momentum indicators"""
        if strength > 0.6:
            return "Strong bullish momentum"
        elif strength > 0.3:
            return "Moderate bullish momentum"
        elif strength > -0.3:
            return "Neutral momentum"
        elif strength > -0.6:
            return "Moderate bearish momentum"
        else:
            return "Strong bearish momentum"
    
    async def handle_generate_signals(self, message: AgentMessage) -> Dict[str, Any]:
        """Generate momentum-based trading signals"""
        symbol = message.payload.get('symbol')
        data = message.payload.get('data')
        
        if not isinstance(data, pd.DataFrame):
            return {'error': 'Invalid data format'}
        
        try:
            # Calculate indicators
            indicators = self._calculate_momentum_indicators(data)
            strength = self._calculate_momentum_strength(indicators)
            
            # Generate signals
            signals = self._generate_momentum_signals(symbol, data, indicators, strength)
            
            # Store in history
            self.signal_history.extend(signals)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            return {
                'symbol': symbol,
                'signals': signals,
                'momentum_strength': strength,
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {'error': str(e)}
    
    def _generate_momentum_signals(self, 
                                 symbol: str,
                                 data: pd.DataFrame,
                                 indicators: Dict[str, float],
                                 strength: float) -> List[Dict[str, Any]]:
        """Generate specific momentum signals"""
        signals = []
        current_price = data['close'].iloc[-1]
        
        # Strong momentum buy signal
        if (strength > 0.5 and
            indicators['rsi'] > 50 and indicators['rsi'] < self.rsi_overbought and
            indicators['macd'] > indicators['macd_signal'] and
            indicators['adx'] > self.strong_trend_adx and
            indicators['volume_ratio'] > 1.2):
            
            # Calculate stop loss and take profit
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, 14)[-1]
            stop_loss = current_price - (atr * self.stop_loss_atr_mult)
            take_profit = current_price + (atr * self.stop_loss_atr_mult * self.take_profit_ratio)
            
            signals.append({
                'type': 'momentum_breakout',
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': min(0.5 + strength * 0.5, 0.95),
                'strategy': 'momentum_trend_following',
                'indicators': {
                    'rsi': indicators['rsi'],
                    'macd': indicators['macd'],
                    'adx': indicators['adx'],
                    'volume_ratio': indicators['volume_ratio']
                },
                'timestamp': datetime.now()
            })
        
        # Momentum reversal signal (oversold bounce)
        elif (indicators['rsi'] < self.rsi_oversold and
              indicators['stoch_k'] < 20 and
              indicators['macd_histogram'] > indicators.get('prev_macd_histogram', indicators['macd_histogram'])):
            
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, 14)[-1]
            stop_loss = current_price - (atr * self.stop_loss_atr_mult)
            take_profit = current_price + (atr * self.stop_loss_atr_mult * self.take_profit_ratio)
            
            signals.append({
                'type': 'oversold_bounce',
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 0.6,
                'strategy': 'momentum_reversal',
                'indicators': {
                    'rsi': indicators['rsi'],
                    'stoch_k': indicators['stoch_k'],
                    'macd_histogram': indicators['macd_histogram']
                },
                'timestamp': datetime.now()
            })
        
        # Momentum exhaustion signal (overbought)
        elif (indicators['rsi'] > self.rsi_overbought and
              indicators['stoch_k'] > 80 and
              strength < 0.3):
            
            signals.append({
                'type': 'momentum_exhaustion',
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'confidence': 0.65,
                'strategy': 'momentum_fade',
                'indicators': {
                    'rsi': indicators['rsi'],
                    'stoch_k': indicators['stoch_k'],
                    'momentum_strength': strength
                },
                'timestamp': datetime.now()
            })
        
        # MACD crossover signals
        if indicators['macd'] > indicators['macd_signal'] and indicators.get('prev_macd', 0) <= indicators.get('prev_macd_signal', 0):
            signals.append({
                'type': 'macd_bullish_cross',
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'confidence': 0.55,
                'strategy': 'macd_crossover',
                'indicators': {
                    'macd': indicators['macd'],
                    'macd_signal': indicators['macd_signal']
                },
                'timestamp': datetime.now()
            })
        
        return signals
    
    async def handle_screen_momentum(self, message: AgentMessage) -> Dict[str, Any]:
        """Screen multiple symbols for momentum opportunities"""
        symbols = message.payload.get('symbols', [])
        market_data = message.payload.get('market_data', {})
        
        opportunities = []
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if not isinstance(data, pd.DataFrame) or len(data) < 50:
                continue
            
            try:
                indicators = self._calculate_momentum_indicators(data)
                strength = self._calculate_momentum_strength(indicators)
                
                # Check for strong momentum
                if abs(strength) > 0.5:
                    opportunities.append({
                        'symbol': symbol,
                        'momentum_strength': strength,
                        'direction': 'bullish' if strength > 0 else 'bearish',
                        'rsi': indicators['rsi'],
                        'adx': indicators['adx'],
                        'volume_ratio': indicators['volume_ratio'],
                        'score': abs(strength) * (1 + indicators['volume_ratio'] / 10)
                    })
            
            except Exception as e:
                self.logger.error(f"Error screening {symbol}: {e}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'opportunities': opportunities[:10],  # Top 10
            'total_screened': len(symbols),
            'timestamp': datetime.now()
        }
    
    async def handle_detect_breakouts(self, message: AgentMessage) -> Dict[str, Any]:
        """Detect momentum breakouts"""
        symbol = message.payload.get('symbol')
        data = message.payload.get('data')
        
        if not isinstance(data, pd.DataFrame):
            return {'error': 'Invalid data format'}
        
        try:
            breakouts = self._detect_breakouts(symbol, data)
            
            return {
                'symbol': symbol,
                'breakouts': breakouts,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting breakouts: {e}")
            return {'error': str(e)}
    
    def _detect_breakouts(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect various types of momentum breakouts"""
        breakouts = []
        
        # Price breakouts
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        current_price = data['close'].iloc[-1]
        
        # Bullish breakout
        if current_price > high_20.iloc[-2] and data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1] * 1.5:
            breakouts.append({
                'type': 'price_breakout',
                'direction': 'bullish',
                'level': high_20.iloc[-2],
                'current_price': current_price,
                'volume_surge': data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1],
                'strength': (current_price - high_20.iloc[-2]) / high_20.iloc[-2]
            })
        
        # Volume breakouts
        volume_std = data['volume'].rolling(20).std()
        volume_mean = data['volume'].rolling(20).mean()
        volume_zscore = (data['volume'] - volume_mean) / volume_std
        
        if volume_zscore.iloc[-1] > 2.0:
            breakouts.append({
                'type': 'volume_breakout',
                'volume_zscore': volume_zscore.iloc[-1],
                'volume_ratio': data['volume'].iloc[-1] / volume_mean.iloc[-1],
                'price_change': data['close'].pct_change().iloc[-1]
            })
        
        # Volatility breakouts
        atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, 14)
        atr_mean = pd.Series(atr).rolling(20).mean()
        
        if atr[-1] > atr_mean.iloc[-1] * 1.5:
            breakouts.append({
                'type': 'volatility_breakout',
                'atr_ratio': atr[-1] / atr_mean.iloc[-1],
                'direction': 'expanding'
            })
        
        return breakouts
    
    async def handle_backtest_strategy(self, message: AgentMessage) -> Dict[str, Any]:
        """Backtest momentum strategy"""
        symbol = message.payload.get('symbol')
        data = message.payload.get('data')
        initial_capital = message.payload.get('initial_capital', 10000)
        
        if not isinstance(data, pd.DataFrame):
            return {'error': 'Invalid data format'}
        
        try:
            results = self._backtest_momentum_strategy(symbol, data, initial_capital)
            return results
            
        except Exception as e:
            self.logger.error(f"Error backtesting strategy: {e}")
            return {'error': str(e)}
    
    def _backtest_momentum_strategy(self, 
                                  symbol: str,
                                  data: pd.DataFrame,
                                  initial_capital: float) -> Dict[str, Any]:
        """Simple backtest of momentum strategy"""
        # This is a simplified backtest - in production, use the full backtesting engine
        positions = []
        trades = []
        capital = initial_capital
        position = None
        
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1]
            
            # Calculate indicators
            indicators = self._calculate_momentum_indicators(current_data)
            strength = self._calculate_momentum_strength(indicators)
            
            current_price = current_data['close'].iloc[-1]
            
            # Check for entry
            if position is None and strength > 0.5:
                # Buy signal
                shares = int(capital * self.position_size / current_price)
                if shares > 0:
                    position = {
                        'entry_price': current_price,
                        'shares': shares,
                        'entry_date': current_data.index[-1]
                    }
                    capital -= shares * current_price
            
            # Check for exit
            elif position is not None and (strength < 0 or 
                                         current_price < position['entry_price'] * 0.98 or
                                         current_price > position['entry_price'] * 1.05):
                # Sell signal
                capital += position['shares'] * current_price
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current_data.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'shares': position['shares'],
                    'pnl': position['shares'] * (current_price - position['entry_price']),
                    'return': (current_price - position['entry_price']) / position['entry_price']
                })
                
                position = None
        
        # Calculate metrics
        if trades:
            total_return = (capital - initial_capital) / initial_capital
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
            avg_win = np.mean([t['return'] for t in trades if t['return'] > 0]) if any(t['return'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['return'] for t in trades if t['return'] < 0]) if any(t['return'] < 0 for t in trades) else 0
            
            return {
                'symbol': symbol,
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'final_capital': capital,
                'trades': trades[-10:]  # Last 10 trades
            }
        else:
            return {
                'symbol': symbol,
                'total_return': 0,
                'win_rate': 0,
                'total_trades': 0,
                'message': 'No trades generated'
            }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request"""
        request_type = request.get('type', 'unknown')
        
        if request_type == 'analyze':
            return await self.handle_analyze_momentum(AgentMessage(
                sender_id='api',
                recipient_id=self.agent_id,
                message_type='analyze_momentum',
                payload=request
            ))
        elif request_type == 'screen':
            return await self.handle_screen_momentum(AgentMessage(
                sender_id='api',
                recipient_id=self.agent_id,
                message_type='screen_momentum',
                payload=request
            ))
        else:
            return await super().process_request(request) 