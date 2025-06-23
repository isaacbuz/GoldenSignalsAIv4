"""
Tests for domain knowledge and risk management in GoldenSignalsAI V2.
Based on best practices for incorporating trading expertise and risk controls.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestDomainRiskManagement:
    """Test domain knowledge integration and risk management"""
    
    @pytest.fixture
    def market_conditions(self):
        """Generate different market condition scenarios"""
        conditions = {
            'bull_market': {
                'trend': 'up',
                'volatility': 0.15,
                'volume': 1.2,  # Above average
                'sentiment': 0.7
            },
            'bear_market': {
                'trend': 'down',
                'volatility': 0.25,
                'volume': 1.5,  # High volume on declines
                'sentiment': 0.3
            },
            'sideways_market': {
                'trend': 'neutral',
                'volatility': 0.10,
                'volume': 0.8,  # Below average
                'sentiment': 0.5
            },
            'high_volatility': {
                'trend': 'neutral',
                'volatility': 0.40,
                'volume': 2.0,  # Very high volume
                'sentiment': 0.4
            }
        }
        return conditions
    
    def test_technical_analysis_integration(self):
        """Test integration of technical analysis rules with AI signals"""
        class TechnicalAnalysisValidator:
            def __init__(self):
                self.rules = {
                    'support_resistance': self.check_support_resistance,
                    'trend_alignment': self.check_trend_alignment,
                    'pattern_confirmation': self.check_pattern_confirmation,
                    'volume_confirmation': self.check_volume_confirmation
                }
            
            def check_support_resistance(self, price, historical_data):
                """Check if price is near support or resistance levels"""
                # Calculate recent highs and lows
                recent_high = historical_data['high'].rolling(20).max().iloc[-1]
                recent_low = historical_data['low'].rolling(20).min().iloc[-1]
                
                # Define zones (2% tolerance)
                resistance_zone = (recent_high * 0.98, recent_high * 1.02)
                support_zone = (recent_low * 0.98, recent_low * 1.02)
                
                near_resistance = resistance_zone[0] <= price <= resistance_zone[1]
                near_support = support_zone[0] <= price <= support_zone[1]
                
                return {
                    'near_resistance': near_resistance,
                    'near_support': near_support,
                    'resistance_level': recent_high,
                    'support_level': recent_low
                }
            
            def check_trend_alignment(self, signal, historical_data):
                """Check if signal aligns with current trend"""
                # Calculate trend using moving averages
                sma_50 = historical_data['close'].rolling(50).mean().iloc[-1]
                sma_200 = historical_data['close'].rolling(200).mean().iloc[-1]
                current_price = historical_data['close'].iloc[-1]
                
                if pd.isna(sma_200):  # Not enough data
                    trend = 'unknown'
                elif current_price > sma_50 > sma_200:
                    trend = 'strong_uptrend'
                elif current_price > sma_50:
                    trend = 'uptrend'
                elif current_price < sma_50 < sma_200:
                    trend = 'strong_downtrend'
                elif current_price < sma_50:
                    trend = 'downtrend'
                else:
                    trend = 'neutral'
                
                # Check alignment
                aligned = False
                if signal == 'buy' and trend in ['uptrend', 'strong_uptrend']:
                    aligned = True
                elif signal == 'sell' and trend in ['downtrend', 'strong_downtrend']:
                    aligned = True
                elif signal == 'hold' and trend == 'neutral':
                    aligned = True
                
                return {
                    'trend': trend,
                    'aligned': aligned,
                    'sma_50': sma_50,
                    'sma_200': sma_200
                }
            
            def check_pattern_confirmation(self, signal, historical_data):
                """Check for candlestick pattern confirmation"""
                # Simple pattern checks
                last_candles = historical_data.tail(3)
                
                # Bullish patterns
                bullish_engulfing = (
                    last_candles.iloc[-2]['close'] < last_candles.iloc[-2]['open'] and  # Red candle
                    last_candles.iloc[-1]['close'] > last_candles.iloc[-1]['open'] and  # Green candle
                    last_candles.iloc[-1]['open'] < last_candles.iloc[-2]['close'] and  # Gap down
                    last_candles.iloc[-1]['close'] > last_candles.iloc[-2]['open']      # Engulfs
                )
                
                # Bearish patterns
                bearish_engulfing = (
                    last_candles.iloc[-2]['close'] > last_candles.iloc[-2]['open'] and  # Green candle
                    last_candles.iloc[-1]['close'] < last_candles.iloc[-1]['open'] and  # Red candle
                    last_candles.iloc[-1]['open'] > last_candles.iloc[-2]['close'] and  # Gap up
                    last_candles.iloc[-1]['close'] < last_candles.iloc[-2]['open']      # Engulfs
                )
                
                pattern_confirms = False
                if signal == 'buy' and bullish_engulfing:
                    pattern_confirms = True
                elif signal == 'sell' and bearish_engulfing:
                    pattern_confirms = True
                
                return {
                    'bullish_pattern': bullish_engulfing,
                    'bearish_pattern': bearish_engulfing,
                    'confirms_signal': pattern_confirms
                }
            
            def check_volume_confirmation(self, signal, historical_data):
                """Check if volume confirms the signal"""
                avg_volume = historical_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = historical_data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Volume should be above average for breakouts
                volume_confirms = False
                if signal in ['buy', 'sell'] and volume_ratio > 1.2:
                    volume_confirms = True
                elif signal == 'hold' and 0.8 <= volume_ratio <= 1.2:
                    volume_confirms = True
                
                return {
                    'volume_ratio': volume_ratio,
                    'confirms_signal': volume_confirms,
                    'avg_volume': avg_volume,
                    'current_volume': current_volume
                }
            
            def validate_signal(self, signal, price, historical_data):
                """Validate AI signal with technical analysis"""
                validations = {}
                
                for rule_name, rule_func in self.rules.items():
                    if rule_name == 'support_resistance':
                        validations[rule_name] = rule_func(price, historical_data)
                    else:
                        validations[rule_name] = rule_func(signal, historical_data)
                
                # Calculate overall validation score
                score = 0
                if validations['trend_alignment']['aligned']:
                    score += 0.4
                if validations['volume_confirmation']['confirms_signal']:
                    score += 0.3
                if validations['pattern_confirmation']['confirms_signal']:
                    score += 0.2
                
                # Adjust for support/resistance
                if signal == 'buy' and validations['support_resistance']['near_support']:
                    score += 0.1
                elif signal == 'sell' and validations['support_resistance']['near_resistance']:
                    score += 0.1
                
                validations['overall_score'] = score
                validations['recommendation'] = 'accept' if score >= 0.5 else 'reject'
                
                return validations
        
        # Generate test data
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
        
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.uniform(-0.5, 0.5, 250),
            'high': prices + np.random.uniform(0, 1, 250),
            'low': prices + np.random.uniform(-1, 0, 250),
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, 250)
        })
        
        # Test validation
        validator = TechnicalAnalysisValidator()
        
        # Test buy signal in uptrend
        signal = 'buy'
        price = historical_data['close'].iloc[-1]
        validation = validator.validate_signal(signal, price, historical_data)
        
        assert 'overall_score' in validation
        assert 0 <= validation['overall_score'] <= 1
        assert validation['recommendation'] in ['accept', 'reject']
    
    def test_risk_management_integration(self):
        """Test comprehensive risk management integration"""
        class RiskManager:
            def __init__(self, account_balance=100000, max_risk_per_trade=0.02):
                self.account_balance = account_balance
                self.max_risk_per_trade = max_risk_per_trade
                self.open_positions = []
                self.daily_loss_limit = 0.05  # 5% daily loss limit
                self.max_positions = 5
                self.correlation_threshold = 0.7
            
            def calculate_position_size(self, entry_price, stop_loss, volatility):
                """Calculate position size based on risk"""
                # Kelly Criterion adjusted for volatility
                risk_amount = self.account_balance * self.max_risk_per_trade
                price_risk = abs(entry_price - stop_loss)
                
                if price_risk == 0:
                    return 0
                
                # Base position size
                position_size = risk_amount / price_risk
                
                # Adjust for volatility
                volatility_adjustment = 1 / (1 + volatility)  # Lower size in high volatility
                adjusted_size = position_size * volatility_adjustment
                
                # Max position value check
                max_position_value = self.account_balance * 0.2  # Max 20% per position
                max_shares = max_position_value / entry_price
                
                return min(adjusted_size, max_shares)
            
            def check_correlation_risk(self, new_symbol, correlation_matrix):
                """Check correlation with existing positions"""
                if not self.open_positions:
                    return True, []
                
                high_correlations = []
                for position in self.open_positions:
                    if position['symbol'] in correlation_matrix and new_symbol in correlation_matrix:
                        correlation = correlation_matrix[position['symbol']][new_symbol]
                        if abs(correlation) > self.correlation_threshold:
                            high_correlations.append({
                                'symbol': position['symbol'],
                                'correlation': correlation
                            })
                
                # Allow if no high correlations
                return len(high_correlations) == 0, high_correlations
            
            def check_daily_loss_limit(self):
                """Check if daily loss limit has been reached"""
                daily_pnl = sum(p.get('realized_pnl', 0) for p in self.open_positions)
                daily_loss_pct = abs(daily_pnl) / self.account_balance
                
                return daily_loss_pct < self.daily_loss_limit
            
            def calculate_portfolio_var(self, confidence_level=0.95):
                """Calculate portfolio Value at Risk"""
                if not self.open_positions:
                    return 0
                
                # Mock VaR calculation
                position_values = [p['size'] * p['current_price'] for p in self.open_positions]
                portfolio_value = sum(position_values)
                
                # Assume normal distribution
                portfolio_volatility = 0.02  # 2% daily volatility
                z_score = 1.645  # 95% confidence
                
                var = portfolio_value * portfolio_volatility * z_score
                return var
            
            def evaluate_new_trade(self, signal):
                """Evaluate if new trade should be taken based on risk"""
                checks = {
                    'position_limit': len(self.open_positions) < self.max_positions,
                    'daily_loss_limit': self.check_daily_loss_limit(),
                    'correlation_check': True,  # Would check actual correlations
                    'var_limit': self.calculate_portfolio_var() < self.account_balance * 0.1
                }
                
                # Calculate overall risk score
                risk_score = sum(checks.values()) / len(checks)
                
                return {
                    'approved': all(checks.values()),
                    'risk_score': risk_score,
                    'checks': checks,
                    'position_size': 0  # Would calculate actual size
                }
        
        # Test risk manager
        risk_manager = RiskManager()
        
        # Add some positions
        risk_manager.open_positions = [
            {'symbol': 'SPY', 'size': 100, 'current_price': 450, 'realized_pnl': -500},
            {'symbol': 'QQQ', 'size': 50, 'current_price': 380, 'realized_pnl': 200}
        ]
        
        # Test new trade evaluation
        new_signal = {
            'symbol': 'IWM',
            'action': 'buy',
            'price': 200,
            'stop_loss': 195
        }
        
        evaluation = risk_manager.evaluate_new_trade(new_signal)
        
        assert 'approved' in evaluation
        assert 'risk_score' in evaluation
        assert 0 <= evaluation['risk_score'] <= 1
        assert 'checks' in evaluation
    
    def test_market_microstructure_awareness(self):
        """Test consideration of market microstructure effects"""
        class MarketMicrostructure:
            def __init__(self):
                self.min_tick_size = 0.01
                self.typical_spread = 0.05
                self.market_hours = {
                    'pre_market': (4, 9.5),  # 4:00 AM - 9:30 AM
                    'regular': (9.5, 16),    # 9:30 AM - 4:00 PM
                    'after_hours': (16, 20)  # 4:00 PM - 8:00 PM
                }
            
            def calculate_spread_cost(self, price, size, time_of_day='regular'):
                """Calculate expected spread cost"""
                base_spread_pct = self.typical_spread / price
                
                # Adjust for time of day
                if time_of_day == 'pre_market':
                    spread_multiplier = 3.0  # Wider spreads
                elif time_of_day == 'after_hours':
                    spread_multiplier = 2.0
                else:
                    spread_multiplier = 1.0
                
                # Adjust for size (larger orders face more slippage)
                size_multiplier = 1 + (size / 10000)  # Increases with size
                
                total_spread_pct = base_spread_pct * spread_multiplier * size_multiplier
                spread_cost = price * size * total_spread_pct / 2  # Pay half the spread
                
                return spread_cost, total_spread_pct
            
            def estimate_market_impact(self, size, avg_volume, volatility):
                """Estimate market impact of order"""
                # Participation rate
                participation_rate = size / avg_volume
                
                # Square root market impact model
                # Impact = k * volatility * sqrt(participation_rate)
                k = 0.1  # Impact coefficient
                impact = k * volatility * np.sqrt(participation_rate)
                
                return {
                    'impact_pct': impact,
                    'participation_rate': participation_rate,
                    'executable': participation_rate < 0.1  # Don't exceed 10% of volume
                }
            
            def calculate_optimal_execution_schedule(self, total_size, time_horizon_minutes):
                """Calculate optimal execution schedule (VWAP-like)"""
                # Typical intraday volume distribution (U-shaped)
                hours = np.linspace(9.5, 16, 13)  # Half-hour intervals
                volume_profile = np.array([
                    0.15, 0.12, 0.08, 0.06, 0.05, 0.05, 0.05,  # Morning
                    0.05, 0.05, 0.06, 0.08, 0.12, 0.15  # Afternoon
                ])
                
                # Calculate execution schedule
                time_slots = min(time_horizon_minutes // 30, len(volume_profile))
                
                if time_slots > 0:
                    # Distribute according to volume profile
                    relevant_profile = volume_profile[:time_slots]
                    relevant_profile = relevant_profile / relevant_profile.sum()
                    
                    schedule = []
                    remaining_size = total_size
                    
                    for i, pct in enumerate(relevant_profile[:-1]):
                        size = int(total_size * pct)
                        schedule.append({
                            'time_slot': i,
                            'minutes_from_start': i * 30,
                            'size': size,
                            'volume_pct': pct
                        })
                        remaining_size -= size
                    
                    # Add remaining size to last slot to ensure total equals total_size
                    if len(relevant_profile) > 0:
                        schedule.append({
                            'time_slot': len(relevant_profile) - 1,
                            'minutes_from_start': (len(relevant_profile) - 1) * 30,
                            'size': remaining_size,
                            'volume_pct': relevant_profile[-1]
                        })
                    
                    return schedule
                else:
                    return [{'time_slot': 0, 'minutes_from_start': 0, 'size': total_size, 'volume_pct': 1.0}]
            
            def check_execution_feasibility(self, signal):
                """Check if signal can be executed given market conditions"""
                current_hour = datetime.now().hour + datetime.now().minute / 60
                
                # Determine market session
                if self.market_hours['pre_market'][0] <= current_hour < self.market_hours['pre_market'][1]:
                    session = 'pre_market'
                elif self.market_hours['regular'][0] <= current_hour < self.market_hours['regular'][1]:
                    session = 'regular'
                elif self.market_hours['after_hours'][0] <= current_hour < self.market_hours['after_hours'][1]:
                    session = 'after_hours'
                else:
                    session = 'closed'
                
                feasibility = {
                    'market_open': session != 'closed',
                    'session': session,
                    'liquidity_adequate': session == 'regular',  # Best liquidity
                    'spread_reasonable': session == 'regular'
                }
                
                feasibility['executable'] = feasibility['market_open']
                
                return feasibility
        
        # Test market microstructure
        market = MarketMicrostructure()
        
        # Test spread cost calculation
        spread_cost, spread_pct = market.calculate_spread_cost(
            price=100, 
            size=1000, 
            time_of_day='regular'
        )
        assert spread_cost > 0
        assert spread_pct > 0
        
        # Test market impact
        impact = market.estimate_market_impact(
            size=10000,
            avg_volume=1000000,
            volatility=0.02
        )
        assert 'impact_pct' in impact
        assert 'executable' in impact
        assert impact['participation_rate'] == 0.01  # 1% of volume
        
        # Test execution schedule
        schedule = market.calculate_optimal_execution_schedule(
            total_size=10000,
            time_horizon_minutes=180  # 3 hours
        )
        assert len(schedule) > 0
        assert sum(s['size'] for s in schedule) == 10000
        
        # Test execution feasibility
        signal = {'symbol': 'SPY', 'action': 'buy', 'size': 1000}
        feasibility = market.check_execution_feasibility(signal)
        assert 'executable' in feasibility
        assert 'session' in feasibility
    
    def test_adaptive_strategy_selection(self, market_conditions):
        """Test adaptive strategy selection based on market conditions"""
        class StrategySelector:
            def __init__(self):
                self.strategies = {
                    'trend_following': {
                        'optimal_conditions': ['bull_market', 'strong_trend'],
                        'avoid_conditions': ['sideways_market', 'high_volatility'],
                        'parameters': {'lookback': 50, 'threshold': 0.02}
                    },
                    'mean_reversion': {
                        'optimal_conditions': ['sideways_market', 'low_volatility'],
                        'avoid_conditions': ['strong_trend', 'high_volatility'],
                        'parameters': {'lookback': 20, 'z_score_threshold': 2}
                    },
                    'volatility_breakout': {
                        'optimal_conditions': ['high_volatility', 'news_driven'],
                        'avoid_conditions': ['low_volatility'],
                        'parameters': {'atr_multiplier': 2, 'confirmation_bars': 3}
                    },
                    'momentum': {
                        'optimal_conditions': ['bull_market', 'bear_market'],
                        'avoid_conditions': ['sideways_market'],
                        'parameters': {'rsi_period': 14, 'momentum_threshold': 60}
                    }
                }
            
            def assess_market_conditions(self, market_data):
                """Assess current market conditions"""
                # Calculate metrics
                returns = market_data['close'].pct_change()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                # Trend strength
                sma_20 = market_data['close'].rolling(20).mean()
                sma_50 = market_data['close'].rolling(50).mean()
                trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
                
                # Classify conditions
                conditions = []
                
                if volatility > 0.25:
                    conditions.append('high_volatility')
                elif volatility < 0.15:
                    conditions.append('low_volatility')
                
                if trend_strength > 0.02:
                    if sma_20.iloc[-1] > sma_50.iloc[-1]:
                        conditions.append('bull_market')
                    else:
                        conditions.append('bear_market')
                    conditions.append('strong_trend')
                else:
                    conditions.append('sideways_market')
                
                return {
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'conditions': conditions
                }
            
            def select_strategy(self, market_assessment):
                """Select optimal strategy based on conditions"""
                scores = {}
                
                for strategy_name, strategy_config in self.strategies.items():
                    score = 0
                    
                    # Check optimal conditions
                    for condition in market_assessment['conditions']:
                        if condition in strategy_config['optimal_conditions']:
                            score += 1
                        elif condition in strategy_config['avoid_conditions']:
                            score -= 1
                    
                    scores[strategy_name] = score
                
                # Select strategy with highest score
                best_strategy = max(scores, key=scores.get)
                
                return {
                    'selected_strategy': best_strategy,
                    'scores': scores,
                    'parameters': self.strategies[best_strategy]['parameters'],
                    'confidence': scores[best_strategy] / len(market_assessment['conditions'])
                }
        
        # Generate test market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate trending market
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        prices = trend + noise
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        # Test strategy selection
        selector = StrategySelector()
        
        # Assess market
        assessment = selector.assess_market_conditions(market_data)
        assert 'conditions' in assessment
        assert len(assessment['conditions']) > 0
        
        # Select strategy
        selection = selector.select_strategy(assessment)
        assert 'selected_strategy' in selection
        assert selection['selected_strategy'] in selector.strategies
        assert 'parameters' in selection
        assert 'confidence' in selection 