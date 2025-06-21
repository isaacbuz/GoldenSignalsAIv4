#!/usr/bin/env python3
"""
Script to integrate ML backtesting with the existing GoldenSignalsAI backend
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_enhanced_backtest_system import MLBacktestEngine, SignalAccuracyImprover
from advanced_backtest_system import AdvancedBacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLBacktestingIntegration:
    """
    Integrates ML backtesting capabilities with the existing backend
    """
    
    def __init__(self):
        self.ml_engine = MLBacktestEngine()
        self.accuracy_improver = SignalAccuracyImprover()
        self.advanced_engine = AdvancedBacktestEngine()
        self.signal_accuracy_history = []
        
    async def validate_current_signals(self, signals: list) -> dict:
        """
        Validate current signals against historical performance
        """
        validation_results = []
        
        for signal in signals:
            symbol = signal.get('symbol')
            action = signal.get('action', signal.get('type', 'BUY'))
            confidence = signal.get('confidence', 0.5)
            
            # Fetch recent historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            df = self.ml_engine.fetch_historical_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if not df.empty:
                # Engineer features
                df = self.ml_engine.engineer_features(df)
                
                # Calculate validation metrics
                recent_volatility = df['volatility'].iloc[-1] if 'volatility' in df else 0.2
                recent_returns = df['returns'].mean() * 252 if 'returns' in df else 0
                
                # Simple scoring based on confidence and market conditions
                risk_score = confidence / (1 + recent_volatility)
                expected_return = confidence * recent_returns
                
                validation = {
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'risk_score': risk_score,
                    'expected_return': expected_return,
                    'volatility': recent_volatility,
                    'recommendation': 'PROCEED' if risk_score > 0.5 else 'CAUTION'
                }
                
                validation_results.append(validation)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'validations': validation_results,
            'summary': {
                'total_signals': len(signals),
                'proceed_count': sum(1 for v in validation_results if v['recommendation'] == 'PROCEED'),
                'caution_count': sum(1 for v in validation_results if v['recommendation'] == 'CAUTION')
            }
        }
    
    async def improve_signal_generation(self, symbols: list) -> dict:
        """
        Get recommendations to improve signal generation
        """
        logger.info(f"Running signal improvement analysis for {symbols}")
        
        improvements = await self.accuracy_improver.improve_signals(symbols)
        
        # Create actionable recommendations
        actionable_recommendations = {
            'feature_priorities': [],
            'parameter_adjustments': {},
            'risk_thresholds': {},
            'implementation_code': []
        }
        
        # Extract top features
        for feature, importance in improvements['recommended_features'][:10]:
            actionable_recommendations['feature_priorities'].append({
                'feature': feature,
                'importance': importance,
                'description': self._get_feature_description(feature)
            })
        
        # Parameter adjustments
        actionable_recommendations['parameter_adjustments'] = {
            'min_confidence': max(0.6, improvements['optimal_parameters'].get('min_win_rate', 0.55)),
            'position_size': min(0.1, improvements['optimal_parameters'].get('recommended_position_size', 0.05)),
            'stop_loss': improvements['risk_management'].get('stop_loss', 0.02),
            'take_profit': improvements['risk_management'].get('take_profit', 0.05)
        }
        
        # Risk thresholds
        actionable_recommendations['risk_thresholds'] = {
            'max_volatility': improvements['risk_management'].get('max_volatility', 0.3),
            'min_sharpe_ratio': improvements['optimal_parameters'].get('min_sharpe_ratio', 0.5),
            'max_drawdown': improvements['optimal_parameters'].get('max_acceptable_drawdown', 0.2)
        }
        
        # Generate implementation code snippets
        actionable_recommendations['implementation_code'] = self._generate_implementation_code(improvements)
        
        return actionable_recommendations
    
    def _get_feature_description(self, feature: str) -> str:
        """
        Get human-readable description of a feature
        """
        descriptions = {
            'rsi': 'Relative Strength Index - momentum oscillator',
            'macd': 'Moving Average Convergence Divergence - trend indicator',
            'volatility': 'Historical volatility measure',
            'volume_ratio': 'Current volume vs average volume',
            'returns': 'Price returns',
            'sma_20': '20-period Simple Moving Average',
            'ema_20': '20-period Exponential Moving Average',
            'bb_high': 'Bollinger Band upper band',
            'bb_low': 'Bollinger Band lower band',
            'atr': 'Average True Range - volatility indicator'
        }
        
        return descriptions.get(feature, feature.replace('_', ' ').title())
    
    def _generate_implementation_code(self, improvements: dict) -> list:
        """
        Generate code snippets for implementing improvements
        """
        code_snippets = []
        
        # Signal filter implementation
        filter_code = """
# Implement signal filters based on ML recommendations
def apply_ml_filters(signal):
    # Volume filter
    if signal['volume'] <= signal['volume_sma']:
        signal['confidence'] *= 0.8
    
    # Volatility filter
    if signal['volatility'] > {max_volatility}:
        signal['confidence'] *= 0.7
    
    # Trend filter
    if signal['sma_20'] <= signal['sma_50']:
        signal['confidence'] *= 0.9
    
    # RSI filter
    if not (30 < signal['rsi'] < 70):
        signal['confidence'] *= 0.85
    
    return signal
""".format(max_volatility=improvements['risk_management'].get('max_volatility', 0.3))
        
        code_snippets.append({
            'title': 'Signal Filtering',
            'code': filter_code,
            'description': 'Apply ML-recommended filters to improve signal quality'
        })
        
        # Risk management implementation
        risk_code = """
# Implement ML-recommended risk management
def calculate_position_size(signal, portfolio_value):
    base_size = {position_size} * portfolio_value
    
    # Adjust for confidence
    adjusted_size = base_size * signal['confidence']
    
    # Adjust for volatility
    vol_adjustment = min(1.0, {max_volatility} / signal['volatility'])
    final_size = adjusted_size * vol_adjustment
    
    return min(final_size, base_size)  # Never exceed base size
""".format(
            position_size=improvements['optimal_parameters'].get('recommended_position_size', 0.05),
            max_volatility=improvements['risk_management'].get('max_volatility', 0.3)
        )
        
        code_snippets.append({
            'title': 'Position Sizing',
            'code': risk_code,
            'description': 'Calculate optimal position size based on ML analysis'
        })
        
        return code_snippets
    
    async def run_production_backtest(self, symbols: list, days_back: int = 30) -> dict:
        """
        Run a quick backtest on recent data for production validation
        """
        logger.info(f"Running production backtest for {symbols} over last {days_back} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        results = await self.ml_engine.run_comprehensive_backtest(
            symbols,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'average_sharpe': 0,
            'average_return': 0,
            'average_drawdown': 0,
            'best_performer': None,
            'worst_performer': None
        }
        
        sharpe_ratios = []
        returns = []
        
        for symbol, data in results.items():
            if 'backtest_metrics' in data:
                metrics = data['backtest_metrics']
                sharpe_ratios.append((symbol, metrics['sharpe_ratio']))
                returns.append((symbol, metrics['annual_return']))
                
        if sharpe_ratios:
            aggregate_metrics['average_sharpe'] = sum(s[1] for s in sharpe_ratios) / len(sharpe_ratios)
            aggregate_metrics['best_performer'] = max(sharpe_ratios, key=lambda x: x[1])[0]
            aggregate_metrics['worst_performer'] = min(sharpe_ratios, key=lambda x: x[1])[0]
        
        if returns:
            aggregate_metrics['average_return'] = sum(r[1] for r in returns) / len(returns)
        
        return {
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'symbols': symbols,
            'aggregate_metrics': aggregate_metrics,
            'detailed_results': results
        }
    
    async def track_signal_accuracy(self, signal: dict, actual_outcome: dict) -> None:
        """
        Track signal accuracy for continuous improvement
        """
        accuracy_record = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'outcome': actual_outcome,
            'accuracy': self._calculate_accuracy(signal, actual_outcome)
        }
        
        self.signal_accuracy_history.append(accuracy_record)
        
        # Save to file for persistence
        with open('signal_accuracy_history.json', 'a') as f:
            f.write(json.dumps(accuracy_record) + '\n')
    
    def _calculate_accuracy(self, signal: dict, outcome: dict) -> float:
        """
        Calculate accuracy score for a signal
        """
        predicted_direction = 1 if signal.get('action') == 'BUY' else -1
        actual_direction = 1 if outcome.get('price_change', 0) > 0 else -1
        
        # Basic accuracy: did we predict the right direction?
        direction_correct = predicted_direction == actual_direction
        
        # Weighted by confidence
        accuracy = signal.get('confidence', 0.5) if direction_correct else 1 - signal.get('confidence', 0.5)
        
        return accuracy


async def main():
    """
    Demonstrate ML backtesting integration
    """
    integration = MLBacktestingIntegration()
    
    # Example 1: Validate current signals
    print("\n" + "="*60)
    print("1. Validating Current Signals")
    print("="*60)
    
    sample_signals = [
        {'symbol': 'AAPL', 'action': 'BUY', 'confidence': 0.75},
        {'symbol': 'GOOGL', 'action': 'SELL', 'confidence': 0.60},
        {'symbol': 'MSFT', 'action': 'BUY', 'confidence': 0.80}
    ]
    
    validation_results = await integration.validate_current_signals(sample_signals)
    print(json.dumps(validation_results, indent=2))
    
    # Example 2: Get improvement recommendations
    print("\n" + "="*60)
    print("2. Signal Improvement Recommendations")
    print("="*60)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    improvements = await integration.improve_signal_generation(symbols)
    
    print("\nTop Features to Focus On:")
    for feature in improvements['feature_priorities'][:5]:
        print(f"  - {feature['feature']}: {feature['description']} (importance: {feature['importance']:.3f})")
    
    print("\nRecommended Parameters:")
    for param, value in improvements['parameter_adjustments'].items():
        print(f"  - {param}: {value}")
    
    # Example 3: Run production backtest
    print("\n" + "="*60)
    print("3. Production Backtest (Last 30 Days)")
    print("="*60)
    
    backtest_results = await integration.run_production_backtest(symbols, days_back=30)
    print(f"\nPeriod: {backtest_results['period']}")
    print(f"Average Sharpe Ratio: {backtest_results['aggregate_metrics']['average_sharpe']:.2f}")
    print(f"Average Return: {backtest_results['aggregate_metrics']['average_return']:.2%}")
    print(f"Best Performer: {backtest_results['aggregate_metrics']['best_performer']}")
    print(f"Worst Performer: {backtest_results['aggregate_metrics']['worst_performer']}")
    
    # Save integration config
    integration_config = {
        'ml_backtesting_enabled': True,
        'validation_threshold': 0.6,
        'backtest_frequency': 'weekly',
        'feature_priorities': [f['feature'] for f in improvements['feature_priorities'][:10]],
        'risk_parameters': improvements['risk_thresholds'],
        'position_sizing': improvements['parameter_adjustments']
    }
    
    with open('ml_integration_config.json', 'w') as f:
        json.dump(integration_config, f, indent=2)
    
    print("\n✅ ML Backtesting Integration Complete!")
    print("📊 Configuration saved to ml_integration_config.json")
    print("🚀 Ready to improve signal accuracy with ML insights")


if __name__ == "__main__":
    asyncio.run(main()) 