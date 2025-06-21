#!/usr/bin/env python3
"""
Run Signal Accuracy Backtest for GoldenSignalsAI
"""

import sys
import argparse
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append('.')

from backtesting.signal_accuracy_validator import (
    BacktestAccuracyEngine, 
    AccuracyVisualizer,
    integrate_with_orchestrator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run signal accuracy backtest')
    parser.add_argument('--symbols', type=str, default='AAPL,GOOGL,MSFT,TSLA,NVDA',
                       help='Comma-separated list of symbols')
    parser.add_argument('--start', type=str, 
                       default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--use-orchestrator', action='store_true',
                       help='Use actual orchestrator instead of mock signals')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--output', type=str, default='accuracy_report.json',
                       help='Output file for report')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    
    logger.info(f"Starting accuracy backtest...")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {args.start} to {args.end}")
    
    if args.use_orchestrator:
        logger.info("Using actual orchestrator for signal generation")
        try:
            results = integrate_with_orchestrator()
        except ImportError as e:
            logger.error(f"Failed to import orchestrator: {e}")
            logger.info("Falling back to mock signals")
            args.use_orchestrator = False
    
    if not args.use_orchestrator:
        # Use mock signal generator
        def mock_signal_generator(symbol, historical_data, date):
            import random
            import numpy as np
            
            # Simple technical analysis mock
            closes = historical_data['Close'].values
            if len(closes) < 20:
                action = 'HOLD'
                confidence = 0.5
            else:
                # Calculate simple indicators
                sma_20 = np.mean(closes[-20:])
                current_price = closes[-1]
                
                # Generate signal based on price vs SMA
                if current_price > sma_20 * 1.02:
                    action = 'BUY'
                    confidence = min(0.9, 0.6 + (current_price - sma_20) / sma_20)
                elif current_price < sma_20 * 0.98:
                    action = 'SELL'
                    confidence = min(0.9, 0.6 + (sma_20 - current_price) / sma_20)
                else:
                    action = 'HOLD'
                    confidence = 0.5
            
            # Mock agent breakdown
            agents = ['rsi', 'macd', 'volume_spike', 'ma_crossover', 'bollinger', 
                     'stochastic', 'ema', 'atr', 'vwap']
            
            agent_breakdown = {}
            for agent in agents:
                # Add some randomness to agent signals
                agent_action = action if random.random() > 0.3 else random.choice(['BUY', 'SELL', 'HOLD'])
                agent_confidence = confidence * random.uniform(0.8, 1.2)
                agent_breakdown[agent] = {
                    'action': agent_action,
                    'confidence': min(1.0, agent_confidence)
                }
            
            return {
                'symbol': symbol,
                'date': date,
                'action': action,
                'confidence': confidence,
                'metadata': {
                    'agent_breakdown': agent_breakdown
                }
            }
        
        # Run backtest with mock signals
        engine = BacktestAccuracyEngine()
        results = engine.run_accuracy_test(
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
            signal_generator=mock_signal_generator
        )
    
    # Generate visualizations
    if not args.no_plots:
        logger.info("Generating visualizations...")
        visualizer = AccuracyVisualizer()
        
        try:
            visualizer.plot_accuracy_by_symbol(results)
            visualizer.plot_agent_performance(results)
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    # Generate report
    logger.info(f"Generating report: {args.output}")
    report = AccuracyVisualizer.generate_accuracy_report(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    print(f"\nOverall Accuracy: {results['aggregate_metrics']['overall_accuracy']:.2%}")
    print(f"Total Signals Tested: {results['aggregate_metrics']['total_signals']}")
    print(f"Average Accuracy: {results['aggregate_metrics']['avg_accuracy']:.2%}")
    print(f"Standard Deviation: {results['aggregate_metrics']['std_accuracy']:.2%}")
    
    # Check if accuracy meets threshold
    min_accuracy = 0.6  # 60% minimum
    if results['aggregate_metrics']['overall_accuracy'] >= min_accuracy:
        print(f"\n✅ PASSED: Accuracy {results['aggregate_metrics']['overall_accuracy']:.2%} >= {min_accuracy:.0%}")
    else:
        print(f"\n❌ FAILED: Accuracy {results['aggregate_metrics']['overall_accuracy']:.2%} < {min_accuracy:.0%}")
        print("\nRecommendations:")
        print("1. Review agent configurations")
        print("2. Adjust confidence thresholds")
        print("3. Analyze underperforming agents")
        print("4. Consider market condition filters")
    
    print(f"\nDetailed report saved to: {args.output}")
    
    # Return exit code based on accuracy
    return 0 if results['aggregate_metrics']['overall_accuracy'] >= min_accuracy else 1

if __name__ == "__main__":
    sys.exit(main()) 