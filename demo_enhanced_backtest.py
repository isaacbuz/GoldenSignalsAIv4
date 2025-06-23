#!/usr/bin/env python3
"""
Demo script for enhanced ML backtesting with Phase 2 integration
Shows how signal generation, filtering, and monitoring enhance backtest results
"""

import asyncio
import json
from datetime import datetime, timedelta
from ml_enhanced_backtest_system import MLBacktestEngine, SignalAccuracyImprover


async def run_enhanced_backtest_demo():
    """Demonstrate enhanced backtesting with signal quality tracking"""
    
    print("🚀 GoldenSignalsAI Enhanced Backtesting Demo")
    print("=" * 60)
    print("Integrating ML backtesting with Phase 2 signal services")
    print("=" * 60)
    
    # Initialize backtest engine
    engine = MLBacktestEngine()
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    
    # Run backtest with shorter period for demo
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year backtest
    
    print(f"\n📊 Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run comprehensive backtest
    results = await engine.run_comprehensive_backtest(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Display results
    print("\n" + "="*80)
    print("📈 BACKTEST RESULTS SUMMARY")
    print("="*80)
    
    for symbol, data in results.items():
        print(f"\n🎯 {symbol}")
        print("-" * 40)
        
        # Basic metrics
        metrics = data['backtest_metrics']
        print(f"📊 Performance Metrics:")
        print(f"  • Annual Return: {metrics['annual_return']:.2%}")
        print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  • Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  • Win Rate: {metrics['win_rate']:.2%}")
        print(f"  • Profit Factor: {metrics['profit_factor']:.2f}")
        
        # ML model performance
        model_perf = data['model_performance']
        print(f"\n🤖 ML Model Performance:")
        print(f"  • Accuracy: {model_perf['accuracy']:.2%}")
        print(f"  • Precision: {model_perf['precision']:.2%}")
        print(f"  • Recall: {model_perf['recall']:.2%}")
        
        # Signal quality metrics (Phase 2)
        signal_quality = data.get('signal_quality', {})
        if signal_quality:
            print(f"\n✨ Signal Quality Metrics (Phase 2):")
            print(f"  • Avg Confidence: {signal_quality.get('avg_signal_confidence', 0):.2f}")
            print(f"  • Avg Quality Score: {signal_quality.get('avg_signal_quality', 0):.2f}")
            print(f"  • Filter Pass Rate: {signal_quality.get('filter_pass_rate', 0):.2%}")
            print(f"  • Signal Win Rate: {signal_quality.get('signal_win_rate', 0):.2%}")
            print(f"  • Signal Sharpe: {signal_quality.get('signal_sharpe', 0):.2f}")
        
        # Top features
        print(f"\n📊 Top 5 Important Features:")
        for i, (feature, importance) in enumerate(data['feature_importance'][:5]):
            print(f"  {i+1}. {feature}: {importance:.3f}")
    
    # Get signal quality summary
    quality_summary = engine.get_signal_quality_summary()
    if quality_summary:
        print("\n" + "="*80)
        print("📊 OVERALL SIGNAL QUALITY SUMMARY")
        print("="*80)
        print(f"Average Confidence: {quality_summary.get('avg_confidence', 0):.2f}")
        print(f"Average Quality: {quality_summary.get('avg_quality', 0):.2f}")
        print(f"Confidence Range: {quality_summary.get('confidence_distribution', {}).get('min', 0):.2f} - {quality_summary.get('confidence_distribution', {}).get('max', 0):.2f}")
    
    # Signal improvement recommendations
    print("\n" + "="*80)
    print("💡 SIGNAL IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    improver = SignalAccuracyImprover()
    improvements = await improver.improve_signals(symbols)
    
    # Display recommendations
    print("\n🎯 Recommended Features (Top 10):")
    for i, (feature, score) in enumerate(improvements['recommended_features'][:10]):
        print(f"  {i+1}. {feature}: {score:.3f}")
    
    print("\n📊 Optimal Parameters:")
    for param, value in improvements['optimal_parameters'].items():
        print(f"  • {param}: {value:.3f}")
    
    print("\n⚠️ Risk Management Rules:")
    for rule, value in improvements['risk_management'].items():
        print(f"  • {rule}: {value:.3f}")
    
    # Save results
    output_file = 'enhanced_backtest_results.json'
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for symbol, data in results.items():
            serializable_results[symbol] = {
                'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                           for k, v in data['backtest_metrics'].items()},
                'model_performance': data['model_performance'],
                'signal_quality': data.get('signal_quality', {}),
                'avg_accuracy': float(data['avg_accuracy'])
            }
        
        json.dump({
            'backtest_results': serializable_results,
            'improvements': {
                'recommended_features': improvements['recommended_features'][:10],
                'optimal_parameters': improvements['optimal_parameters'],
                'risk_management': improvements['risk_management']
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print("\n✅ Enhanced backtest complete!")


if __name__ == "__main__":
    # Handle missing numpy import
    import numpy as np
    
    # Run the demo
    asyncio.run(run_enhanced_backtest_demo()) 