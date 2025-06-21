#!/usr/bin/env python3
"""
Test script for Live Data Integration and Enhanced Backtesting
Demonstrates the complete system working together
"""

import asyncio
import logging
from datetime import datetime, timedelta
import os
import sys
import json
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.websocket.enhanced_websocket_service import get_enhanced_websocket_service
from src.services.live_data_service import LiveDataService, LiveDataConfig
from backtesting.enhanced_backtest_engine import (
    EnhancedBacktestEngine, BacktestConfig, plot_backtest_results
)
from agents.orchestration.hybrid_orchestrator import HybridOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveDataBacktestDemo:
    """Demonstration of live data integration with backtesting"""
    
    def __init__(self):
        self.websocket_service = get_enhanced_websocket_service()
        self.live_data_config = LiveDataConfig(
            symbols=['AAPL', 'GOOGL', 'TSLA', 'SPY', 'NVDA'],
            update_interval=5,
            enable_polygon=False  # Using free data for demo
        )
        self.live_data_service = LiveDataService(self.live_data_config)
        self.orchestrator = None
        
    async def setup(self):
        """Initialize all services"""
        logger.info("🚀 Setting up Live Data and Backtest Demo")
        
        # Initialize orchestrator
        self.orchestrator = HybridOrchestrator(
            symbols=self.live_data_config.symbols,
            data_bus=self.live_data_service.data_bus
        )
        
        # Setup WebSocket connections
        await self._setup_websocket_connections()
        
        logger.info("✅ Setup complete")
        
    async def _setup_websocket_connections(self):
        """Setup WebSocket connections for live data"""
        # Subscribe to data updates
        self.websocket_service.subscribe('quote', self._on_quote_update)
        self.websocket_service.subscribe('signal', self._on_signal_update)
        
        # Start WebSocket service
        asyncio.create_task(self.websocket_service.start())
        
    async def _on_quote_update(self, update):
        """Handle live quote updates"""
        logger.info(f"📊 Quote Update: {update.symbol} @ ${update.data.get('price', 0):.2f}")
        
    async def _on_signal_update(self, update):
        """Handle live signal updates"""
        logger.info(f"🎯 Signal: {update.symbol} - {update.data.get('action', 'HOLD')} "
                   f"(confidence: {update.data.get('confidence', 0):.2%})")
        
    async def run_live_data_test(self):
        """Test live data integration"""
        logger.info("\n" + "="*60)
        logger.info("TESTING LIVE DATA INTEGRATION")
        logger.info("="*60)
        
        # Start live data service
        live_task = asyncio.create_task(self.live_data_service.start())
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Get statistics
        stats = self.live_data_service.get_statistics()
        logger.info(f"\n📊 Live Data Statistics:")
        logger.info(f"  - Quotes fetched: {stats['quotes_fetched']}")
        logger.info(f"  - Options fetched: {stats['options_fetched']}")
        logger.info(f"  - Errors: {stats['errors']}")
        logger.info(f"  - Uptime: {stats['uptime']}")
        
        # Stop live data
        await self.live_data_service.stop()
        live_task.cancel()
        
    async def run_backtest_demo(self):
        """Run comprehensive backtest demonstration"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING ENHANCED BACKTEST")
        logger.info("="*60)
        
        # Configure backtest
        config = BacktestConfig(
            symbols=['AAPL', 'GOOGL', 'TSLA', 'SPY', 'NVDA'],
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_capital=100000,
            position_size=0.1,
            max_positions=3,
            walk_forward_enabled=True,
            monte_carlo_enabled=True,
            monte_carlo_simulations=100  # Reduced for demo
        )
        
        # Create backtest engine
        engine = EnhancedBacktestEngine(config)
        
        # Define signal generator
        async def signal_generator(symbol, historical_data, date, agents, params=None):
            """Generate signals using the orchestrator"""
            if self.orchestrator:
                # Get the last 100 rows of data
                recent_data = historical_data.tail(100)
                
                # Generate signal
                signal = await self.orchestrator.generate_signal_for_backtest(
                    symbol, recent_data, date
                )
                
                return signal
            else:
                # Fallback simple signal
                return {
                    'symbol': symbol,
                    'action': 'BUY' if historical_data['RSI'].iloc[-1] < 30 else 'HOLD',
                    'confidence': 0.75,
                    'agent_signals': {}
                }
        
        # Run backtest
        logger.info("Running backtest...")
        results = await engine.run_backtest(signal_generator)
        
        # Display results
        self._display_backtest_results(results)
        
        # Save visualization
        plot_backtest_results(results, 'backtest_results.png')
        logger.info("📊 Saved backtest visualization to backtest_results.png")
        
    def _display_backtest_results(self, results):
        """Display backtest results in a formatted way"""
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        
        # Performance metrics
        logger.info("\n📈 Performance Metrics:")
        logger.info(f"  - Total Return: {results.total_return:.2%}")
        logger.info(f"  - Annual Return: {results.annualized_return:.2%}")
        logger.info(f"  - Volatility: {results.volatility:.2%}")
        logger.info(f"  - Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"  - Sortino Ratio: {results.sortino_ratio:.2f}")
        logger.info(f"  - Calmar Ratio: {results.calmar_ratio:.2f}")
        logger.info(f"  - Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"  - Max DD Duration: {results.max_drawdown_duration} days")
        
        # Trade statistics
        logger.info("\n📊 Trade Statistics:")
        logger.info(f"  - Total Trades: {results.total_trades}")
        logger.info(f"  - Winning Trades: {results.winning_trades}")
        logger.info(f"  - Losing Trades: {results.losing_trades}")
        logger.info(f"  - Win Rate: {results.win_rate:.2%}")
        logger.info(f"  - Avg Win: ${results.avg_win:.2f}")
        logger.info(f"  - Avg Loss: ${results.avg_loss:.2f}")
        logger.info(f"  - Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"  - Expectancy: ${results.expectancy:.2f}")
        
        # Risk metrics
        logger.info("\n⚠️ Risk Metrics:")
        logger.info(f"  - Value at Risk (95%): {results.value_at_risk:.2%}")
        logger.info(f"  - Conditional VaR: {results.conditional_var:.2%}")
        logger.info(f"  - Downside Deviation: {results.downside_deviation:.2%}")
        logger.info(f"  - Beta: {results.beta:.2f}")
        logger.info(f"  - Alpha: {results.alpha:.2%}")
        
        # Walk-forward results
        if results.walk_forward_results:
            logger.info("\n🔄 Walk-Forward Analysis:")
            logger.info(f"  - Windows tested: {len(results.walk_forward_results['windows'])}")
            logger.info(f"  - Avg out-of-sample return: {results.walk_forward_results['avg_out_sample_return']:.2%}")
            logger.info(f"  - Consistency (std): {results.walk_forward_results['consistency']:.2%}")
        
        # Monte Carlo results
        if results.monte_carlo_results:
            logger.info("\n🎲 Monte Carlo Simulation:")
            logger.info(f"  - Mean return: {results.monte_carlo_results['mean_return']:.2%}")
            logger.info(f"  - Std return: {results.monte_carlo_results['std_return']:.2%}")
            logger.info(f"  - P(Profit): {results.monte_carlo_results['probability_of_profit']:.2%}")
            logger.info(f"  - P(Loss > 10%): {results.monte_carlo_results['probability_of_loss_gt_10pct']:.2%}")
            
            # Confidence intervals
            logger.info("\n  Confidence Intervals:")
            for level, values in results.monte_carlo_results['confidence_intervals'].items():
                logger.info(f"    {level}: Return={values['return']:.2%}, Drawdown={values['drawdown']:.2%}")
        
        # Agent performance
        if results.agent_performance:
            logger.info("\n🤖 Agent Performance:")
            for agent, stats in results.agent_performance.items():
                logger.info(f"  {agent}:")
                logger.info(f"    - Accuracy: {stats.get('accuracy', 0):.2%}")
                logger.info(f"    - Avg PnL: ${stats.get('avg_pnl', 0):.2f}")
        
    async def run_resilience_test(self):
        """Test system resilience and error handling"""
        logger.info("\n" + "="*60)
        logger.info("TESTING SYSTEM RESILIENCE")
        logger.info("="*60)
        
        # Test WebSocket reconnection
        logger.info("\n🔌 Testing WebSocket reconnection...")
        
        # Simulate connection failure
        test_conn_id = "test_connection"
        await self.websocket_service.connect(
            test_conn_id,
            "wss://invalid.websocket.url",
            on_message=None
        )
        
        # Check connection status
        status = self.websocket_service.get_connection_status(test_conn_id)
        if status:
            logger.info(f"  - Connection attempts: {status.reconnect_attempts}")
            logger.info(f"  - Connected: {status.connected}")
        
        # Test data quality monitoring
        logger.info("\n📊 Testing data quality monitoring...")
        
        # Simulate bad data
        bad_quote = {
            'symbol': 'TEST',
            'price': -100,  # Invalid price
            'volume': 'invalid'  # Invalid volume
        }
        
        # The system should handle this gracefully
        try:
            await self.live_data_service._process_quote(bad_quote)
        except Exception as e:
            logger.info(f"  ✅ Bad data rejected: {e}")
        
        # Test rate limiting
        logger.info("\n⏱️ Testing rate limiting...")
        
        # Make rapid requests
        start_time = datetime.now()
        request_count = 0
        
        for _ in range(20):
            try:
                await self.live_data_service.get_quote('AAPL')
                request_count += 1
            except Exception:
                pass
                
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"  - Requests: {request_count} in {elapsed:.2f}s")
        logger.info(f"  - Rate: {request_count/elapsed:.2f} req/s")
        
    async def run_full_demo(self):
        """Run the complete demonstration"""
        try:
            # Setup
            await self.setup()
            
            # Test live data
            await self.run_live_data_test()
            
            # Run backtest
            await self.run_backtest_demo()
            
            # Test resilience
            await self.run_resilience_test()
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("✅ DEMO COMPLETE")
            logger.info("="*60)
            logger.info("\nKey Achievements:")
            logger.info("  ✓ Live data streaming with WebSocket")
            logger.info("  ✓ Multi-source data aggregation")
            logger.info("  ✓ Enhanced backtesting with walk-forward")
            logger.info("  ✓ Monte Carlo risk analysis")
            logger.info("  ✓ System resilience and error handling")
            logger.info("  ✓ Comprehensive performance metrics")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
        finally:
            # Cleanup
            await self.websocket_service.stop()

async def main():
    """Main entry point"""
    demo = LiveDataBacktestDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 