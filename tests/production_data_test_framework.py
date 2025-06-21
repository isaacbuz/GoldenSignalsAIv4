#!/usr/bin/env python3
"""
Production Data Testing Framework for GoldenSignalsAI V2
Tests with real market data and production-like conditions
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import unittest
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests to run"""
    LIVE_DATA = "live_data"
    HISTORICAL = "historical"
    BACKTESTING = "backtesting"
    STRESS_TEST = "stress_test"
    INTEGRATION = "integration"


@dataclass
class ProductionTestConfig:
    """Configuration for production tests"""
    base_url: str = "http://localhost:8000"
    test_symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ"])
    historical_days: int = 30
    backtest_period: str = "3mo"
    stress_test_concurrent_requests: int = 50
    signal_accuracy_threshold: float = 0.60  # 60% accuracy required
    latency_threshold_ms: float = 500.0
    use_real_time_data: bool = True
    save_test_data: bool = True
    test_data_dir: str = "test_data/production"


@dataclass
class MarketDataSnapshot:
    """Snapshot of market data for testing"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    high: float
    low: float
    open: float
    close: float
    indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class SignalTestResult:
    """Result of signal testing"""
    signal_id: str
    symbol: str
    action: str
    confidence: float
    entry_price: float
    actual_price: float
    predicted_direction: str
    actual_direction: str
    is_accurate: bool
    profit_loss: float
    holding_period: timedelta
    indicators_at_signal: Dict[str, float]
    market_conditions: Dict[str, Any]


class ProductionDataCollector:
    """Collects and manages production data for testing"""
    
    def __init__(self, config: ProductionTestConfig):
        self.config = config
        self.market_data_cache: Dict[str, List[MarketDataSnapshot]] = {}
        self.signal_history: List[Dict[str, Any]] = []
        
    async def collect_live_market_data(self) -> Dict[str, MarketDataSnapshot]:
        """Collect current market data from multiple sources"""
        snapshots = {}
        
        for symbol in self.config.test_symbols:
            try:
                # Get data from yfinance
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current quote
                current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
                
                # Get intraday data for more accurate current values
                intraday = ticker.history(period="1d", interval="1m")
                if not intraday.empty:
                    latest = intraday.iloc[-1]
                    
                    snapshot = MarketDataSnapshot(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        price=float(latest['Close']),
                        volume=int(latest['Volume']),
                        bid=info.get('bid', current_price),
                        ask=info.get('ask', current_price),
                        high=float(latest['High']),
                        low=float(latest['Low']),
                        open=float(latest['Open']),
                        close=float(latest['Close'])
                    )
                    
                    # Calculate indicators
                    snapshot.indicators = await self._calculate_indicators(symbol, intraday)
                    
                    snapshots[symbol] = snapshot
                    logger.info(f"Collected live data for {symbol}: ${snapshot.price:.2f}")
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                
        return snapshots
    
    async def _calculate_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from market data"""
        indicators = {}
        
        try:
            closes = data['Close'].values
            
            # RSI
            if len(closes) >= 14:
                indicators['rsi'] = self._calculate_rsi(closes)
            
            # MACD
            if len(closes) >= 26:
                macd_result = self._calculate_macd(closes)
                indicators.update(macd_result)
            
            # Bollinger Bands
            if len(closes) >= 20:
                bb_result = self._calculate_bollinger_bands(closes)
                indicators.update(bb_result)
            
            # Volume indicators
            volumes = data['Volume'].values
            indicators['volume_sma'] = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        return float(100.0 - (100.0 / (1.0 + rs)))
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate MACD"""
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return {
            'macd': float(macd.iloc[-1]),
            'macd_signal': float(signal.iloc[-1]),
            'macd_histogram': float(macd.iloc[-1] - signal.iloc[-1])
        }
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        return {
            'bb_upper': float(sma + 2 * std),
            'bb_middle': float(sma),
            'bb_lower': float(sma - 2 * std),
            'bb_width': float(4 * std),
            'bb_percent': float((prices[-1] - (sma - 2 * std)) / (4 * std)) if std > 0 else 0.5
        }
    
    async def collect_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Collect historical data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get daily data
            daily_data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            # Get hourly data for more granular analysis
            hourly_data = ticker.history(period=f"{min(days, 60)}d", interval="1h")
            
            logger.info(f"Collected {len(daily_data)} daily and {len(hourly_data)} hourly data points for {symbol}")
            
            return daily_data, hourly_data
            
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()


class SignalAccuracyTester:
    """Tests signal accuracy against real market movements"""
    
    def __init__(self, config: ProductionTestConfig):
        self.config = config
        self.results: List[SignalTestResult] = []
        
    async def test_signal_accuracy(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> SignalTestResult:
        """Test a signal's accuracy against actual market movement"""
        try:
            signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
            symbol = signal['symbol']
            action = signal['action']
            entry_price = signal['price']
            
            # Find market data after signal
            future_data = market_data[market_data.index > signal_time]
            
            if future_data.empty:
                logger.warning(f"No future data available for signal {signal['id']}")
                return None
            
            # Determine holding period based on signal type
            holding_periods = {
                'BUY': timedelta(days=5),
                'SELL': timedelta(days=5),
                'HOLD': timedelta(days=1)
            }
            holding_period = holding_periods.get(action, timedelta(days=3))
            
            # Get price at end of holding period
            exit_time = signal_time + holding_period
            exit_data = future_data[future_data.index <= exit_time]
            
            if exit_data.empty:
                exit_data = future_data.iloc[:1]
            
            exit_price = float(exit_data.iloc[-1]['Close'])
            
            # Calculate actual movement
            price_change = exit_price - entry_price
            price_change_pct = (price_change / entry_price) * 100
            
            # Determine if signal was accurate
            if action == 'BUY':
                is_accurate = price_change > 0
                predicted_direction = 'up'
            elif action == 'SELL':
                is_accurate = price_change < 0
                predicted_direction = 'down'
            else:  # HOLD
                is_accurate = abs(price_change_pct) < 2.0  # Less than 2% movement
                predicted_direction = 'sideways'
            
            actual_direction = 'up' if price_change > 0 else ('down' if price_change < 0 else 'sideways')
            
            # Calculate profit/loss
            if action == 'BUY':
                profit_loss = price_change
            elif action == 'SELL':
                profit_loss = -price_change  # Profit from short position
            else:
                profit_loss = 0
            
            result = SignalTestResult(
                signal_id=signal['id'],
                symbol=symbol,
                action=action,
                confidence=signal['confidence'],
                entry_price=entry_price,
                actual_price=exit_price,
                predicted_direction=predicted_direction,
                actual_direction=actual_direction,
                is_accurate=is_accurate,
                profit_loss=profit_loss,
                holding_period=exit_data.index[-1] - signal_time,
                indicators_at_signal=signal.get('indicators', {}),
                market_conditions={
                    'volatility': float(future_data['Close'].std()),
                    'volume_avg': float(future_data['Volume'].mean()),
                    'price_range': float(future_data['High'].max() - future_data['Low'].min())
                }
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error testing signal accuracy: {e}")
            return None
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall accuracy metrics"""
        if not self.results:
            return {}
        
        accurate_signals = [r for r in self.results if r.is_accurate]
        total_profit = sum(r.profit_loss for r in self.results)
        
        metrics = {
            'total_signals': len(self.results),
            'accurate_signals': len(accurate_signals),
            'accuracy_rate': len(accurate_signals) / len(self.results),
            'total_profit_loss': total_profit,
            'average_profit_per_signal': total_profit / len(self.results),
            'win_rate': len([r for r in self.results if r.profit_loss > 0]) / len(self.results),
            'average_holding_period': sum(r.holding_period.total_seconds() for r in self.results) / len(self.results) / 86400,  # days
            'by_action': {}
        }
        
        # Metrics by action type
        for action in ['BUY', 'SELL', 'HOLD']:
            action_results = [r for r in self.results if r.action == action]
            if action_results:
                metrics['by_action'][action] = {
                    'count': len(action_results),
                    'accuracy': len([r for r in action_results if r.is_accurate]) / len(action_results),
                    'avg_profit': sum(r.profit_loss for r in action_results) / len(action_results)
                }
        
        return metrics


class ProductionIntegrationTester:
    """Tests system integration with production data"""
    
    def __init__(self, config: ProductionTestConfig):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_live_data_flow(self) -> Dict[str, Any]:
        """Test complete data flow with live market data"""
        results = {
            'market_data_test': False,
            'signal_generation_test': False,
            'websocket_test': False,
            'latency_ms': 0,
            'errors': []
        }
        
        try:
            # Test 1: Fetch live market data
            start_time = datetime.now()
            async with self.session.get(f"{self.config.base_url}/api/v1/market-data/SPY") as resp:
                if resp.status == 200:
                    market_data = await resp.json()
                    results['market_data_test'] = True
                    results['market_data'] = market_data
                else:
                    results['errors'].append(f"Market data API returned {resp.status}")
            
            # Test 2: Generate signals based on live data
            async with self.session.get(f"{self.config.base_url}/api/v1/signals/SPY") as resp:
                if resp.status == 200:
                    signals = await resp.json()
                    results['signal_generation_test'] = len(signals) > 0
                    results['signal_count'] = len(signals)
                else:
                    results['errors'].append(f"Signals API returned {resp.status}")
            
            # Test 3: WebSocket connection
            # This would require websocket-client library
            results['websocket_test'] = await self._test_websocket()
            
            # Calculate latency
            results['latency_ms'] = (datetime.now() - start_time).total_seconds() * 1000
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Integration test error: {e}")
        
        return results
    
    async def _test_websocket(self) -> bool:
        """Test WebSocket connection"""
        # Simplified test - in production would use websocket-client
        try:
            async with self.session.ws_connect(f"{self.config.base_url.replace('http', 'ws')}/ws") as ws:
                # Wait for connection message
                msg = await ws.receive_json(timeout=5)
                return msg.get('type') == 'connection'
        except:
            return False
    
    async def stress_test(self) -> Dict[str, Any]:
        """Perform stress testing with concurrent requests"""
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency_ms': 0,
            'max_latency_ms': 0,
            'errors_by_type': {}
        }
        
        tasks = []
        latencies = []
        
        # Create concurrent requests
        for i in range(self.config.stress_test_concurrent_requests):
            symbol = self.config.test_symbols[i % len(self.config.test_symbols)]
            tasks.append(self._make_request(f"/api/v1/market-data/{symbol}"))
            tasks.append(self._make_request(f"/api/v1/signals/{symbol}"))
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        for response in responses:
            results['total_requests'] += 1
            
            if isinstance(response, Exception):
                results['failed_requests'] += 1
                error_type = type(response).__name__
                results['errors_by_type'][error_type] = results['errors_by_type'].get(error_type, 0) + 1
            else:
                results['successful_requests'] += 1
                latencies.append(response['latency_ms'])
        
        if latencies:
            results['average_latency_ms'] = np.mean(latencies)
            results['max_latency_ms'] = max(latencies)
            results['p95_latency_ms'] = np.percentile(latencies, 95)
            results['p99_latency_ms'] = np.percentile(latencies, 99)
        
        return results
    
    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make a single request and measure latency"""
        start_time = datetime.now()
        
        try:
            async with self.session.get(f"{self.config.base_url}{endpoint}") as resp:
                await resp.json()
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    'endpoint': endpoint,
                    'status': resp.status,
                    'latency_ms': latency_ms
                }
        except Exception as e:
            raise e


class ProductionTestRunner:
    """Main test runner for production data tests"""
    
    def __init__(self, config: ProductionTestConfig):
        self.config = config
        self.collector = ProductionDataCollector(config)
        self.accuracy_tester = SignalAccuracyTester(config)
        self.results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all production data tests"""
        logger.info(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}Production Data Testing Framework{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # 1. Collect live market data
        logger.info(f"\n{Fore.YELLOW}1. Collecting Live Market Data...{Style.RESET_ALL}")
        live_data = await self.collector.collect_live_market_data()
        self.results['live_data_collection'] = {
            'symbols_collected': list(live_data.keys()),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # 2. Test signal accuracy with historical data
        logger.info(f"\n{Fore.YELLOW}2. Testing Signal Accuracy...{Style.RESET_ALL}")
        accuracy_results = await self._test_signal_accuracy()
        self.results['signal_accuracy'] = accuracy_results
        
        # 3. Integration tests
        logger.info(f"\n{Fore.YELLOW}3. Running Integration Tests...{Style.RESET_ALL}")
        async with ProductionIntegrationTester(self.config) as tester:
            integration_results = await tester.test_live_data_flow()
            self.results['integration'] = integration_results
            
            # 4. Stress tests
            logger.info(f"\n{Fore.YELLOW}4. Running Stress Tests...{Style.RESET_ALL}")
            stress_results = await tester.stress_test()
            self.results['stress_test'] = stress_results
        
        # 5. Generate report
        self._generate_report()
        
        return self.results
    
    async def _test_signal_accuracy(self) -> Dict[str, Any]:
        """Test signal accuracy against historical data"""
        all_results = []
        
        for symbol in self.config.test_symbols[:3]:  # Test first 3 symbols
            try:
                # Get historical data
                daily_data, hourly_data = await self.collector.collect_historical_data(
                    symbol, 
                    self.config.historical_days
                )
                
                if daily_data.empty:
                    continue
                
                # Get signals from API
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.config.base_url}/api/v1/signals/{symbol}") as resp:
                        if resp.status == 200:
                            signals = await resp.json()
                            
                            # Test each signal
                            for signal in signals[:5]:  # Test up to 5 signals per symbol
                                result = await self.accuracy_tester.test_signal_accuracy(
                                    signal, 
                                    hourly_data if not hourly_data.empty else daily_data
                                )
                                if result:
                                    all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing signals for {symbol}: {e}")
        
        # Calculate overall metrics
        metrics = self.accuracy_tester.calculate_metrics()
        
        return {
            'metrics': metrics,
            'total_signals_tested': len(all_results),
            'meets_threshold': metrics.get('accuracy_rate', 0) >= self.config.signal_accuracy_threshold
        }
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        report = {
            'test_run': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config': {
                    'symbols': self.config.test_symbols,
                    'historical_days': self.config.historical_days,
                    'accuracy_threshold': self.config.signal_accuracy_threshold,
                    'latency_threshold_ms': self.config.latency_threshold_ms
                }
            },
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # Save report
        os.makedirs(self.config.test_data_dir, exist_ok=True)
        report_path = os.path.join(
            self.config.test_data_dir,
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n{Fore.GREEN}Report saved to: {report_path}{Style.RESET_ALL}")
        
        # Print summary
        self._print_summary(report['summary'])
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'overall_status': 'PASS',
            'tests_passed': 0,
            'tests_failed': 0,
            'key_metrics': {}
        }
        
        # Check signal accuracy
        if 'signal_accuracy' in self.results:
            accuracy = self.results['signal_accuracy'].get('metrics', {}).get('accuracy_rate', 0)
            summary['key_metrics']['signal_accuracy'] = f"{accuracy:.1%}"
            
            if accuracy < self.config.signal_accuracy_threshold:
                summary['overall_status'] = 'FAIL'
                summary['tests_failed'] += 1
            else:
                summary['tests_passed'] += 1
        
        # Check integration tests
        if 'integration' in self.results:
            integration = self.results['integration']
            if integration.get('market_data_test') and integration.get('signal_generation_test'):
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
                summary['overall_status'] = 'FAIL'
            
            summary['key_metrics']['api_latency_ms'] = integration.get('latency_ms', 0)
        
        # Check stress test results
        if 'stress_test' in self.results:
            stress = self.results['stress_test']
            success_rate = stress['successful_requests'] / stress['total_requests'] if stress['total_requests'] > 0 else 0
            
            summary['key_metrics']['stress_test_success_rate'] = f"{success_rate:.1%}"
            summary['key_metrics']['p95_latency_ms'] = stress.get('p95_latency_ms', 0)
            
            if success_rate < 0.95 or stress.get('p95_latency_ms', 0) > self.config.latency_threshold_ms:
                summary['overall_status'] = 'FAIL'
                summary['tests_failed'] += 1
            else:
                summary['tests_passed'] += 1
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        logger.info(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}Test Summary{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        status_color = Fore.GREEN if summary['overall_status'] == 'PASS' else Fore.RED
        logger.info(f"Overall Status: {status_color}{summary['overall_status']}{Style.RESET_ALL}")
        logger.info(f"Tests Passed: {Fore.GREEN}{summary['tests_passed']}{Style.RESET_ALL}")
        logger.info(f"Tests Failed: {Fore.RED}{summary['tests_failed']}{Style.RESET_ALL}")
        
        logger.info(f"\n{Fore.YELLOW}Key Metrics:{Style.RESET_ALL}")
        for metric, value in summary['key_metrics'].items():
            logger.info(f"  {metric}: {value}")


async def main():
    """Main entry point for production testing"""
    # Create test configuration
    config = ProductionTestConfig(
        test_symbols=["AAPL", "GOOGL", "MSFT", "SPY", "QQQ"],
        historical_days=30,
        signal_accuracy_threshold=0.55,  # 55% accuracy required
        stress_test_concurrent_requests=20,
        use_real_time_data=True
    )
    
    # Run tests
    runner = ProductionTestRunner(config)
    results = await runner.run_all_tests()
    
    # Return exit code based on results
    if results.get('summary', {}).get('overall_status') == 'PASS':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
