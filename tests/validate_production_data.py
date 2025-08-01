#!/usr/bin/env python3
"""
Production Data Validator for GoldenSignalsAI V2
Validates system accuracy with real market data
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import time

import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDataValidator:
    """Validates production data and signal accuracy"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_symbols = ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ"]
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'api_tests': {},
            'data_accuracy': {},
            'signal_validation': {},
            'performance_metrics': {}
        }

    async def run_validation(self):
        """Run all validation tests"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}GoldenSignalsAI Production Data Validation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

        # 1. Validate API endpoints
        print(f"{Fore.YELLOW}1. Validating API Endpoints...{Style.RESET_ALL}")
        await self.validate_api_endpoints()

        # 2. Validate market data accuracy
        print(f"\n{Fore.YELLOW}2. Validating Market Data Accuracy...{Style.RESET_ALL}")
        await self.validate_market_data()

        # 3. Validate signal generation
        print(f"\n{Fore.YELLOW}3. Validating Signal Generation...{Style.RESET_ALL}")
        await self.validate_signals()

        # 4. Performance tests
        print(f"\n{Fore.YELLOW}4. Running Performance Tests...{Style.RESET_ALL}")
        await self.test_performance()

        # 5. Generate report
        self.generate_report()

    async def validate_api_endpoints(self):
        """Test all API endpoints"""
        endpoints = [
            ("/", "Backend Health"),
            ("/api/v1/signals", "Signals List"),
            ("/api/v1/market-data/SPY", "Market Data"),
            ("/api/v1/market-data/SPY/historical?period=1d&interval=5m", "Historical Data"),
            ("/api/v1/signals/SPY/insights", "Signal Insights"),
            ("/api/v1/market/opportunities", "Market Opportunities"),
            ("/api/v1/signals/precise-options?symbol=SPY&timeframe=15m", "Options Signals")
        ]

        async with aiohttp.ClientSession() as session:
            for endpoint, name in endpoints:
                try:
                    start_time = time.time()
                    async with session.get(f"{self.base_url}{endpoint}") as resp:
                        latency = (time.time() - start_time) * 1000

                        if resp.status == 200:
                            data = await resp.json() if endpoint != "/" else await resp.text()
                            self.results['api_tests'][name] = {
                                'status': 'PASS',
                                'latency_ms': round(latency, 2),
                                'response_size': len(str(data))
                            }
                            self.results['tests_passed'] += 1
                            print(f"  ✅ {name}: {Fore.GREEN}PASS{Style.RESET_ALL} ({latency:.0f}ms)")
                        else:
                            self.results['api_tests'][name] = {
                                'status': 'FAIL',
                                'error': f'HTTP {resp.status}'
                            }
                            self.results['tests_failed'] += 1
                            print(f"  ❌ {name}: {Fore.RED}FAIL{Style.RESET_ALL} (HTTP {resp.status})")

                except Exception as e:
                    self.results['api_tests'][name] = {
                        'status': 'FAIL',
                        'error': str(e)
                    }
                    self.results['tests_failed'] += 1
                    print(f"  ❌ {name}: {Fore.RED}FAIL{Style.RESET_ALL} ({str(e)[:50]}...)")

    async def validate_market_data(self):
        """Validate market data accuracy against yfinance"""
        for symbol in self.test_symbols[:3]:  # Test first 3 symbols
            try:
                # Get data from API
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/v1/market-data/{symbol}") as resp:
                        if resp.status != 200:
                            continue
                        api_data = await resp.json()

                # Get data from yfinance
                ticker = yf.Ticker(symbol)
                yf_data = ticker.history(period="1d", interval="1m")

                if not yf_data.empty:
                    latest_yf = yf_data.iloc[-1]

                    # Compare prices
                    api_price = api_data['price']
                    yf_price = float(latest_yf['Close'])
                    price_diff = abs(api_price - yf_price)
                    price_diff_pct = (price_diff / yf_price) * 100 if yf_price > 0 else 0

                    # Accuracy threshold: within 1%
                    is_accurate = price_diff_pct < 1.0

                    self.results['data_accuracy'][symbol] = {
                        'api_price': api_price,
                        'yf_price': yf_price,
                        'difference_pct': round(price_diff_pct, 2),
                        'accurate': is_accurate
                    }

                    if is_accurate:
                        self.results['tests_passed'] += 1
                        print(f"  ✅ {symbol}: {Fore.GREEN}ACCURATE{Style.RESET_ALL} (diff: {price_diff_pct:.2f}%)")
                    else:
                        self.results['tests_failed'] += 1
                        print(f"  ❌ {symbol}: {Fore.RED}INACCURATE{Style.RESET_ALL} (diff: {price_diff_pct:.2f}%)")

            except Exception as e:
                logger.error(f"Error validating {symbol}: {e}")
                self.results['tests_failed'] += 1

    async def validate_signals(self):
        """Validate signal generation logic"""
        validation_results = []

        async with aiohttp.ClientSession() as session:
            for symbol in self.test_symbols[:3]:
                try:
                    # Get signals
                    async with session.get(f"{self.base_url}/api/v1/signals/{symbol}") as resp:
                        if resp.status != 200:
                            continue
                        signals = await resp.json()

                    # Get historical data
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(period="5d", interval="1h")

                    if signals and not hist_data.empty:
                        # Validate signal fields
                        signal = signals[0] if isinstance(signals, list) else signals

                        required_fields = ['id', 'symbol', 'action', 'confidence', 'price', 'timestamp']
                        has_all_fields = all(field in signal for field in required_fields)

                        # Validate signal logic
                        closes = hist_data['Close'].values
                        current_price = closes[-1]
                        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)

                        # Simple validation: BUY below SMA, SELL above SMA
                        if signal['action'] == 'BUY':
                            logic_valid = current_price < sma_20 * 1.02  # 2% buffer
                        elif signal['action'] == 'SELL':
                            logic_valid = current_price > sma_20 * 0.98  # 2% buffer
                        else:
                            logic_valid = True  # HOLD is always valid

                        validation_results.append({
                            'symbol': symbol,
                            'has_all_fields': has_all_fields,
                            'logic_valid': logic_valid,
                            'confidence_valid': 0 <= signal.get('confidence', 0) <= 1
                        })

                        is_valid = has_all_fields and logic_valid

                        if is_valid:
                            self.results['tests_passed'] += 1
                            print(f"  ✅ {symbol}: {Fore.GREEN}VALID{Style.RESET_ALL} ({signal['action']} @ ${signal['price']:.2f})")
                        else:
                            self.results['tests_failed'] += 1
                            print(f"  ❌ {symbol}: {Fore.RED}INVALID{Style.RESET_ALL} (fields: {has_all_fields}, logic: {logic_valid})")

                except Exception as e:
                    logger.error(f"Error validating signals for {symbol}: {e}")
                    self.results['tests_failed'] += 1

        self.results['signal_validation'] = validation_results

    async def test_performance(self):
        """Test system performance"""
        latencies = []

        async with aiohttp.ClientSession() as session:
            # Test 10 concurrent requests
            tasks = []
            for i in range(10):
                symbol = self.test_symbols[i % len(self.test_symbols)]
                tasks.append(self._measure_latency(session, f"/api/v1/market-data/{symbol}"))

            results = await asyncio.gather(*tasks)
            latencies = [r for r in results if r is not None]

        if latencies:
            self.results['performance_metrics'] = {
                'avg_latency_ms': round(np.mean(latencies), 2),
                'max_latency_ms': round(max(latencies), 2),
                'min_latency_ms': round(min(latencies), 2),
                'p95_latency_ms': round(np.percentile(latencies, 95), 2)
            }

            # Performance pass if avg latency < 500ms
            if self.results['performance_metrics']['avg_latency_ms'] < 500:
                self.results['tests_passed'] += 1
                print(f"  ✅ Performance: {Fore.GREEN}PASS{Style.RESET_ALL} (avg: {self.results['performance_metrics']['avg_latency_ms']}ms)")
            else:
                self.results['tests_failed'] += 1
                print(f"  ❌ Performance: {Fore.RED}FAIL{Style.RESET_ALL} (avg: {self.results['performance_metrics']['avg_latency_ms']}ms)")

    async def _measure_latency(self, session, endpoint):
        """Measure request latency"""
        try:
            start_time = time.time()
            async with session.get(f"{self.base_url}{endpoint}") as resp:
                await resp.json()
                return (time.time() - start_time) * 1000
        except:
            return None

    def generate_report(self):
        """Generate validation report"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Validation Summary{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        pass_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0

        status = "PASS" if pass_rate >= 80 else "FAIL"
        status_color = Fore.GREEN if status == "PASS" else Fore.RED

        print(f"Overall Status: {status_color}{status}{Style.RESET_ALL}")
        print(f"Tests Passed: {Fore.GREEN}{self.results['tests_passed']}{Style.RESET_ALL}")
        print(f"Tests Failed: {Fore.RED}{self.results['tests_failed']}{Style.RESET_ALL}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if 'performance_metrics' in self.results:
            print(f"\nPerformance Metrics:")
            for metric, value in self.results['performance_metrics'].items():
                print(f"  {metric}: {value}")

        # Save detailed report
        report_path = f"test_data/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")

        return status == "PASS"


async def main():
    """Main entry point"""
    validator = ProductionDataValidator()
    success = await validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
