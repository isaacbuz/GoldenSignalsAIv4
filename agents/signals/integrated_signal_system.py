"""
🚀 Integrated Signal System
Combines precise options signals with arbitrage detection for comprehensive trading opportunities
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from .arbitrage_signals import (
    ArbitrageSignal,
    ArbitrageSignalManager,
    RiskArbitrageDetector,
    SpatialArbitrageDetector,
    StatisticalArbitrageDetector,
)
from .precise_options_signals import PreciseOptionsSignal, PreciseSignalGenerator


class IntegratedSignalSystem:
    """Master system combining all signal types"""

    def __init__(self):
        # Initialize all signal generators
        self.options_generator = PreciseSignalGenerator()
        self.arbitrage_manager = ArbitrageSignalManager()

        # Signal storage
        self.active_signals = {
            'options': [],
            'arbitrage': [],
            'combined': []  # Options + Arbitrage hybrid signals
        }

        # Configuration
        self.config = {
            'scan_interval': 60,  # seconds
            'max_signals_per_type': 10,
            'risk_limits': {
                'max_positions': 5,
                'max_risk_per_trade': 1000,
                'daily_loss_limit': 2500
            }
        }

    async def scan_all_markets(self, symbols: List[str]) -> Dict[str, List]:
        """Comprehensive market scan for all signal types"""

        print("🔍 Integrated Signal System - Full Market Scan")
        print("="*60)
        print(f"Scanning {len(symbols)} symbols at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Parallel scanning
        tasks = [
            self._scan_options_signals(symbols),
            self._scan_arbitrage_signals(symbols),
            self._scan_combined_signals(symbols)
        ]

        results = await asyncio.gather(*tasks)

        # Update active signals
        self.active_signals['options'] = results[0]
        self.active_signals['arbitrage'] = results[1]
        self.active_signals['combined'] = results[2]

        # Generate summary
        self._print_signal_summary()

        return self.active_signals

    async def _scan_options_signals(self, symbols: List[str]) -> List[PreciseOptionsSignal]:
        """Scan for precise options signals"""
        signals = []

        for symbol in symbols:
            try:
                signal = self.options_generator.analyze_symbol(symbol)
                if signal and signal.confidence > 70:  # Min 70% confidence
                    signals.append(signal)
            except Exception as e:
                print(f"Error scanning {symbol} for options: {e}")

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:self.config['max_signals_per_type']]

    async def _scan_arbitrage_signals(self, symbols: List[str]) -> List[ArbitrageSignal]:
        """Scan for arbitrage opportunities"""
        signals = await self.arbitrage_manager.scan_all_opportunities(symbols)
        return signals[:self.config['max_signals_per_type']]

    async def _scan_combined_signals(self, symbols: List[str]) -> List[Dict]:
        """Find combined options + arbitrage opportunities"""
        combined_signals = []

        # Look for options trades that also have arbitrage potential
        for symbol in symbols:
            combined = await self._find_combined_opportunity(symbol)
            if combined:
                combined_signals.append(combined)

        return combined_signals

    async def _find_combined_opportunity(self, symbol: str) -> Optional[Dict]:
        """Find opportunities that combine options and arbitrage"""

        # Example: Options trade on a stock with spatial arbitrage
        # This creates a "supercharged" signal

        # Check if symbol has both types of opportunities
        has_options = any(s.symbol == symbol for s in self.active_signals.get('options', []))
        has_arbitrage = any(s.primary_asset == symbol for s in self.active_signals.get('arbitrage', []))

        if has_options and has_arbitrage:
            return {
                'type': 'COMBINED',
                'symbol': symbol,
                'strategy': 'Options + Arbitrage',
                'description': f'Execute options trade on {symbol} while capturing arbitrage spread',
                'estimated_return': '15-25%',
                'complexity': 'ADVANCED',
                'capital_required': 15000
            }

        # Check for volatility arbitrage with options
        if symbol in ['TSLA', 'NVDA', 'GME']:  # High vol stocks
            current_iv = 0.45  # Mock implied volatility
            if current_iv > 0.40:
                return {
                    'type': 'VOL_ARB_OPTIONS',
                    'symbol': symbol,
                    'strategy': 'Volatility Arbitrage + Directional Options',
                    'description': f'Sell volatility on {symbol} while buying directional options',
                    'trades': [
                        f'1. Sell {symbol} straddle (capture IV premium)',
                        f'2. Buy OTM calls/puts based on trend',
                        '3. Delta hedge the straddle',
                        '4. Profit from vol crush + directional move'
                    ],
                    'estimated_return': '20-30%',
                    'complexity': 'EXPERT',
                    'capital_required': 25000
                }

        return None

    def _print_signal_summary(self):
        """Print summary of all active signals"""

        total_signals = sum(len(signals) for signals in self.active_signals.values())

        print("\n📊 SIGNAL SUMMARY")
        print("="*60)
        print(f"Total Active Signals: {total_signals}")
        print()

        # Options signals
        if self.active_signals['options']:
            print("🎯 OPTIONS SIGNALS:")
            for signal in self.active_signals['options'][:3]:  # Top 3
                emoji = "🟢" if signal.signal_type == "BUY_CALL" else "🔴"
                print(f"   {emoji} {signal.symbol} - {signal.signal_type} ({signal.confidence}%)")
                print(f"      Strike: ${signal.strike_price}, Entry: ${signal.entry_trigger}")
                print(f"      Targets: ${signal.targets[0]['price']}, ${signal.targets[1]['price']}")
                print()

        # Arbitrage signals
        if self.active_signals['arbitrage']:
            print("💎 ARBITRAGE SIGNALS:")
            for signal in self.active_signals['arbitrage'][:3]:  # Top 3
                print(f"   {signal.arb_type} - {signal.primary_asset}")
                print(f"      Spread: {signal.spread_pct:.2f}%, Profit: ${signal.estimated_profit:.0f}")
                print(f"      Risk: {signal.risk_level}, Hold: {signal.holding_period}")
                print()

        # Combined signals
        if self.active_signals['combined']:
            print("🚀 COMBINED SIGNALS (Advanced):")
            for signal in self.active_signals['combined']:
                print(f"   {signal['symbol']} - {signal['strategy']}")
                print(f"      Return: {signal['estimated_return']}, Capital: ${signal['capital_required']:,}")
                print()

    def get_top_opportunities(self,
                            risk_tolerance: str = 'MEDIUM',
                            capital: float = 10000,
                            types: List[str] = None) -> List[Dict]:
        """Get top opportunities based on criteria"""

        opportunities = []

        # Filter by type
        if types is None:
            types = ['options', 'arbitrage', 'combined']

        # Options opportunities
        if 'options' in types:
            for signal in self.active_signals.get('options', []):
                if self._matches_criteria(signal, risk_tolerance, capital):
                    opportunities.append({
                        'type': 'OPTIONS',
                        'signal': signal,
                        'score': signal.confidence * signal.risk_reward_ratio
                    })

        # Arbitrage opportunities
        if 'arbitrage' in types:
            for signal in self.active_signals.get('arbitrage', []):
                risk_match = (
                    (risk_tolerance == 'LOW' and signal.risk_level == 'LOW') or
                    (risk_tolerance == 'MEDIUM' and signal.risk_level in ['LOW', 'MEDIUM']) or
                    (risk_tolerance == 'HIGH')
                )
                if risk_match and signal.capital_required <= capital:
                    opportunities.append({
                        'type': 'ARBITRAGE',
                        'signal': signal,
                        'score': signal.confidence * (signal.estimated_profit / signal.capital_required)
                    })

        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities[:5]  # Top 5

    def _matches_criteria(self, signal: PreciseOptionsSignal,
                         risk_tolerance: str, capital: float) -> bool:
        """Check if signal matches user criteria"""

        # Risk check
        if risk_tolerance == 'LOW' and signal.stop_loss_pct > 2:
            return False
        elif risk_tolerance == 'MEDIUM' and signal.stop_loss_pct > 5:
            return False

        # Capital check
        position_cost = signal.position_size * signal.max_premium * 100
        if position_cost > capital * 0.2:  # Max 20% per position
            return False

        return True

    def generate_execution_plan(self, opportunities: List[Dict]) -> Dict:
        """Generate detailed execution plan for selected opportunities"""

        plan = {
            'generated_at': datetime.now().isoformat(),
            'total_capital_required': 0,
            'estimated_returns': {
                'best_case': 0,
                'expected': 0,
                'worst_case': 0
            },
            'trades': []
        }

        for opp in opportunities:
            signal = opp['signal']

            if opp['type'] == 'OPTIONS':
                trade = {
                    'type': 'OPTIONS',
                    'symbol': signal.symbol,
                    'action': signal.signal_type,
                    'contract': f"${signal.strike_price} {signal.expiration_date}",
                    'entry': {
                        'trigger': signal.entry_trigger,
                        'window': signal.entry_window,
                        'max_premium': signal.max_premium
                    },
                    'exit': {
                        'stop_loss': signal.stop_loss,
                        'targets': signal.targets,
                        'rules': signal.exit_rules[:3]
                    },
                    'capital': signal.position_size * signal.max_premium * 100,
                    'max_risk': signal.max_risk_dollars
                }
            else:  # ARBITRAGE
                trade = {
                    'type': 'ARBITRAGE',
                    'subtype': signal.arb_type,
                    'symbol': signal.primary_asset,
                    'venues': {
                        'buy': signal.buy_venue,
                        'sell': signal.sell_venue
                    },
                    'spread': f"{signal.spread_pct:.2f}%",
                    'execution': signal.execution_steps[:3],
                    'capital': signal.capital_required,
                    'estimated_profit': signal.estimated_profit
                }

            plan['trades'].append(trade)
            plan['total_capital_required'] += trade['capital']

        # Calculate returns
        for trade in plan['trades']:
            if trade['type'] == 'OPTIONS':
                # Rough estimates
                plan['estimated_returns']['best_case'] += trade['capital'] * 1.5
                plan['estimated_returns']['expected'] += trade['capital'] * 0.3
                plan['estimated_returns']['worst_case'] -= trade['max_risk']
            else:
                profit = trade['estimated_profit']
                plan['estimated_returns']['best_case'] += profit
                plan['estimated_returns']['expected'] += profit * 0.7
                plan['estimated_returns']['worst_case'] += profit * 0.2

        return plan

    async def execute_paper_trades(self, plan: Dict) -> Dict:
        """Execute trades in paper trading mode"""

        results = {
            'executed_at': datetime.now().isoformat(),
            'trades': []
        }

        for trade in plan['trades']:
            # Simulate execution
            await asyncio.sleep(0.1)  # Simulate latency

            result = {
                'trade': trade,
                'status': 'EXECUTED',
                'fill_price': trade.get('entry', {}).get('trigger', 100),
                'timestamp': datetime.now().isoformat()
            }

            results['trades'].append(result)

            print(f"✅ Executed: {trade['type']} - {trade.get('symbol', 'N/A')}")

        return results

async def demonstrate_integrated_system():
    """Demonstrate the integrated signal system"""

    system = IntegratedSignalSystem()

    # Define symbols to scan
    symbols = [
        # Stocks
        'TSLA', 'AAPL', 'NVDA', 'AMD', 'SPY', 'QQQ',
        # For arbitrage
        'MSFT', 'GOOGL', 'META', 'AMZN',
        # Crypto (for arbitrage)
        'BTC-USD', 'ETH-USD'
    ]

    print("🚀 GoldenSignalsAI - Integrated Signal System Demo")
    print("="*60)
    print()

    # Run comprehensive scan
    signals = await system.scan_all_markets(symbols)

    # Get top opportunities for different profiles
    print("\n💼 TOP OPPORTUNITIES BY PROFILE")
    print("="*60)

    # Conservative investor
    print("\n🟢 CONSERVATIVE (Low Risk, $10K capital):")
    conservative_opps = system.get_top_opportunities('LOW', 10000)
    for i, opp in enumerate(conservative_opps[:3], 1):
        signal = opp['signal']
        print(f"{i}. {opp['type']} - {getattr(signal, 'symbol', getattr(signal, 'primary_asset', 'N/A'))}")
        print(f"   Score: {opp['score']:.1f}")

    # Moderate investor
    print("\n🟡 MODERATE (Medium Risk, $25K capital):")
    moderate_opps = system.get_top_opportunities('MEDIUM', 25000)
    for i, opp in enumerate(moderate_opps[:3], 1):
        signal = opp['signal']
        print(f"{i}. {opp['type']} - {getattr(signal, 'symbol', getattr(signal, 'primary_asset', 'N/A'))}")
        print(f"   Score: {opp['score']:.1f}")

    # Aggressive investor
    print("\n🔴 AGGRESSIVE (High Risk, $50K capital):")
    aggressive_opps = system.get_top_opportunities('HIGH', 50000)
    for i, opp in enumerate(aggressive_opps[:3], 1):
        signal = opp['signal']
        print(f"{i}. {opp['type']} - {getattr(signal, 'symbol', getattr(signal, 'primary_asset', 'N/A'))}")
        print(f"   Score: {opp['score']:.1f}")

    # Generate execution plan for moderate investor
    print("\n📋 EXECUTION PLAN (Moderate Profile)")
    print("="*60)

    plan = system.generate_execution_plan(moderate_opps[:3])

    print(f"Total Capital Required: ${plan['total_capital_required']:,.0f}")
    print(f"Expected Return: ${plan['estimated_returns']['expected']:,.0f}")
    print(f"Best Case: ${plan['estimated_returns']['best_case']:,.0f}")
    print(f"Worst Case: ${plan['estimated_returns']['worst_case']:,.0f}")

    print("\nTrades to Execute:")
    for i, trade in enumerate(plan['trades'], 1):
        print(f"\n{i}. {trade['type']} - {trade.get('symbol', 'Multi-asset')}")
        if trade['type'] == 'OPTIONS':
            print(f"   {trade['action']} {trade['contract']}")
            print(f"   Entry: ${trade['entry']['trigger']}")
        else:
            print(f"   {trade['subtype']} - Spread: {trade['spread']}")

    # Execute paper trades
    print("\n🚀 EXECUTING PAPER TRADES...")
    print("="*60)

    results = await system.execute_paper_trades(plan)

    print(f"\n✅ Successfully executed {len(results['trades'])} trades")

    # Save results
    with open('integrated_signals_demo.json', 'w') as f:
        json.dump({
            'signals': {
                'options': [s.__dict__ for s in signals.get('options', [])],
                'arbitrage': [s.__dict__ for s in signals.get('arbitrage', [])]
            },
            'execution_plan': plan,
            'results': results
        }, f, indent=2, default=str)

    print("\n📄 Results saved to integrated_signals_demo.json")

if __name__ == "__main__":
    asyncio.run(demonstrate_integrated_system())
