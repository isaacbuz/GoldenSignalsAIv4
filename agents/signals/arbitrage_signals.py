"""
🎯 Arbitrage Signal Agent
Identifies arbitrage opportunities across stocks, options, crypto, and other markets
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class ArbitrageSignal:
    """Arbitrage opportunity signal"""
    signal_id: str
    timestamp: datetime

    # Arbitrage Type
    arb_type: str  # SPATIAL, STATISTICAL, RISK, MERGER, OPTIONS
    confidence: float

    # Assets
    primary_asset: str
    secondary_asset: Optional[str]  # For pairs trading
    asset_type: str  # STOCK, OPTION, CRYPTO, FOREX

    # Pricing
    buy_venue: str
    buy_price: float
    sell_venue: str
    sell_price: float

    # Opportunity
    spread: float  # Dollar amount
    spread_pct: float  # Percentage
    estimated_profit: float

    # Execution
    entry_window: str  # "Immediate", "1-5 minutes", etc.
    holding_period: str  # "Seconds", "Minutes", "Hours"
    capital_required: float

    # Risk
    risk_level: str  # LOW, MEDIUM, HIGH
    risk_factors: List[str]
    execution_complexity: str  # SIMPLE, MODERATE, COMPLEX

    # Instructions
    execution_steps: List[str]
    prerequisites: List[str]

    # Event-based (for risk arb)
    catalyst_event: Optional[str]
    event_date: Optional[datetime]

class ArbitrageDetector(ABC):
    """Base class for arbitrage detection"""

    @abstractmethod
    async def detect_opportunities(self, assets: List[str]) -> List[ArbitrageSignal]:
        pass

class SpatialArbitrageDetector(ArbitrageDetector):
    """Detect price differences across exchanges/venues"""

    def __init__(self):
        self.venues = {
            'stocks': ['NYSE', 'NASDAQ', 'ARCA', 'IEX'],
            'crypto': ['Binance', 'Coinbase', 'Kraken', 'Gemini'],
            'forex': ['Oanda', 'IG', 'CMC', 'Saxo']
        }

    async def detect_opportunities(self, assets: List[str]) -> List[ArbitrageSignal]:
        """Find spatial arbitrage opportunities"""
        signals = []

        for asset in assets:
            # Get prices from multiple venues
            prices = await self._fetch_multi_venue_prices(asset)

            if len(prices) < 2:
                continue

            # Find best bid/ask across venues
            sorted_bids = sorted(prices.items(), key=lambda x: x[1]['bid'], reverse=True)
            sorted_asks = sorted(prices.items(), key=lambda x: x[1]['ask'])

            best_bid_venue, best_bid_data = sorted_bids[0]
            best_ask_venue, best_ask_data = sorted_asks[0]

            # Calculate spread
            spread = best_bid_data['bid'] - best_ask_data['ask']
            spread_pct = (spread / best_ask_data['ask']) * 100

            # Only signal if spread is profitable after costs
            if spread_pct > 0.1:  # 0.1% minimum
                signal = ArbitrageSignal(
                    signal_id=f"SPATIAL_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),

                    arb_type="SPATIAL",
                    confidence=min(spread_pct * 20, 95),  # Higher spread = higher confidence

                    primary_asset=asset,
                    secondary_asset=None,
                    asset_type=self._determine_asset_type(asset),

                    buy_venue=best_ask_venue,
                    buy_price=best_ask_data['ask'],
                    sell_venue=best_bid_venue,
                    sell_price=best_bid_data['bid'],

                    spread=spread,
                    spread_pct=spread_pct,
                    estimated_profit=self._calculate_profit(spread, 1000),  # Per $1000

                    entry_window="Immediate - 30 seconds",
                    holding_period="Seconds",
                    capital_required=10000,  # Minimum for meaningful profit

                    risk_level="LOW" if spread_pct < 0.5 else "MEDIUM",
                    risk_factors=self._identify_spatial_risks(asset, spread_pct),
                    execution_complexity="MODERATE",

                    execution_steps=[
                        f"1. Buy {asset} at {best_ask_venue} for ${best_ask_data['ask']:.2f}",
                        f"2. Simultaneously sell at {best_bid_venue} for ${best_bid_data['bid']:.2f}",
                        f"3. Net profit: ${spread:.2f} per unit ({spread_pct:.2f}%)",
                        "4. Repeat until spread closes"
                    ],

                    prerequisites=[
                        f"Account access to both {best_ask_venue} and {best_bid_venue}",
                        "Fast execution capability (< 1 second)",
                        "Sufficient capital on both venues",
                        "Low-latency data feed"
                    ],

                    catalyst_event=None,
                    event_date=None
                )

                signals.append(signal)

        return signals

    async def _fetch_multi_venue_prices(self, asset: str) -> Dict[str, Dict[str, float]]:
        """Fetch prices from multiple venues (mock for demo)"""
        # In production, this would connect to real APIs
        base_price = 100  # Mock base

        # Simulate small price differences
        prices = {}
        for venue in ['NYSE', 'NASDAQ', 'ARCA']:
            variation = np.random.uniform(-0.5, 0.5)
            bid = base_price + variation - 0.01
            ask = base_price + variation + 0.01
            prices[venue] = {'bid': bid, 'ask': ask}

        return prices

    def _determine_asset_type(self, asset: str) -> str:
        if asset.endswith('-USD') or asset in ['BTC', 'ETH', 'SOL']:
            return 'CRYPTO'
        elif '/' in asset:
            return 'FOREX'
        else:
            return 'STOCK'

    def _calculate_profit(self, spread: float, capital: float) -> float:
        """Calculate estimated profit after costs"""
        units = capital / 100  # Assuming $100 per unit
        gross_profit = spread * units

        # Subtract estimated costs (0.05% per side)
        costs = capital * 0.001
        return gross_profit - costs

    def _identify_spatial_risks(self, asset: str, spread_pct: float) -> List[str]:
        """Identify risks for spatial arbitrage"""
        risks = []

        if spread_pct > 1.0:
            risks.append("Large spread may indicate liquidity issues")

        if self._determine_asset_type(asset) == 'CRYPTO':
            risks.append("Network congestion may delay transfers")
            risks.append("Exchange withdrawal limits may apply")

        risks.extend([
            "Execution delays could eliminate profit",
            "Regulatory differences between venues",
            "Hidden fees may reduce profitability"
        ])

        return risks

class StatisticalArbitrageDetector(ArbitrageDetector):
    """Detect statistical mispricings using quantitative models"""

    def __init__(self):
        self.lookback_period = 60  # days
        self.z_score_threshold = 2.0

    async def detect_opportunities(self, assets: List[str]) -> List[ArbitrageSignal]:
        """Find statistical arbitrage opportunities"""
        signals = []

        # Pairs trading opportunities
        pairs = self._generate_pairs(assets)

        for asset1, asset2 in pairs:
            signal = await self._analyze_pair(asset1, asset2)
            if signal:
                signals.append(signal)

        # Mean reversion opportunities
        for asset in assets:
            signal = await self._analyze_mean_reversion(asset)
            if signal:
                signals.append(signal)

        return signals

    async def _analyze_pair(self, asset1: str, asset2: str) -> Optional[ArbitrageSignal]:
        """Analyze a pair for statistical arbitrage"""
        # Fetch historical data
        data1 = await self._fetch_historical_data(asset1)
        data2 = await self._fetch_historical_data(asset2)

        if data1 is None or data2 is None:
            return None

        # Calculate spread
        ratio = data1['Close'] / data2['Close']
        mean_ratio = ratio.rolling(20).mean()
        std_ratio = ratio.rolling(20).std()
        z_score = (ratio.iloc[-1] - mean_ratio.iloc[-1]) / std_ratio.iloc[-1]

        # Check for significant deviation
        if abs(z_score) > self.z_score_threshold:
            # Determine direction
            if z_score > self.z_score_threshold:
                # Asset1 overvalued relative to asset2
                action = f"Short {asset1}, Long {asset2}"
                buy_asset = asset2
                sell_asset = asset1
            else:
                # Asset1 undervalued relative to asset2
                action = f"Long {asset1}, Short {asset2}"
                buy_asset = asset1
                sell_asset = asset2

            return ArbitrageSignal(
                signal_id=f"STAT_PAIR_{asset1}_{asset2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),

                arb_type="STATISTICAL",
                confidence=min(abs(z_score) * 30, 90),

                primary_asset=asset1,
                secondary_asset=asset2,
                asset_type="STOCK",

                buy_venue="Primary Exchange",
                buy_price=data1['Close'].iloc[-1] if buy_asset == asset1 else data2['Close'].iloc[-1],
                sell_venue="Primary Exchange",
                sell_price=data2['Close'].iloc[-1] if sell_asset == asset2 else data1['Close'].iloc[-1],

                spread=abs(ratio.iloc[-1] - mean_ratio.iloc[-1]),
                spread_pct=abs(z_score) * std_ratio.iloc[-1] / mean_ratio.iloc[-1] * 100,
                estimated_profit=self._estimate_stat_arb_profit(z_score, std_ratio.iloc[-1]),

                entry_window="Next 1-2 hours",
                holding_period="2-5 days",
                capital_required=20000,

                risk_level="MEDIUM",
                risk_factors=[
                    "Correlation breakdown risk",
                    "Mean reversion may not occur",
                    "Requires margin for short positions",
                    "Market regime change risk"
                ],
                execution_complexity="COMPLEX",

                execution_steps=[
                    f"1. {action}",
                    f"2. Z-score: {z_score:.2f} (threshold: {self.z_score_threshold})",
                    f"3. Hold until z-score returns to [-1, 1] range",
                    f"4. Exit both positions simultaneously",
                    f"5. Expected mean reversion in 2-5 days"
                ],

                prerequisites=[
                    "Margin account for short selling",
                    "Ability to trade both assets",
                    "Risk management system",
                    "Statistical monitoring tools"
                ],

                catalyst_event=None,
                event_date=None
            )

        return None

    async def _analyze_mean_reversion(self, asset: str) -> Optional[ArbitrageSignal]:
        """Analyze single asset for mean reversion"""
        data = await self._fetch_historical_data(asset)
        if data is None:
            return None

        # Calculate indicators
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        current_price = data['Close'].iloc[-1]

        # Calculate deviation
        deviation = (current_price - sma_20.iloc[-1]) / std_20.iloc[-1]

        if abs(deviation) > 2.5:  # 2.5 standard deviations
            if deviation > 0:
                signal_type = "OVERBOUGHT"
                action = "SHORT"
                target = sma_20.iloc[-1] + std_20.iloc[-1]
            else:
                signal_type = "OVERSOLD"
                action = "LONG"
                target = sma_20.iloc[-1] - std_20.iloc[-1]

            return ArbitrageSignal(
                signal_id=f"STAT_MEAN_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),

                arb_type="STATISTICAL",
                confidence=min(abs(deviation) * 25, 85),

                primary_asset=asset,
                secondary_asset=None,
                asset_type="STOCK",

                buy_venue="Primary Exchange" if action == "LONG" else "N/A",
                buy_price=current_price if action == "LONG" else 0,
                sell_venue="Primary Exchange" if action == "SHORT" else "N/A",
                sell_price=current_price if action == "SHORT" else 0,

                spread=abs(current_price - sma_20.iloc[-1]),
                spread_pct=abs(deviation) * std_20.iloc[-1] / sma_20.iloc[-1] * 100,
                estimated_profit=(abs(current_price - target) / current_price) * 100,

                entry_window="Next 30 minutes",
                holding_period="1-3 days",
                capital_required=10000,

                risk_level="MEDIUM" if abs(deviation) < 3 else "HIGH",
                risk_factors=[
                    f"Asset is {signal_type.lower()}",
                    "Trend continuation risk",
                    "False signal in strong trends",
                    "Volatility expansion risk"
                ],
                execution_complexity="SIMPLE",

                execution_steps=[
                    f"1. {action} {asset} at current price ${current_price:.2f}",
                    f"2. Target: ${target:.2f} (20-day SMA ± 1 std)",
                    f"3. Stop loss: ${current_price * (1.03 if action == 'SHORT' else 0.97):.2f}",
                    f"4. Deviation: {deviation:.2f} standard deviations",
                    "5. Exit on return to mean or stop loss"
                ],

                prerequisites=[
                    "Sufficient capital",
                    "Risk management plan",
                    "Ability to monitor position",
                    "Understanding of mean reversion"
                ],

                catalyst_event=None,
                event_date=None
            )

        return None

    def _generate_pairs(self, assets: List[str]) -> List[Tuple[str, str]]:
        """Generate potential pairs for analysis"""
        # In production, use correlation analysis
        # For demo, create some logical pairs
        pairs = []

        # Tech pairs
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
        for i in range(len(tech_stocks)):
            for j in range(i+1, len(tech_stocks)):
                if tech_stocks[i] in assets and tech_stocks[j] in assets:
                    pairs.append((tech_stocks[i], tech_stocks[j]))

        return pairs[:3]  # Limit for demo

    async def _fetch_historical_data(self, asset: str) -> Optional[pd.DataFrame]:
        """Fetch historical data (mock for demo)"""
        # In production, use real data
        dates = pd.date_range(end=datetime.now(), periods=self.lookback_period, freq='D')

        # Simulate price data
        np.random.seed(hash(asset) % 1000)  # Consistent per asset
        returns = np.random.normal(0.0005, 0.02, self.lookback_period)
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'Close': prices,
            'Volume': np.random.lognormal(17, 0.5, self.lookback_period)
        }, index=dates)

    def _estimate_stat_arb_profit(self, z_score: float, std: float) -> float:
        """Estimate profit from statistical arbitrage"""
        # Expected move back to mean
        expected_move = (abs(z_score) - 1) * std
        return expected_move * 100  # Per $100 invested

class RiskArbitrageDetector(ArbitrageDetector):
    """Detect event-driven arbitrage opportunities"""

    def __init__(self):
        self.events_calendar = {
            'TSLA': {
                'event': 'Robotaxi Launch',
                'date': datetime(2025, 6, 12),
                'impact': 'HIGH'
            },
            'AAPL': {
                'event': 'WWDC Keynote',
                'date': datetime(2025, 6, 10),
                'impact': 'MEDIUM'
            }
        }

    async def detect_opportunities(self, assets: List[str]) -> List[ArbitrageSignal]:
        """Find risk arbitrage opportunities around events"""
        signals = []

        for asset in assets:
            if asset in self.events_calendar:
                event_data = self.events_calendar[asset]

                # Check if event is upcoming
                days_to_event = (event_data['date'] - datetime.now()).days

                if 0 <= days_to_event <= 5:  # Within 5 days
                    signal = await self._analyze_event_arbitrage(asset, event_data)
                    if signal:
                        signals.append(signal)

        # Check for volatility arbitrage
        for asset in assets:
            vol_signal = await self._analyze_volatility_arbitrage(asset)
            if vol_signal:
                signals.append(vol_signal)

        return signals

    async def _analyze_event_arbitrage(self, asset: str, event_data: Dict) -> Optional[ArbitrageSignal]:
        """Analyze event-driven arbitrage opportunity"""

        # Get current price and implied volatility
        current_price = 295.0  # Mock for TSLA
        implied_vol = 0.45  # 45% IV

        # Calculate expected move
        days_to_event = (event_data['date'] - datetime.now()).days
        expected_move = current_price * implied_vol * np.sqrt(days_to_event / 365)

        # Determine strategy based on event type
        if event_data['impact'] == 'HIGH':
            strategy = "Straddle" if implied_vol < 0.5 else "Iron Condor"
        else:
            strategy = "Strangle"

        return ArbitrageSignal(
            signal_id=f"RISK_EVENT_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),

            arb_type="RISK",
            confidence=75 if event_data['impact'] == 'HIGH' else 60,

            primary_asset=asset,
            secondary_asset=None,
            asset_type="OPTIONS",

            buy_venue="Options Exchange",
            buy_price=current_price,
            sell_venue="Options Exchange",
            sell_price=current_price + expected_move,

            spread=expected_move,
            spread_pct=(expected_move / current_price) * 100,
            estimated_profit=self._estimate_event_profit(strategy, expected_move, current_price),

            entry_window=f"Before {event_data['date'].strftime('%b %d')}",
            holding_period="Through event + 1 day",
            capital_required=5000,

            risk_level="HIGH",
            risk_factors=[
                "Event may not meet expectations",
                "IV crush post-event",
                "Binary outcome risk",
                "Time decay if delayed"
            ],
            execution_complexity="COMPLEX",

            execution_steps=self._get_event_strategy_steps(strategy, asset, current_price, expected_move),

            prerequisites=[
                "Options trading approval",
                "Understanding of options Greeks",
                "Ability to manage multi-leg trades",
                "Risk capital only"
            ],

            catalyst_event=event_data['event'],
            event_date=event_data['date']
        )

    async def _analyze_volatility_arbitrage(self, asset: str) -> Optional[ArbitrageSignal]:
        """Analyze volatility arbitrage opportunities"""
        # Compare implied vs realized volatility
        implied_vol = 0.35  # Mock 35% IV
        realized_vol = 0.25  # Mock 25% HV

        vol_spread = implied_vol - realized_vol

        if vol_spread > 0.08:  # IV premium > 8%
            return ArbitrageSignal(
                signal_id=f"RISK_VOL_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),

                arb_type="RISK",
                confidence=min(vol_spread * 500, 80),

                primary_asset=asset,
                secondary_asset=None,
                asset_type="OPTIONS",

                buy_venue="Options Exchange",
                buy_price=100,  # Mock option price
                sell_venue="Options Exchange",
                sell_price=100 * (1 + vol_spread),

                spread=vol_spread * 100,
                spread_pct=vol_spread * 100,
                estimated_profit=vol_spread * 1000,  # Per $1000

                entry_window="Next 1-2 hours",
                holding_period="5-10 days",
                capital_required=3000,

                risk_level="MEDIUM",
                risk_factors=[
                    "Volatility can remain elevated",
                    "Requires precise timing",
                    "Gamma risk on large moves",
                    "Assignment risk on shorts"
                ],
                execution_complexity="COMPLEX",

                execution_steps=[
                    f"1. Sell {asset} options with {implied_vol:.1%} IV",
                    f"2. Delta-hedge with underlying stock",
                    f"3. Realized vol only {realized_vol:.1%}",
                    f"4. Capture {vol_spread:.1%} volatility premium",
                    "5. Adjust hedge daily"
                ],

                prerequisites=[
                    "Advanced options knowledge",
                    "Margin account",
                    "Greeks monitoring tools",
                    "Daily management capability"
                ],

                catalyst_event="Volatility Mean Reversion",
                event_date=None
            )

        return None

    def _estimate_event_profit(self, strategy: str, expected_move: float, price: float) -> float:
        """Estimate profit from event arbitrage"""
        if strategy == "Straddle":
            # Profit if move > premium paid
            premium = expected_move * 0.4  # Rough estimate
            return (expected_move - premium) / price * 100
        elif strategy == "Iron Condor":
            # Profit if stays within range
            return 15.0  # 15% on risk
        else:  # Strangle
            return (expected_move * 0.6) / price * 100

    def _get_event_strategy_steps(self, strategy: str, asset: str, price: float, move: float) -> List[str]:
        """Get specific steps for event strategy"""
        if strategy == "Straddle":
            return [
                f"1. Buy {asset} ${price:.0f} Call",
                f"2. Buy {asset} ${price:.0f} Put",
                f"3. Same expiration (post-event)",
                f"4. Profit if move > {move/price*100:.1f}%",
                "5. Exit day after event"
            ]
        elif strategy == "Iron Condor":
            return [
                f"1. Sell ${price-move:.0f}/{price+move:.0f} Strangle",
                f"2. Buy ${price-move*1.5:.0f}/{price+move*1.5:.0f} Strangle",
                "3. Collect premium upfront",
                f"4. Profit if stays between ${price-move:.0f}-${price+move:.0f}",
                "5. Max profit if expires in range"
            ]
        else:
            return [
                f"1. Buy ${price*1.03:.0f} Call",
                f"2. Buy ${price*0.97:.0f} Put",
                "3. Lower cost than straddle",
                "4. Profit on large moves only",
                "5. Let expire if small move"
            ]

class ArbitrageSignalManager:
    """Manages all arbitrage detection strategies"""

    def __init__(self):
        self.detectors = [
            SpatialArbitrageDetector(),
            StatisticalArbitrageDetector(),
            RiskArbitrageDetector()
        ]

    async def scan_all_opportunities(self, assets: List[str]) -> List[ArbitrageSignal]:
        """Scan for all types of arbitrage opportunities"""
        all_signals = []

        # Run all detectors in parallel
        tasks = [detector.detect_opportunities(assets) for detector in self.detectors]
        results = await asyncio.gather(*tasks)

        # Combine and sort by confidence
        for signals in results:
            all_signals.extend(signals)

        # Sort by confidence and estimated profit
        all_signals.sort(key=lambda x: (x.confidence, x.estimated_profit), reverse=True)

        return all_signals

    def format_signal_alert(self, signal: ArbitrageSignal) -> str:
        """Format arbitrage signal for display"""

        # Emoji based on type
        emoji_map = {
            'SPATIAL': '🌐',
            'STATISTICAL': '📊',
            'RISK': '⚡',
            'MERGER': '🤝'
        }
        emoji = emoji_map.get(signal.arb_type, '💎')

        # Risk color
        risk_color = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}[signal.risk_level]

        alert = f"""
{emoji} {signal.arb_type} ARBITRAGE - {signal.primary_asset}
{'='*50}

🎯 Confidence: {signal.confidence:.0f}%
{risk_color} Risk Level: {signal.risk_level}

💰 OPPORTUNITY:
   Spread: ${signal.spread:.2f} ({signal.spread_pct:.2f}%)
   Est. Profit: ${signal.estimated_profit:.2f}
   Capital Required: ${signal.capital_required:,.0f}

📍 EXECUTION:
   Entry Window: {signal.entry_window}
   Holding Period: {signal.holding_period}
   Complexity: {signal.execution_complexity}
"""

        if signal.secondary_asset:
            alert += f"   Pair: {signal.primary_asset}/{signal.secondary_asset}\n"

        if signal.catalyst_event:
            alert += f"\n🎭 CATALYST: {signal.catalyst_event}\n"
            if signal.event_date:
                alert += f"   Date: {signal.event_date.strftime('%b %d, %Y')}\n"

        alert += "\n📋 EXECUTION STEPS:\n"
        for step in signal.execution_steps:
            alert += f"   {step}\n"

        alert += "\n⚠️ RISK FACTORS:\n"
        for risk in signal.risk_factors[:3]:  # Top 3 risks
            alert += f"   • {risk}\n"

        alert += "\n✅ PREREQUISITES:\n"
        for prereq in signal.prerequisites[:3]:  # Top 3 prereqs
            alert += f"   • {prereq}\n"

        alert += "\n" + "="*50

        return alert

# Example usage
async def demonstrate_arbitrage_signals():
    """Demo arbitrage signal detection"""

    manager = ArbitrageSignalManager()

    # Scan popular assets
    assets = ['TSLA', 'AAPL', 'NVDA', 'SPY', 'BTC-USD', 'ETH-USD']

    print("🔍 Scanning for Arbitrage Opportunities...")
    print("="*60)

    signals = await manager.scan_all_opportunities(assets)

    # Display top opportunities
    print(f"\n📊 Found {len(signals)} Arbitrage Opportunities\n")

    for signal in signals[:5]:  # Top 5
        print(manager.format_signal_alert(signal))
        print()

if __name__ == "__main__":
    asyncio.run(demonstrate_arbitrage_signals())
