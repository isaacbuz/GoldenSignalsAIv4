"""
ETF Arbitrage Agent - Detects arbitrage opportunities between ETFs and their underlying assets.
Analyzes ETF premium/discount, creation/redemption flows, and basket deviation.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from agents.common.base.base_agent import BaseAgent
from scipy import stats

logger = logging.getLogger(__name__)

class ETFArbAgent(BaseAgent):
    """Agent that detects ETF arbitrage opportunities and analyzes ETF flows."""

    def __init__(
        self,
        name: str = "ETFArb",
        premium_threshold: float = 0.005,  # 0.5% premium/discount threshold
        large_premium_threshold: float = 0.015,  # 1.5% large premium/discount
        volume_spike_threshold: float = 2.0,  # 2x average volume
        creation_redemption_threshold: float = 0.01,  # 1% of shares outstanding
        arbitrage_confidence_threshold: float = 0.7
    ):
        """
        Initialize ETF Arbitrage agent.

        Args:
            name: Agent name
            premium_threshold: Threshold for significant premium/discount
            large_premium_threshold: Threshold for large arbitrage opportunity
            volume_spike_threshold: Volume spike threshold for flow analysis
            creation_redemption_threshold: Significant creation/redemption threshold
            arbitrage_confidence_threshold: Minimum confidence for arbitrage signals
        """
        super().__init__(name=name, agent_type="arbitrage")
        self.premium_threshold = premium_threshold
        self.large_premium_threshold = large_premium_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.creation_redemption_threshold = creation_redemption_threshold
        self.arbitrage_confidence_threshold = arbitrage_confidence_threshold

    def calculate_etf_premium_discount(self, etf_price: float, nav: float, intraday_nav: Optional[float] = None) -> Dict[str, float]:
        """Calculate ETF premium/discount to NAV."""
        try:
            if nav <= 0:
                return {'premium_discount': 0.0, 'error': 'Invalid NAV'}

            # Primary premium/discount calculation
            premium_discount = (etf_price - nav) / nav

            # If intraday NAV available, also calculate against that
            intraday_premium = None
            if intraday_nav and intraday_nav > 0:
                intraday_premium = (etf_price - intraday_nav) / intraday_nav

            # Classify the premium/discount
            if abs(premium_discount) >= self.large_premium_threshold:
                classification = 'large_opportunity'
                opportunity_size = 'large'
            elif abs(premium_discount) >= self.premium_threshold:
                classification = 'moderate_opportunity'
                opportunity_size = 'moderate'
            else:
                classification = 'normal'
                opportunity_size = 'small'

            return {
                'premium_discount': premium_discount,
                'intraday_premium': intraday_premium,
                'classification': classification,
                'opportunity_size': opportunity_size,
                'arbitrage_bps': abs(premium_discount) * 10000,  # Basis points
                'direction': 'premium' if premium_discount > 0 else 'discount'
            }

        except Exception as e:
            logger.error(f"ETF premium/discount calculation failed: {str(e)}")
            return {'premium_discount': 0.0, 'error': str(e)}

    def analyze_etf_flows(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ETF creation and redemption flows."""
        try:
            shares_outstanding = flow_data.get('shares_outstanding', 0)
            daily_creations = flow_data.get('daily_creations', 0)
            daily_redemptions = flow_data.get('daily_redemptions', 0)
            historical_flows = flow_data.get('historical_flows', [])

            if shares_outstanding <= 0:
                return {'flow_analysis': 'insufficient_data'}

            # Calculate net flows
            net_flow = daily_creations - daily_redemptions
            net_flow_pct = net_flow / shares_outstanding

            # Analyze flow magnitude
            creation_pct = daily_creations / shares_outstanding
            redemption_pct = daily_redemptions / shares_outstanding

            # Flow classification
            if creation_pct >= self.creation_redemption_threshold:
                creation_signal = 'large_creation'
            elif creation_pct >= self.creation_redemption_threshold / 2:
                creation_signal = 'moderate_creation'
            else:
                creation_signal = 'normal_creation'

            if redemption_pct >= self.creation_redemption_threshold:
                redemption_signal = 'large_redemption'
            elif redemption_pct >= self.creation_redemption_threshold / 2:
                redemption_signal = 'moderate_redemption'
            else:
                redemption_signal = 'normal_redemption'

            # Historical flow analysis
            flow_trend = 'neutral'
            if len(historical_flows) >= 5:
                recent_flows = [f.get('net_flow', 0) for f in historical_flows[-5:]]
                if all(f > 0 for f in recent_flows[-3:]):
                    flow_trend = 'persistent_inflows'
                elif all(f < 0 for f in recent_flows[-3:]):
                    flow_trend = 'persistent_outflows'
                elif sum(recent_flows) > shares_outstanding * 0.05:
                    flow_trend = 'strong_inflows'
                elif sum(recent_flows) < -shares_outstanding * 0.05:
                    flow_trend = 'strong_outflows'

            return {
                'net_flow': net_flow,
                'net_flow_pct': net_flow_pct,
                'creation_signal': creation_signal,
                'redemption_signal': redemption_signal,
                'flow_trend': flow_trend,
                'creation_pct': creation_pct,
                'redemption_pct': redemption_pct
            }

        except Exception as e:
            logger.error(f"ETF flow analysis failed: {str(e)}")
            return {'flow_analysis': 'error'}

    def analyze_basket_deviation(self, etf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deviation between ETF price and basket of underlying securities."""
        try:
            holdings = etf_data.get('holdings', [])
            etf_price = etf_data.get('etf_price', 0)

            if not holdings or etf_price <= 0:
                return {'basket_analysis': 'insufficient_data'}

            # Calculate theoretical basket value
            basket_value = 0.0
            total_weight = 0.0

            for holding in holdings:
                weight = holding.get('weight', 0)
                price = holding.get('current_price', 0)
                shares = holding.get('shares_per_unit', 0)

                if weight > 0 and price > 0:
                    holding_value = weight * price * shares
                    basket_value += holding_value
                    total_weight += weight

            if total_weight < 0.8:  # Require at least 80% of holdings priced
                return {'basket_analysis': 'incomplete_pricing'}

            # Calculate basket deviation
            if basket_value > 0:
                basket_deviation = (etf_price - basket_value) / basket_value

                # Analyze individual holdings for outliers
                outlier_holdings = []
                for holding in holdings:
                    holding_deviation = holding.get('price_deviation', 0)
                    if abs(holding_deviation) > 0.02:  # 2% deviation
                        outlier_holdings.append({
                            'symbol': holding.get('symbol', 'unknown'),
                            'deviation': holding_deviation,
                            'weight': holding.get('weight', 0)
                        })

                return {
                    'basket_value': basket_value,
                    'basket_deviation': basket_deviation,
                    'basket_deviation_bps': abs(basket_deviation) * 10000,
                    'outlier_holdings': outlier_holdings,
                    'pricing_coverage': total_weight
                }

            return {'basket_analysis': 'calculation_error'}

        except Exception as e:
            logger.error(f"Basket deviation analysis failed: {str(e)}")
            return {'basket_analysis': 'error'}

    def detect_arbitrage_opportunities(self, premium_data: Dict[str, Any], flow_data: Dict[str, Any], basket_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect specific arbitrage opportunities."""
        try:
            opportunities = []

            premium_discount = premium_data.get('premium_discount', 0)
            opportunity_size = premium_data.get('opportunity_size', 'small')

            # Primary arbitrage: ETF vs NAV
            if opportunity_size in ['moderate', 'large']:
                if premium_discount > self.premium_threshold:
                    # ETF trading at premium - sell ETF, buy basket
                    opportunities.append({
                        'type': 'etf_premium_arbitrage',
                        'direction': 'sell_etf_buy_basket',
                        'magnitude': abs(premium_discount),
                        'expected_profit_bps': abs(premium_discount) * 10000,
                        'confidence': 0.8 if opportunity_size == 'large' else 0.6,
                        'description': f'ETF trading at {premium_discount:.2%} premium to NAV'
                    })
                elif premium_discount < -self.premium_threshold:
                    # ETF trading at discount - buy ETF, sell basket
                    opportunities.append({
                        'type': 'etf_discount_arbitrage',
                        'direction': 'buy_etf_sell_basket',
                        'magnitude': abs(premium_discount),
                        'expected_profit_bps': abs(premium_discount) * 10000,
                        'confidence': 0.8 if opportunity_size == 'large' else 0.6,
                        'description': f'ETF trading at {abs(premium_discount):.2%} discount to NAV'
                    })

            # Flow-based arbitrage opportunities
            creation_signal = flow_data.get('creation_signal', 'normal_creation')
            redemption_signal = flow_data.get('redemption_signal', 'normal_redemption')
            flow_trend = flow_data.get('flow_trend', 'neutral')

            if creation_signal == 'large_creation' and premium_discount > 0:
                opportunities.append({
                    'type': 'creation_arbitrage',
                    'direction': 'create_units',
                    'magnitude': abs(premium_discount),
                    'expected_profit_bps': abs(premium_discount) * 10000 * 0.8,  # Adjusted for costs
                    'confidence': 0.7,
                    'description': 'Large creation activity with ETF premium'
                })

            if redemption_signal == 'large_redemption' and premium_discount < 0:
                opportunities.append({
                    'type': 'redemption_arbitrage',
                    'direction': 'redeem_units',
                    'magnitude': abs(premium_discount),
                    'expected_profit_bps': abs(premium_discount) * 10000 * 0.8,  # Adjusted for costs
                    'confidence': 0.7,
                    'description': 'Large redemption activity with ETF discount'
                })

            # Basket deviation arbitrage
            basket_deviation = basket_data.get('basket_deviation', 0)
            if abs(basket_deviation) > self.premium_threshold:
                opportunities.append({
                    'type': 'basket_arbitrage',
                    'direction': 'exploit_basket_deviation',
                    'magnitude': abs(basket_deviation),
                    'expected_profit_bps': abs(basket_deviation) * 10000 * 0.6,  # Adjusted for complexity
                    'confidence': 0.5,
                    'description': f'Basket deviation of {basket_deviation:.2%}'
                })

            # Filter opportunities by confidence threshold
            high_confidence_opportunities = [
                opp for opp in opportunities
                if opp['confidence'] >= self.arbitrage_confidence_threshold
            ]

            return high_confidence_opportunities

        except Exception as e:
            logger.error(f"Arbitrage opportunity detection failed: {str(e)}")
            return []

    def calculate_arbitrage_risk_factors(self, etf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk factors that could affect arbitrage profitability."""
        try:
            risk_factors = {}

            # Liquidity risk
            etf_volume = etf_data.get('volume', 0)
            avg_volume = etf_data.get('avg_volume_30d', 1)
            liquidity_ratio = etf_volume / max(avg_volume, 1)

            if liquidity_ratio < 0.5:
                risk_factors['liquidity_risk'] = 'high'
            elif liquidity_ratio < 1.0:
                risk_factors['liquidity_risk'] = 'medium'
            else:
                risk_factors['liquidity_risk'] = 'low'

            # Volatility risk
            volatility = etf_data.get('volatility_30d', 0)
            if volatility > 0.3:  # 30% annualized volatility
                risk_factors['volatility_risk'] = 'high'
            elif volatility > 0.2:
                risk_factors['volatility_risk'] = 'medium'
            else:
                risk_factors['volatility_risk'] = 'low'

            # Concentration risk (from basket analysis)
            holdings = etf_data.get('holdings', [])
            if holdings:
                top_5_weight = sum(h.get('weight', 0) for h in holdings[:5])
                if top_5_weight > 0.5:
                    risk_factors['concentration_risk'] = 'high'
                elif top_5_weight > 0.3:
                    risk_factors['concentration_risk'] = 'medium'
                else:
                    risk_factors['concentration_risk'] = 'low'

            # Tracking error risk
            tracking_error = etf_data.get('tracking_error_30d', 0)
            if tracking_error > 0.02:  # 2% tracking error
                risk_factors['tracking_risk'] = 'high'
            elif tracking_error > 0.01:
                risk_factors['tracking_risk'] = 'medium'
            else:
                risk_factors['tracking_risk'] = 'low'

            # Overall risk assessment
            high_risk_count = sum(1 for risk in risk_factors.values() if risk == 'high')
            if high_risk_count >= 2:
                risk_factors['overall_risk'] = 'high'
            elif high_risk_count >= 1:
                risk_factors['overall_risk'] = 'medium'
            else:
                risk_factors['overall_risk'] = 'low'

            return risk_factors

        except Exception as e:
            logger.error(f"Risk factor calculation failed: {str(e)}")
            return {'overall_risk': 'unknown'}

    def generate_arbitrage_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on ETF arbitrage analysis."""
        try:
            etf_price = data.get('etf_price', 0)
            nav = data.get('nav', 0)
            intraday_nav = data.get('intraday_nav')
            flow_data = data.get('flow_data', {})
            etf_data = data.get('etf_data', {})

            if etf_price <= 0 or nav <= 0:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'insufficient_data',
                    'reasoning': 'Missing ETF price or NAV data'
                }

            # Calculate premium/discount
            premium_data = self.calculate_etf_premium_discount(etf_price, nav, intraday_nav)

            # Analyze flows
            flow_analysis = self.analyze_etf_flows(flow_data)

            # Analyze basket deviation
            basket_analysis = self.analyze_basket_deviation(etf_data)

            # Detect arbitrage opportunities
            opportunities = self.detect_arbitrage_opportunities(premium_data, flow_analysis, basket_analysis)

            # Calculate risk factors
            risk_factors = self.calculate_arbitrage_risk_factors(etf_data)

            # Generate primary signal
            action = "hold"
            confidence = 0.0
            signal_type = "no_arbitrage"
            reasoning = []

            if opportunities:
                best_opportunity = max(opportunities, key=lambda x: x['confidence'])

                if best_opportunity['direction'] in ['sell_etf_buy_basket', 'redeem_units']:
                    action = "sell"
                    signal_type = "arbitrage_sell_etf"
                elif best_opportunity['direction'] in ['buy_etf_sell_basket', 'create_units']:
                    action = "buy"
                    signal_type = "arbitrage_buy_etf"
                else:
                    action = "hold"
                    signal_type = "complex_arbitrage"

                confidence = best_opportunity['confidence']
                reasoning.append(best_opportunity['description'])
                reasoning.append(f"Expected profit: {best_opportunity['expected_profit_bps']:.0f} bps")

            # Risk adjustments
            overall_risk = risk_factors.get('overall_risk', 'medium')
            if overall_risk == 'high':
                confidence *= 0.6
                reasoning.append("High risk factors detected")
            elif overall_risk == 'low':
                confidence *= 1.2
                reasoning.append("Low risk environment")

            # Multiple opportunities boost confidence
            if len(opportunities) > 1:
                confidence *= 1.1
                reasoning.append(f"Multiple arbitrage opportunities ({len(opportunities)})")

            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"ETF arbitrage signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ETF data and generate arbitrage signals."""
        try:
            if "etf_price" not in data or "nav" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "ETF price or NAV not provided"}
                }

            # Generate signals
            signal_data = self.generate_arbitrage_signals(data)

            # Get comprehensive analysis
            etf_price = data["etf_price"]
            nav = data["nav"]
            intraday_nav = data.get("intraday_nav")
            flow_data = data.get("flow_data", {})
            etf_data = data.get("etf_data", {})

            premium_data = self.calculate_etf_premium_discount(etf_price, nav, intraday_nav)
            flow_analysis = self.analyze_etf_flows(flow_data)
            basket_analysis = self.analyze_basket_deviation(etf_data)
            opportunities = self.detect_arbitrage_opportunities(premium_data, flow_analysis, basket_analysis)
            risk_factors = self.calculate_arbitrage_risk_factors(etf_data)

            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "premium_data": premium_data,
                    "flow_analysis": flow_analysis,
                    "basket_analysis": basket_analysis,
                    "arbitrage_opportunities": opportunities,
                    "risk_factors": risk_factors
                }
            }

        except Exception as e:
            logger.error(f"ETF arbitrage signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
