"""
Options Flow Intelligence RAG
Detects and analyzes institutional options flow patterns for early signals
Issue #182: RAG-3: Implement Options Flow Intelligence RAG
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
from dataclasses import dataclass
import logging
from enum import Enum
import re

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of options flow"""
    SWEEP = "sweep"  # Aggressive multi-exchange sweep
    BLOCK = "block"  # Large single block trade
    SPLIT = "split"  # Order split across strikes/expiries
    REPEAT = "repeat"  # Repeated similar orders
    UNUSUAL = "unusual"  # Unusual size/strike/expiry


class InstitutionType(Enum):
    """Types of institutional traders"""
    HEDGE_FUND = "hedge_fund"
    MARKET_MAKER = "market_maker"
    PROPRIETARY = "proprietary_trading"
    INSURANCE = "insurance_company"
    PENSION = "pension_fund"
    RETAIL = "retail_aggregate"
    UNKNOWN = "unknown"


class PositionIntent(Enum):
    """Inferred intent of the position"""
    DIRECTIONAL_BULLISH = "directional_bullish"
    DIRECTIONAL_BEARISH = "directional_bearish"
    HEDGE = "hedge"
    VOLATILITY_LONG = "volatility_long"
    VOLATILITY_SHORT = "volatility_short"
    INCOME = "income_generation"
    SPREAD = "spread_strategy"


@dataclass
class OptionsFlow:
    """Represents an options flow event"""
    id: str
    timestamp: datetime
    symbol: str
    underlying_price: float
    strike: float
    expiry: datetime
    call_put: str  # 'C' or 'P'
    side: str  # 'BUY' or 'SELL'
    size: int
    price: float
    notional: float  # Size * Price * 100
    implied_volatility: float
    delta: float
    gamma: float
    flow_type: FlowType
    exchange: str
    # Analysis fields
    institution_type: InstitutionType
    position_intent: PositionIntent
    aggressiveness: float  # 0-1 scale
    smart_money_score: float  # 0-100
    # Historical outcome
    price_after_1d: Optional[float] = None
    price_after_3d: Optional[float] = None
    price_after_7d: Optional[float] = None
    max_profit_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'underlying_price': self.underlying_price,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'call_put': self.call_put,
            'side': self.side,
            'size': self.size,
            'price': self.price,
            'notional': self.notional,
            'iv': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'flow_type': self.flow_type.value,
            'institution_type': self.institution_type.value,
            'position_intent': self.position_intent.value,
            'aggressiveness': self.aggressiveness,
            'smart_money_score': self.smart_money_score
        }
    
    def to_search_text(self) -> str:
        """Convert flow to searchable text"""
        moneyness = (self.strike / self.underlying_price - 1) * 100
        dte = (self.expiry - self.timestamp).days
        
        return f"""
        {self.symbol} {self.call_put} Strike: ${self.strike:.2f} ({moneyness:+.1f}% OTM)
        Expiry: {dte} days, Size: {self.size:,} contracts (${self.notional:,.0f})
        Type: {self.flow_type.value}, Institution: {self.institution_type.value}
        Intent: {self.position_intent.value}, Aggressiveness: {self.aggressiveness:.2f}
        Greeks: Delta {self.delta:.3f}, Gamma {self.gamma:.3f}, IV {self.implied_volatility:.1%}
        Smart Money Score: {self.smart_money_score:.0f}/100
        """


class FlowAnalyzer:
    """Analyzes options flow characteristics"""
    
    def __init__(self):
        self.institution_patterns = {
            InstitutionType.HEDGE_FUND: {
                'min_size': 500,
                'typical_dte': (7, 45),
                'aggressive': True,
                'repeat_trades': True
            },
            InstitutionType.MARKET_MAKER: {
                'min_size': 1000,
                'typical_dte': (0, 7),
                'aggressive': False,
                'delta_neutral': True
            },
            InstitutionType.INSURANCE: {
                'min_size': 5000,
                'typical_dte': (30, 180),
                'aggressive': False,
                'protective': True
            }
        }
    
    def identify_institution_type(self, flow: Dict[str, Any]) -> InstitutionType:
        """Identify likely institution type based on flow characteristics"""
        size = flow['size']
        dte = flow.get('days_to_expiry', 30)
        aggressive = flow.get('aggressive_order', False)
        
        # Large size + long dated = Insurance/Pension
        if size > 5000 and dte > 60:
            return InstitutionType.INSURANCE
        
        # Medium size + short dated + aggressive = Hedge Fund
        if 500 <= size <= 5000 and dte < 45 and aggressive:
            return InstitutionType.HEDGE_FUND
        
        # Very short dated + large = Market Maker
        if dte < 7 and size > 1000:
            return InstitutionType.MARKET_MAKER
        
        # Small size = Retail
        if size < 100:
            return InstitutionType.RETAIL
        
        return InstitutionType.UNKNOWN
    
    def infer_position_intent(self, flow: Dict[str, Any]) -> PositionIntent:
        """Infer the intent behind the position"""
        call_put = flow['call_put']
        side = flow['side']
        delta = flow.get('delta', 0.5)
        iv = flow.get('implied_volatility', 0.3)
        moneyness = flow.get('moneyness', 0)
        
        # Directional plays
        if side == 'BUY':
            if call_put == 'C' and delta > 0.6:
                return PositionIntent.DIRECTIONAL_BULLISH
            elif call_put == 'P' and delta < -0.6:
                return PositionIntent.DIRECTIONAL_BEARISH
        
        # Selling premium (income)
        if side == 'SELL' and abs(moneyness) < 5:
            return PositionIntent.INCOME
        
        # Volatility plays
        if abs(moneyness) > 10:  # Far OTM
            if side == 'BUY':
                return PositionIntent.VOLATILITY_LONG
            else:
                return PositionIntent.VOLATILITY_SHORT
        
        # Protective hedges
        if call_put == 'P' and side == 'BUY' and moneyness < -5:
            return PositionIntent.HEDGE
        
        return PositionIntent.SPREAD
    
    def calculate_smart_money_score(self, flow: Dict[str, Any]) -> float:
        """Calculate how likely this is smart money"""
        score = 50.0  # Base score
        
        # Size factor
        size = flow['size']
        if size > 1000:
            score += 20
        elif size > 500:
            score += 10
        elif size < 100:
            score -= 20
        
        # Aggressiveness (sweep orders)
        if flow.get('flow_type') == FlowType.SWEEP.value:
            score += 15
        
        # Timing (before events)
        if flow.get('days_to_event', 999) < 5:
            score += 10
        
        # Unusual activity
        if flow.get('volume_ratio', 1) > 3:
            score += 15
        
        # Institution type
        inst_type = flow.get('institution_type')
        if inst_type == InstitutionType.HEDGE_FUND.value:
            score += 10
        elif inst_type == InstitutionType.RETAIL.value:
            score -= 15
        
        return max(0, min(100, score))


class OptionsFlowIntelligenceRAG:
    """
    RAG system for institutional options flow patterns
    Detects smart money movements and predicts price action
    """
    
    def __init__(self, use_mock_db: bool = True):
        """Initialize the Options Flow Intelligence RAG"""
        self.use_mock_db = use_mock_db
        self.flow_analyzer = FlowAnalyzer()
        self.options_flows: List[OptionsFlow] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        
        if use_mock_db:
            self._load_mock_options_flows()
    
    def _load_mock_options_flows(self):
        """Load mock historical options flow data"""
        mock_flows = [
            # Hedge fund accumulation before earnings
            OptionsFlow(
                id="hf_aapl_calls_2023",
                timestamp=datetime(2023, 7, 25, 10, 30),
                symbol="AAPL",
                underlying_price=190.0,
                strike=200.0,
                expiry=datetime(2023, 8, 18),
                call_put="C",
                side="BUY",
                size=2500,
                price=3.50,
                notional=875000,
                implied_volatility=0.28,
                delta=0.35,
                gamma=0.02,
                flow_type=FlowType.SWEEP,
                exchange="MULTIPLE",
                institution_type=InstitutionType.HEDGE_FUND,
                position_intent=PositionIntent.DIRECTIONAL_BULLISH,
                aggressiveness=0.85,
                smart_money_score=85.0,
                price_after_1d=193.0,
                price_after_3d=198.0,
                price_after_7d=205.0,
                max_profit_percent=180.0
            ),
            # Market maker hedging
            OptionsFlow(
                id="mm_spy_puts_2023",
                timestamp=datetime(2023, 9, 15, 14, 45),
                symbol="SPY",
                underlying_price=440.0,
                strike=435.0,
                expiry=datetime(2023, 9, 22),
                call_put="P",
                side="BUY",
                size=10000,
                price=2.10,
                notional=2100000,
                implied_volatility=0.22,
                delta=-0.30,
                gamma=0.03,
                flow_type=FlowType.BLOCK,
                exchange="CBOE",
                institution_type=InstitutionType.MARKET_MAKER,
                position_intent=PositionIntent.HEDGE,
                aggressiveness=0.20,
                smart_money_score=60.0,
                price_after_1d=438.0,
                price_after_3d=432.0,
                price_after_7d=430.0,
                max_profit_percent=40.0
            ),
            # Insurance company protection
            OptionsFlow(
                id="ins_spy_collar_2023",
                timestamp=datetime(2023, 6, 1, 11, 0),
                symbol="SPY",
                underlying_price=425.0,
                strike=400.0,
                expiry=datetime(2023, 12, 15),
                call_put="P",
                side="BUY",
                size=25000,
                price=12.50,
                notional=31250000,
                implied_volatility=0.18,
                delta=-0.25,
                gamma=0.01,
                flow_type=FlowType.BLOCK,
                exchange="ISE",
                institution_type=InstitutionType.INSURANCE,
                position_intent=PositionIntent.HEDGE,
                aggressiveness=0.10,
                smart_money_score=70.0,
                price_after_1d=424.0,
                price_after_3d=422.0,
                price_after_7d=420.0,
                max_profit_percent=5.0
            ),
            # Unusual call buying before takeover
            OptionsFlow(
                id="unusual_acq_calls_2023",
                timestamp=datetime(2023, 8, 10, 15, 30),
                symbol="XYZ",
                underlying_price=50.0,
                strike=55.0,
                expiry=datetime(2023, 9, 15),
                call_put="C",
                side="BUY",
                size=5000,
                price=1.20,
                notional=600000,
                implied_volatility=0.45,
                delta=0.30,
                gamma=0.04,
                flow_type=FlowType.UNUSUAL,
                exchange="MULTIPLE",
                institution_type=InstitutionType.HEDGE_FUND,
                position_intent=PositionIntent.DIRECTIONAL_BULLISH,
                aggressiveness=0.90,
                smart_money_score=95.0,
                price_after_1d=52.0,
                price_after_3d=58.0,
                price_after_7d=62.0,
                max_profit_percent=400.0
            ),
            # Volatility play before Fed
            OptionsFlow(
                id="vol_play_vix_2023",
                timestamp=datetime(2023, 9, 18, 9, 45),
                symbol="SPY",
                underlying_price=445.0,
                strike=440.0,
                expiry=datetime(2023, 9, 20),
                call_put="P",
                side="BUY",
                size=3000,
                price=1.50,
                notional=450000,
                implied_volatility=0.35,
                delta=-0.40,
                gamma=0.08,
                flow_type=FlowType.SWEEP,
                exchange="MULTIPLE",
                institution_type=InstitutionType.PROPRIETARY,
                position_intent=PositionIntent.VOLATILITY_LONG,
                aggressiveness=0.75,
                smart_money_score=80.0,
                price_after_1d=442.0,
                price_after_3d=438.0,
                price_after_7d=435.0,
                max_profit_percent=150.0
            )
        ]
        
        self.options_flows.extend(mock_flows)
        self._generate_mock_embeddings()
    
    def _generate_mock_embeddings(self):
        """Generate embeddings for options flows"""
        for flow in self.options_flows:
            # Create embedding based on flow characteristics
            moneyness = (flow.strike / flow.underlying_price - 1) * 100
            dte = (flow.expiry - flow.timestamp).days
            
            embedding = np.array([
                1.0 if flow.call_put == 'C' else -1.0,
                1.0 if flow.side == 'BUY' else -1.0,
                flow.size / 10000,  # Normalized size
                moneyness / 10,  # Normalized moneyness
                dte / 180,  # Normalized days to expiry
                flow.implied_volatility,
                flow.delta,
                flow.aggressiveness,
                flow.smart_money_score / 100,
                flow.notional / 1000000  # Millions
            ])
            
            self.embeddings[flow.id] = embedding
    
    def _calculate_similarity(self, query_embedding: np.ndarray,
                            flow_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(query_embedding, flow_embedding)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(flow_embedding)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product
    
    async def analyze_options_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a new options flow for institutional patterns"""
        # Identify institution type
        institution_type = self.flow_analyzer.identify_institution_type(flow_data)
        
        # Infer position intent
        position_intent = self.flow_analyzer.infer_position_intent(flow_data)
        
        # Calculate smart money score
        smart_money_score = self.flow_analyzer.calculate_smart_money_score(flow_data)
        
        # Create embedding for similarity search
        moneyness = ((flow_data['strike'] / flow_data['underlying_price']) - 1) * 100
        dte = flow_data.get('days_to_expiry', 30)
        
        query_embedding = np.array([
            1.0 if flow_data['call_put'] == 'C' else -1.0,
            1.0 if flow_data['side'] == 'BUY' else -1.0,
            flow_data['size'] / 10000,
            moneyness / 10,
            dte / 180,
            flow_data.get('implied_volatility', 0.3),
            flow_data.get('delta', 0.5),
            flow_data.get('aggressiveness', 0.5),
            smart_money_score / 100,
            flow_data.get('notional', flow_data['size'] * flow_data['price'] * 100) / 1000000
        ])
        
        # Find similar historical flows
        similarities = []
        for flow in self.options_flows:
            if flow.symbol == flow_data['symbol']:  # Same underlying
                flow_embedding = self.embeddings[flow.id]
                similarity = self._calculate_similarity(query_embedding, flow_embedding)
                similarities.append((similarity, flow))
        
        # Sort and get top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_matches = similarities[:5]
        
        # Calculate expected outcomes
        if top_matches:
            avg_1d_move = np.mean([(f.price_after_1d / f.underlying_price - 1) * 100 
                                  for _, f in top_matches if f.price_after_1d])
            avg_3d_move = np.mean([(f.price_after_3d / f.underlying_price - 1) * 100 
                                  for _, f in top_matches if f.price_after_3d])
            avg_profit = np.mean([f.max_profit_percent for _, f in top_matches 
                                if f.max_profit_percent])
        else:
            avg_1d_move = 0
            avg_3d_move = 0
            avg_profit = 0
        
        return {
            'flow_analysis': {
                'institution_type': institution_type.value,
                'position_intent': position_intent.value,
                'smart_money_score': smart_money_score,
                'aggressiveness': flow_data.get('aggressiveness', 0.5)
            },
            'similar_flows': [
                {
                    'similarity': float(sim),
                    'date': flow.timestamp.isoformat(),
                    'description': f"{flow.symbol} {flow.call_put} ${flow.strike}",
                    'outcome': {
                        '1d_move': f"{(flow.price_after_1d / flow.underlying_price - 1) * 100:.1f}%" if flow.price_after_1d else "N/A",
                        '3d_move': f"{(flow.price_after_3d / flow.underlying_price - 1) * 100:.1f}%" if flow.price_after_3d else "N/A",
                        'max_profit': f"{flow.max_profit_percent:.0f}%" if flow.max_profit_percent else "N/A"
                    }
                }
                for sim, flow in top_matches[:3]
            ],
            'expected_impact': {
                '1d_price_move': float(avg_1d_move),
                '3d_price_move': float(avg_3d_move),
                'option_profit_potential': float(avg_profit),
                'confidence': float(np.mean([s for s, _ in top_matches])) if top_matches else 0
            },
            'trading_signals': self._generate_trading_signals(
                flow_data, institution_type, position_intent, 
                smart_money_score, avg_3d_move
            )
        }
    
    def _generate_trading_signals(self, flow_data: Dict[str, Any],
                                institution_type: InstitutionType,
                                position_intent: PositionIntent,
                                smart_money_score: float,
                                expected_move: float) -> Dict[str, Any]:
        """Generate specific trading signals based on flow analysis"""
        signals = {
            'follow_smart_money': False,
            'action': 'monitor',
            'confidence': 0.0,
            'strategy': None,
            'entry_timing': 'wait',
            'position_size': 0.0,
            'risk_management': {}
        }
        
        # High smart money score + directional intent = follow
        if smart_money_score > 75:
            if position_intent == PositionIntent.DIRECTIONAL_BULLISH:
                signals['follow_smart_money'] = True
                signals['action'] = 'buy'
                signals['strategy'] = 'buy_calls_or_stock'
                signals['confidence'] = smart_money_score / 100
            elif position_intent == PositionIntent.DIRECTIONAL_BEARISH:
                signals['follow_smart_money'] = True
                signals['action'] = 'sell'
                signals['strategy'] = 'buy_puts_or_short'
                signals['confidence'] = smart_money_score / 100
            elif position_intent == PositionIntent.VOLATILITY_LONG:
                signals['follow_smart_money'] = True
                signals['action'] = 'hedge'
                signals['strategy'] = 'buy_straddle'
                signals['confidence'] = smart_money_score / 100 * 0.8
        
        # Determine entry timing
        if institution_type == InstitutionType.HEDGE_FUND and signals['follow_smart_money']:
            signals['entry_timing'] = 'immediate'
        elif institution_type == InstitutionType.INSURANCE:
            signals['entry_timing'] = 'scale_in'
        else:
            signals['entry_timing'] = 'wait_confirmation'
        
        # Position sizing based on confidence
        if signals['confidence'] > 0.8:
            signals['position_size'] = 1.0
        elif signals['confidence'] > 0.6:
            signals['position_size'] = 0.5
        else:
            signals['position_size'] = 0.25
        
        # Risk management
        if abs(expected_move) > 5:
            signals['risk_management'] = {
                'stop_loss': abs(expected_move) * 0.5,
                'take_profit': abs(expected_move) * 1.5,
                'max_risk': '2% of portfolio'
            }
        else:
            signals['risk_management'] = {
                'stop_loss': 2.0,
                'take_profit': 5.0,
                'max_risk': '1% of portfolio'
            }
        
        return signals
    
    async def detect_unusual_activity(self, symbol: str, 
                                    timeframe: str = '1d') -> Dict[str, Any]:
        """Detect unusual options activity for a symbol"""
        # Filter flows for the symbol
        symbol_flows = [f for f in self.options_flows if f.symbol == symbol]
        
        if not symbol_flows:
            return {
                'symbol': symbol,
                'unusual_activity': False,
                'message': 'No options flow data available'
            }
        
        # Calculate metrics
        total_volume = sum(f.size for f in symbol_flows)
        total_notional = sum(f.notional for f in symbol_flows)
        
        # Identify unusual patterns
        unusual_flows = []
        for flow in symbol_flows:
            if flow.smart_money_score > 80 or flow.flow_type == FlowType.UNUSUAL:
                unusual_flows.append({
                    'timestamp': flow.timestamp.isoformat(),
                    'description': f"{flow.call_put} ${flow.strike} x{flow.size}",
                    'smart_money_score': flow.smart_money_score,
                    'institution_type': flow.institution_type.value,
                    'intent': flow.position_intent.value
                })
        
        # Aggregate by intent
        intent_summary = {}
        for flow in symbol_flows:
            intent = flow.position_intent.value
            if intent not in intent_summary:
                intent_summary[intent] = {'count': 0, 'notional': 0}
            intent_summary[intent]['count'] += 1
            intent_summary[intent]['notional'] += flow.notional
        
        # Determine overall bias
        bullish_notional = sum(intent_summary.get(intent, {}).get('notional', 0)
                              for intent in ['directional_bullish'])
        bearish_notional = sum(intent_summary.get(intent, {}).get('notional', 0)
                              for intent in ['directional_bearish', 'hedge'])
        
        if bullish_notional > bearish_notional * 2:
            overall_bias = 'strongly_bullish'
        elif bullish_notional > bearish_notional * 1.5:
            overall_bias = 'bullish'
        elif bearish_notional > bullish_notional * 2:
            overall_bias = 'strongly_bearish'
        elif bearish_notional > bullish_notional * 1.5:
            overall_bias = 'bearish'
        else:
            overall_bias = 'neutral'
        
        return {
            'symbol': symbol,
            'unusual_activity': len(unusual_flows) > 0,
            'unusual_flows': unusual_flows,
            'total_volume': total_volume,
            'total_notional': total_notional,
            'intent_summary': intent_summary,
            'overall_bias': overall_bias,
            'smart_money_detected': any(f.smart_money_score > 75 for f in symbol_flows),
            'recommendation': self._generate_activity_recommendation(
                unusual_flows, overall_bias, intent_summary
            )
        }
    
    def _generate_activity_recommendation(self, unusual_flows: List[Dict],
                                        overall_bias: str,
                                        intent_summary: Dict) -> Dict[str, Any]:
        """Generate recommendation based on unusual activity"""
        if not unusual_flows:
            return {
                'action': 'no_action',
                'reason': 'No unusual activity detected'
            }
        
        # Strong smart money signal
        if any(f['smart_money_score'] > 85 for f in unusual_flows):
            if overall_bias in ['strongly_bullish', 'bullish']:
                return {
                    'action': 'follow_bullish',
                    'confidence': 'high',
                    'reason': 'Strong institutional buying detected',
                    'suggested_strategy': 'Buy calls or stock'
                }
            elif overall_bias in ['strongly_bearish', 'bearish']:
                return {
                    'action': 'follow_bearish',
                    'confidence': 'high',
                    'reason': 'Strong institutional hedging/selling detected',
                    'suggested_strategy': 'Buy puts or reduce longs'
                }
        
        # Mixed signals
        return {
            'action': 'monitor',
            'confidence': 'medium',
            'reason': 'Unusual activity detected but mixed signals',
            'suggested_strategy': 'Wait for confirmation'
        }


# Demo function
async def demo_options_flow_rag():
    """Demonstrate the Options Flow Intelligence RAG"""
    rag = OptionsFlowIntelligenceRAG(use_mock_db=True)
    
    print("Options Flow Intelligence RAG Demo")
    print("="*70)
    
    # Test 1: Analyze a new bullish flow
    print("\n1. Analyzing Bullish Call Sweep:")
    print("-"*50)
    
    new_flow = {
        'symbol': 'AAPL',
        'underlying_price': 195.0,
        'strike': 200.0,
        'days_to_expiry': 30,
        'call_put': 'C',
        'side': 'BUY',
        'size': 3000,
        'price': 4.0,
        'notional': 1200000,
        'implied_volatility': 0.32,
        'delta': 0.45,
        'flow_type': FlowType.SWEEP.value,
        'aggressive_order': True,
        'volume_ratio': 5.0  # 5x normal volume
    }
    
    result = await rag.analyze_options_flow(new_flow)
    
    print(f"Institution Type: {result['flow_analysis']['institution_type']}")
    print(f"Position Intent: {result['flow_analysis']['position_intent']}")
    print(f"Smart Money Score: {result['flow_analysis']['smart_money_score']:.0f}/100")
    
    print(f"\nExpected Impact:")
    print(f"  1-Day Move: {result['expected_impact']['1d_price_move']:.1f}%")
    print(f"  3-Day Move: {result['expected_impact']['3d_price_move']:.1f}%")
    print(f"  Option Profit: {result['expected_impact']['option_profit_potential']:.0f}%")
    
    print(f"\nTrading Signal:")
    signals = result['trading_signals']
    print(f"  Action: {signals['action'].upper()}")
    print(f"  Strategy: {signals['strategy']}")
    print(f"  Confidence: {signals['confidence']:.1%}")
    print(f"  Position Size: {signals['position_size']:.1%}")
    
    # Test 2: Detect unusual activity
    print("\n\n2. Detecting Unusual Activity for AAPL:")
    print("-"*50)
    
    activity = await rag.detect_unusual_activity('AAPL')
    
    print(f"Unusual Activity: {'YES' if activity['unusual_activity'] else 'NO'}")
    print(f"Overall Bias: {activity['overall_bias'].upper()}")
    print(f"Smart Money Detected: {'YES' if activity['smart_money_detected'] else 'NO'}")
    
    if activity['unusual_flows']:
        print(f"\nUnusual Flows Detected ({len(activity['unusual_flows'])}):")
        for flow in activity['unusual_flows'][:3]:
            print(f"  - {flow['description']} (Score: {flow['smart_money_score']:.0f})")
    
    print(f"\nRecommendation:")
    rec = activity['recommendation']
    print(f"  Action: {rec['action'].upper()}")
    print(f"  Reason: {rec['reason']}")
    if 'suggested_strategy' in rec:
        print(f"  Strategy: {rec['suggested_strategy']}")
    
    # Test 3: Analyze bearish hedge flow
    print("\n\n3. Analyzing Protective Put Buying:")
    print("-"*50)
    
    hedge_flow = {
        'symbol': 'SPY',
        'underlying_price': 450.0,
        'strike': 440.0,
        'days_to_expiry': 7,
        'call_put': 'P',
        'side': 'BUY',
        'size': 5000,
        'price': 2.5,
        'notional': 1250000,
        'implied_volatility': 0.25,
        'delta': -0.35,
        'flow_type': FlowType.BLOCK.value,
        'volume_ratio': 3.0
    }
    
    hedge_result = await rag.analyze_options_flow(hedge_flow)
    
    print(f"Institution Type: {hedge_result['flow_analysis']['institution_type']}")
    print(f"Position Intent: {hedge_result['flow_analysis']['position_intent']}")
    print(f"Smart Money Score: {hedge_result['flow_analysis']['smart_money_score']:.0f}/100")
    
    print(f"\nRisk Management Suggestion:")
    rm = hedge_result['trading_signals']['risk_management']
    print(f"  Stop Loss: {rm['stop_loss']:.1f}%")
    print(f"  Take Profit: {rm['take_profit']:.1f}%")
    print(f"  Max Risk: {rm['max_risk']}")


if __name__ == "__main__":
    asyncio.run(demo_options_flow_rag()) 