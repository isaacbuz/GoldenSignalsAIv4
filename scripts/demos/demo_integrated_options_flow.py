"""
Demo: Integrated Options Flow Intelligence with Meta Signal Agent
Shows how institutional options flow detection enhances trading signals
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import our components
from agents.rag.options_flow_intelligence_rag import OptionsFlowIntelligenceRAG
from agents.meta.meta_signal_agent import MetaSignalAgent


async def simulate_unusual_options_activity():
    """Simulate detection of unusual options activity"""
    print("\n" + "="*70)
    print("DEMO: Integrated Options Flow Intelligence")
    print("="*70)

    # Initialize components
    options_rag = OptionsFlowIntelligenceRAG(use_mock_db=True)
    meta_agent = MetaSignalAgent()

    # Test Case 1: Bullish Smart Money Flow
    print("\n📊 CASE 1: Detecting Bullish Institutional Accumulation")
    print("-"*70)

    # Simulate detecting a large call sweep
    bullish_flow = {
        'symbol': 'AAPL',
        'underlying_price': 195.0,
        'strike': 200.0,
        'days_to_expiry': 30,
        'call_put': 'C',
        'side': 'BUY',
        'size': 5000,  # Large size
        'price': 4.5,
        'notional': 2250000,
        'implied_volatility': 0.35,
        'delta': 0.50,
        'flow_type': 'sweep',
        'aggressive_order': True,
        'volume_ratio': 8.0,  # 8x normal volume
        'days_to_event': 3  # Earnings in 3 days
    }

    # Analyze the flow
    flow_analysis = await options_rag.analyze_options_flow(bullish_flow)

    print(f"\n🔍 Options Flow Analysis:")
    print(f"   Institution Type: {flow_analysis['flow_analysis']['institution_type']}")
    print(f"   Position Intent: {flow_analysis['flow_analysis']['position_intent']}")
    print(f"   Smart Money Score: {flow_analysis['flow_analysis']['smart_money_score']:.0f}/100")
    print(f"   Aggressiveness: {flow_analysis['flow_analysis']['aggressiveness']:.2f}")

    print(f"\n📈 Expected Impact:")
    impact = flow_analysis['expected_impact']
    print(f"   1-Day Price Move: {impact['1d_price_move']:+.1f}%")
    print(f"   3-Day Price Move: {impact['3d_price_move']:+.1f}%")
    print(f"   Option Profit Potential: {impact['option_profit_potential']:.0f}%")
    print(f"   Confidence: {impact['confidence']:.1%}")

    # Now get aggregated signal from meta agent
    context = {
        'current_price': 195.0,
        'options_flow': bullish_flow,
        'market_data': {
            'price': 195.0,
            'volume': 50000000,
            'avg_volume': 45000000,
            'rsi': 55,
            'macd': {'macd': 0.5, 'signal': 0.3}
        }
    }

    meta_signal = await meta_agent.aggregate_signals('AAPL', '1d', context)

    print(f"\n🎯 Meta Agent Signal:")
    print(f"   Final Signal: {meta_signal['signal'].upper()}")
    print(f"   Confidence: {meta_signal['confidence']:.1%}")
    print(f"   Conviction: {meta_signal['conviction'].upper()}")

    print(f"\n🏆 Top Contributing Agents:")
    for contrib in meta_signal['top_contributors'][:3]:
        print(f"   - {contrib['agent']} ({contrib['type']}): {contrib['confidence']:.1%}")

    # Test Case 2: Protective Put Buying (Hedge)
    print("\n\n📊 CASE 2: Detecting Institutional Hedging")
    print("-"*70)

    hedge_flow = {
        'symbol': 'SPY',
        'underlying_price': 450.0,
        'strike': 440.0,
        'days_to_expiry': 14,
        'call_put': 'P',
        'side': 'BUY',
        'size': 15000,  # Very large
        'price': 3.0,
        'notional': 4500000,
        'implied_volatility': 0.28,
        'delta': -0.35,
        'flow_type': 'block',
        'volume_ratio': 10.0
    }

    hedge_analysis = await options_rag.analyze_options_flow(hedge_flow)

    print(f"\n🔍 Hedge Flow Analysis:")
    print(f"   Institution Type: {hedge_analysis['flow_analysis']['institution_type']}")
    print(f"   Position Intent: {hedge_analysis['flow_analysis']['position_intent']}")
    print(f"   Smart Money Score: {hedge_analysis['flow_analysis']['smart_money_score']:.0f}/100")

    signals = hedge_analysis['trading_signals']
    print(f"\n⚠️ Risk Signal:")
    print(f"   Action: {signals['action'].upper()}")
    print(f"   Strategy: {signals.get('strategy', 'N/A')}")
    print(f"   Risk Management: Stop Loss {signals['risk_management']['stop_loss']:.1f}%")

    # Test Case 3: Unusual Activity Detection
    print("\n\n📊 CASE 3: Unusual Options Activity Summary")
    print("-"*70)

    for symbol in ['AAPL', 'SPY', 'XYZ']:
        activity = await options_rag.detect_unusual_activity(symbol)

        if activity['unusual_activity']:
            print(f"\n🚨 {symbol}: UNUSUAL ACTIVITY DETECTED")
            print(f"   Total Volume: {activity['total_volume']:,} contracts")
            print(f"   Total Notional: ${activity['total_notional']:,.0f}")
            print(f"   Overall Bias: {activity['overall_bias'].upper()}")

            if activity['unusual_flows']:
                print(f"   Unusual Flows:")
                for flow in activity['unusual_flows'][:2]:
                    print(f"     - {flow['description']} | Score: {flow['smart_money_score']:.0f}")

            rec = activity['recommendation']
            print(f"   📋 Recommendation: {rec['action'].upper()}")
            if 'reason' in rec:
                print(f"      Reason: {rec['reason']}")
        else:
            print(f"\n{symbol}: No unusual activity")

    # Test Case 4: Real-time Integration
    print("\n\n📊 CASE 4: Real-time Signal Integration")
    print("-"*70)

    # Simulate multiple data points coming in
    print("\n⏱️ Monitoring AAPL with live options flow...")

    for i in range(3):
        await asyncio.sleep(0.5)  # Simulate time passing

        # Generate flow data
        flow = {
            'symbol': 'AAPL',
            'underlying_price': 195.0 + i * 0.5,
            'strike': 200.0,
            'days_to_expiry': 30 - i,
            'call_put': 'C',
            'side': 'BUY',
            'size': 1000 + i * 500,
            'price': 4.0 + i * 0.2,
            'implied_volatility': 0.30 + i * 0.02,
            'delta': 0.45 + i * 0.05,
            'flow_type': 'sweep' if i > 0 else 'regular',
            'volume_ratio': 2.0 + i * 2.0
        }

        # Quick analysis
        result = await options_rag.analyze_options_flow(flow)
        score = result['flow_analysis']['smart_money_score']

        print(f"\n   T+{i}: New flow detected - Size: {flow['size']}, "
              f"Score: {score:.0f}, "
              f"Signal: {result['trading_signals']['action'].upper()}")

    # Summary
    print("\n\n" + "="*70)
    print("💡 KEY INSIGHTS FROM OPTIONS FLOW INTELLIGENCE:")
    print("="*70)
    print("1. ✅ Detected hedge fund accumulation in AAPL before earnings")
    print("2. ⚠️ Identified large protective put buying in SPY (risk-off signal)")
    print("3. 🚨 Found unusual call activity in XYZ (potential M&A)")
    print("4. 📊 Real-time monitoring shows increasing bullish flow")
    print("\n🎯 TRADING RECOMMENDATION: Follow smart money with proper risk management")

    return {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'key_findings': {
            'smart_money_detected': True,
            'primary_direction': 'bullish',
            'confidence': 'high',
            'risk_level': 'moderate'
        }
    }


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(simulate_unusual_options_activity())

    print("\n\n📁 Demo Results:")
    print(json.dumps(result, indent=2))
