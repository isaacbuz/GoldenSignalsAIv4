"""
Test Hybrid Sentiment System
Demonstrates the full capabilities of the hybrid agent architecture
"""

import asyncio
import time
from datetime import datetime
import json

# Import the hybrid system components
from agents.orchestration.hybrid_orchestrator import HybridOrchestrator

def print_signal_analysis(signal: dict):
    """Pretty print signal analysis"""
    print("\n" + "="*80)
    print(f"📊 SIGNAL ANALYSIS for {signal['symbol']}")
    print("="*80)
    
    # Main signal
    print(f"\n🎯 Final Signal: {signal['action']} (Confidence: {signal['confidence']:.2%})")
    print(f"📝 Reasoning: {signal['metadata']['reasoning']}")
    
    # Market sentiment
    sentiment = signal['metadata']['market_sentiment']
    print(f"\n🎭 Market Sentiment: {sentiment['overall'].upper()}")
    print(f"   Confidence: {sentiment['confidence']:.2%}")
    print(f"   Agreement: {sentiment.get('agreement', 0):.2%}")
    
    # Divergence analysis
    divergence = signal['metadata']['divergence_analysis']
    print(f"\n🔄 Divergence Analysis:")
    print(f"   Total Divergences: {divergence['count']}")
    print(f"   Strong Divergences: {len(divergence['strong_divergences'])}")
    if divergence['opportunities']:
        print(f"   Opportunities: {', '.join(divergence['opportunities'])}")
    
    # Agent breakdown
    print(f"\n🤖 Agent Breakdown:")
    breakdown = signal['metadata']['agent_breakdown']
    
    for agent, data in breakdown.items():
        final = data['final']
        components = data.get('components', {})
        sentiment = data.get('sentiment', {})
        
        print(f"\n   {agent}:")
        print(f"   ├─ Final: {final['action']} ({final['confidence']:.2%})")
        
        if components:
            # Show independent vs collaborative
            ind = components.get('independent', {})
            col = components.get('collaborative', {})
            
            if ind and col:
                print(f"   ├─ Independent: {ind['action']} ({ind['confidence']:.2%}) - {ind.get('sentiment', 'neutral')}")
                print(f"   ├─ Collaborative: {col['action']} ({col['confidence']:.2%}) - {col.get('sentiment', 'neutral')}")
                
                # Show divergence if any
                div = components.get('divergence', {})
                if div and div.get('type') != 'none':
                    print(f"   ├─ ⚠️  Divergence: {div['type']}")
        
        print(f"   └─ Reasoning: {final['reasoning'][:100]}...")

def test_basic_functionality():
    """Test basic hybrid system functionality"""
    print("\n🧪 TESTING BASIC HYBRID FUNCTIONALITY")
    print("="*80)
    
    # Create orchestrator
    orchestrator = HybridOrchestrator(symbols=['AAPL', 'TSLA', 'GOOGL'])
    
    # Test single symbol
    print("\n1️⃣ Testing single symbol analysis...")
    signal = orchestrator.generate_signals_for_symbol('AAPL')
    print_signal_analysis(signal)
    
    # Test sentiment analysis
    print("\n2️⃣ Testing sentiment analysis...")
    sentiment = orchestrator.get_sentiment_analysis('AAPL')
    
    print("\n📊 Sentiment Summary:")
    print(f"Market Sentiment: {sentiment['market_sentiment']}")
    print(f"\nAgent Sentiments:")
    
    for agent, data in sentiment['agent_sentiments'].items():
        current = data['current']
        print(f"  {agent}: Independent={current.get('independent', 'N/A')}, "
              f"Collaborative={current.get('collaborative', 'N/A')}, "
              f"Final={current.get('final', 'N/A')}")
    
    # Test performance dashboard
    print("\n3️⃣ Testing performance dashboard...")
    dashboard = orchestrator.get_performance_dashboard()
    
    print("\n📈 Performance Metrics:")
    print(f"Total Signals: {dashboard['market_performance']['total_signals']}")
    print(f"Total Divergences: {dashboard['market_performance']['total_divergences']}")
    print(f"Average Divergence Rate: {dashboard['market_performance']['average_divergence_rate']:.2%}")

def test_divergence_scenarios():
    """Test how system handles divergences"""
    print("\n\n🧪 TESTING DIVERGENCE SCENARIOS")
    print("="*80)
    
    orchestrator = HybridOrchestrator(symbols=['NVDA'])
    
    # Generate multiple signals to create divergences
    print("\n🔄 Generating signals to test divergence handling...")
    
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        signal = orchestrator.generate_signals_for_symbol('NVDA')
        
        # Check for divergences
        divergences = signal['metadata']['divergence_analysis']
        if divergences['count'] > 0:
            print(f"✅ Found {divergences['count']} divergences!")
            
            for div in divergences.get('strong_divergences', []):
                print(f"   Strong divergence in {div['agent']}: "
                      f"Independent={div['independent']} vs Collaborative={div['collaborative']}")
        else:
            print("❌ No divergences detected")
        
        time.sleep(1)  # Small delay between iterations

def test_market_sentiment_evolution():
    """Test how market sentiment evolves over time"""
    print("\n\n🧪 TESTING MARKET SENTIMENT EVOLUTION")
    print("="*80)
    
    orchestrator = HybridOrchestrator(symbols=['MSFT', 'META', 'AMZN'])
    
    print("\n📊 Tracking sentiment evolution over multiple signals...")
    
    sentiment_history = []
    
    for i in range(5):
        print(f"\n--- Time {i+1} ---")
        
        # Generate signals for all symbols
        for symbol in orchestrator.symbols:
            signal = orchestrator.generate_signals_for_symbol(symbol)
            
            # Get sentiment
            sentiment = orchestrator.get_sentiment_analysis()
            market_sentiment = sentiment['market_sentiment']
            
            sentiment_history.append({
                'iteration': i+1,
                'timestamp': datetime.now().isoformat(),
                'sentiment': market_sentiment['overall'],
                'confidence': market_sentiment['confidence'],
                'breakdown': market_sentiment['breakdown']
            })
            
            print(f"{symbol}: {signal['action']} | Sentiment: {market_sentiment['overall']}")
        
        # Show sentiment trends
        trends = sentiment['sentiment_trends']
        if trends['sentiment_shifts']:
            print(f"\n🔄 Sentiment Shifts Detected:")
            for shift in trends['sentiment_shifts'][-3:]:  # Show last 3 shifts
                print(f"   {shift['agent']}: {shift['from']} → {shift['to']}")
        
        time.sleep(1)
    
    # Analyze sentiment evolution
    print("\n\n📈 Sentiment Evolution Summary:")
    for record in sentiment_history:
        print(f"T{record['iteration']}: {record['sentiment']} "
              f"(confidence: {record['confidence']:.2%})")

def test_performance_simulation():
    """Simulate performance tracking and weight adjustment"""
    print("\n\n🧪 TESTING PERFORMANCE SIMULATION")
    print("="*80)
    
    orchestrator = HybridOrchestrator(symbols=['SPY'])
    
    print("\n🎯 Simulating signal outcomes and performance tracking...")
    
    # Generate initial signals
    initial_weights = {}
    for agent_name in orchestrator.agents:
        agent = orchestrator.agents[agent_name]
        metrics = agent.get_performance_metrics()
        initial_weights[agent_name] = metrics['current_weights']
    
    print("\nInitial Agent Weights:")
    for agent, weights in initial_weights.items():
        print(f"  {agent}: Independent={weights['independent']:.2f}, "
              f"Collaborative={weights['collaborative']:.2f}")
    
    # Simulate outcomes
    print("\n📊 Simulating 10 trading signals with random outcomes...")
    
    for i in range(10):
        # Generate signal
        signal = orchestrator.generate_signals_for_symbol('SPY')
        
        # Simulate outcome (simplified - in reality would track actual performance)
        import random
        
        # Bias outcome based on confidence
        if signal['confidence'] > 0.7:
            outcome = 1.0 if random.random() < 0.7 else -1.0
        elif signal['confidence'] > 0.5:
            outcome = 1.0 if random.random() < 0.5 else -1.0
        else:
            outcome = 1.0 if random.random() < 0.3 else -1.0
        
        # Update performance for each agent
        for agent_name in orchestrator.agents:
            orchestrator.update_agent_performance(agent_name, f"signal_{i}", outcome)
        
        print(f"\nSignal {i+1}: {signal['action']} "
              f"(confidence: {signal['confidence']:.2%}) → "
              f"Outcome: {'✅ WIN' if outcome > 0 else '❌ LOSS'}")
    
    # Check updated weights
    print("\n\nUpdated Agent Weights After Performance Tracking:")
    for agent_name in orchestrator.agents:
        agent = orchestrator.agents[agent_name]
        metrics = agent.get_performance_metrics()
        weights = metrics['current_weights']
        
        print(f"\n{agent_name}:")
        print(f"  Independent: {initial_weights[agent_name]['independent']:.2f} → {weights['independent']:.2f}")
        print(f"  Collaborative: {initial_weights[agent_name]['collaborative']:.2f} → {weights['collaborative']:.2f}")
        print(f"  Divergence Bonus: {weights.get('divergence_bonus', 0):.2f}")

def test_comprehensive_report():
    """Generate a comprehensive system report"""
    print("\n\n🧪 GENERATING COMPREHENSIVE SYSTEM REPORT")
    print("="*80)
    
    orchestrator = HybridOrchestrator(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'])
    
    # Generate signals for all symbols
    print("\n📊 Generating signals for all symbols...")
    all_signals = []
    
    for symbol in orchestrator.symbols:
        signal = orchestrator.generate_signals_for_symbol(symbol)
        all_signals.append(signal)
        print(f"✅ {symbol}: {signal['action']} ({signal['confidence']:.2%})")
    
    # Get comprehensive reports
    sentiment_report = orchestrator.get_sentiment_analysis()
    performance_report = orchestrator.get_performance_dashboard()
    
    # Summary Report
    print("\n\n" + "="*80)
    print("📊 GOLDENSIGNALS AI - HYBRID SYSTEM REPORT")
    print("="*80)
    
    print(f"\n🏛️ System Configuration:")
    print(f"  Active Agents: {len(orchestrator.agents)}")
    print(f"  Monitored Symbols: {len(orchestrator.symbols)}")
    print(f"  Data Bus: {'✅ Active' if orchestrator.data_bus else '❌ Inactive'}")
    
    print(f"\n📈 Market Overview:")
    market_sentiment = sentiment_report['market_sentiment']
    print(f"  Overall Sentiment: {market_sentiment['overall'].upper()}")
    print(f"  Sentiment Confidence: {market_sentiment['confidence']:.2%}")
    print(f"  Participating Agents: {market_sentiment['agent_count']}")
    
    print(f"\n🎯 Signal Summary:")
    buy_signals = sum(1 for s in all_signals if s['action'] == 'BUY')
    sell_signals = sum(1 for s in all_signals if s['action'] == 'SELL')
    hold_signals = sum(1 for s in all_signals if s['action'] == 'HOLD')
    
    print(f"  BUY Signals: {buy_signals} ({buy_signals/len(all_signals)*100:.0f}%)")
    print(f"  SELL Signals: {sell_signals} ({sell_signals/len(all_signals)*100:.0f}%)")
    print(f"  HOLD Signals: {hold_signals} ({hold_signals/len(all_signals)*100:.0f}%)")
    
    avg_confidence = sum(s['confidence'] for s in all_signals) / len(all_signals)
    print(f"  Average Confidence: {avg_confidence:.2%}")
    
    print(f"\n🔄 Divergence Analysis:")
    total_divergences = sum(s['metadata']['divergence_analysis']['count'] for s in all_signals)
    print(f"  Total Divergences: {total_divergences}")
    print(f"  Divergence Rate: {performance_report['market_performance']['average_divergence_rate']:.2%}")
    
    print(f"\n🤖 Agent Performance Highlights:")
    for agent_name, data in performance_report['agents'].items():
        if data['divergence_rate'] > 0.2:
            print(f"  ⚠️  {agent_name}: High divergence rate ({data['divergence_rate']:.2%})")
    
    print(f"\n💡 Key Insights:")
    
    # Sentiment momentum
    trends = sentiment_report['sentiment_trends']
    if trends['bullish_momentum'] > trends['bearish_momentum']:
        print(f"  ↗️  Bullish momentum building ({trends['bullish_momentum']} agents)")
    elif trends['bearish_momentum'] > trends['bullish_momentum']:
        print(f"  ↘️  Bearish momentum building ({trends['bearish_momentum']} agents)")
    
    # Divergence opportunities
    high_div_symbols = [s['symbol'] for s in all_signals 
                       if s['metadata']['divergence_analysis']['count'] >= 2]
    if high_div_symbols:
        print(f"  🔍 High divergence symbols: {', '.join(high_div_symbols)}")
    
    # Confidence distribution
    high_conf_signals = [s for s in all_signals if s['confidence'] > 0.8]
    if high_conf_signals:
        print(f"  🎯 High confidence signals ({len(high_conf_signals)}): " +
              ', '.join([f"{s['symbol']}({s['action']})" for s in high_conf_signals]))
    
    print("\n" + "="*80)
    print("✅ Report Generation Complete")

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 GOLDENSIGNALS AI - HYBRID SENTIMENT SYSTEM TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run tests
        test_basic_functionality()
        test_divergence_scenarios()
        test_market_sentiment_evolution()
        test_performance_simulation()
        test_comprehensive_report()
        
        print("\n\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main() 