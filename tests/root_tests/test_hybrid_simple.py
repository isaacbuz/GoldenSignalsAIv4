"""
Simple Hybrid System Demonstration
Shows key features without complex dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid __init__.py issues
from agents.orchestration.hybrid_orchestrator import HybridOrchestrator
from datetime import datetime

def main():
    print("\n" + "="*60)
    print("🚀 GOLDENSIGNALS AI - HYBRID SENTIMENT SYSTEM DEMO")
    print("="*60)
    
    try:
        # Create orchestrator
        print("\n📊 Creating Hybrid Orchestrator...")
        orchestrator = HybridOrchestrator(symbols=['AAPL', 'TSLA'])
        print(f"✅ Initialized with {len(orchestrator.agents)} hybrid agents")
        
        # Generate a signal
        print("\n🎯 Generating signal for AAPL...")
        signal = orchestrator.generate_signals_for_symbol('AAPL')
        
        print(f"\n📈 SIGNAL RESULT:")
        print(f"   Action: {signal['action']}")
        print(f"   Confidence: {signal['confidence']:.2%}")
        print(f"   Reasoning: {signal['metadata']['reasoning']}")
        
        # Show sentiment
        sentiment = signal['metadata']['market_sentiment']
        print(f"\n🎭 MARKET SENTIMENT:")
        print(f"   Overall: {sentiment['overall'].upper()}")
        print(f"   Confidence: {sentiment['confidence']:.2%}")
        
        # Show divergences
        divergences = signal['metadata']['divergence_analysis']
        print(f"\n🔄 DIVERGENCE ANALYSIS:")
        print(f"   Total Divergences: {divergences['count']}")
        if divergences['strong_divergences']:
            print(f"   Strong Divergences: {len(divergences['strong_divergences'])}")
        
        # Show some agent details
        print(f"\n🤖 AGENT SIGNALS:")
        breakdown = signal['metadata']['agent_breakdown']
        for agent, data in list(breakdown.items())[:3]:  # Show first 3 agents
            final = data['final']
            print(f"   {agent}: {final['action']} ({final['confidence']:.2%})")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 