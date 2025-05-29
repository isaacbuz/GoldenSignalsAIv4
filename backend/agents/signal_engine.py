"""
Agent Orchestrator: runs all registered agents, aggregates their output, and emits final signals.
Usage:
    python signal_engine.py --symbol AAPL
"""
import argparse
from backend.agents.agent_manager import AgentManager

"""
Agent Orchestrator: runs all registered agents, aggregates their output, and emits final signals.
Now uses dynamic agent discovery via BaseSignalAgent registry.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    args = parser.parse_args()
    manager = AgentManager(args.symbol)
    consensus = manager.vote()
    print("\nFinal Consensus Signal:")
    print(consensus)

if __name__ == "__main__":
    main()

def generate_ai_signal(market_data, symbol="UNKNOWN", use_tv=True, agent_classes=None):
    """
    Run all agents and TradingView fetchers, normalize signals, and return unified bundle.
    Crowd sentiment is used to adjust confidence if available.
    """
    results = []
    agent_classes = agent_classes or []
    sentiment = None
    crowd_score = None
    adjustment = 1
    adjustment = 1 + (crowd_score * 0.3) if crowd_score else 1
    for AgentClass in agent_classes:
        agent = AgentClass()
        try:
            result = agent.run(market_data)
            norm = normalize_signal(result, default_source=AgentClass.__name__)
            base_conf = norm["confidence"]
            norm["confidence"] = round(base_conf * adjustment, 2)
            results.append(norm)
        except Exception as e:
            norm = normalize_signal({
                "name": AgentClass.__name__,
                "signal": "error",
                "confidence": 0,
                "explanation": f"Agent error: {str(e)}"
            }, default_source=AgentClass.__name__)
            results.append(norm)
    if use_tv:
        tv_signal = get_tradingview_signal(symbol)
        norm_tv = normalize_signal(tv_signal, default_source="TradingView")
        base_conf = norm_tv["confidence"]
        norm_tv["confidence"] = round(base_conf * adjustment, 2)
        results.append(norm_tv)
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "signals": results,
        "crowd_sentiment": sentiment
    }
