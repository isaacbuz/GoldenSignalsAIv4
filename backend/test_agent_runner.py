import argparse
from backend.agents.agent_manager import AgentManager
from backend.agents.ml.sentiment_agent import SentimentAgent
from backend.agents.indicators.rsi_agent import RSIAgent
from backend.agents.indicators.macd_agent import MACDAgent
from backend.agents.indicators.ivrank_agent import IVRankAgent
from backend.agents.indicators.adaptive_oscillator_agent import AdaptiveOscillatorAgent
from backend.agents.indicators.knn_trend_agent import KNNTrendAgent
from backend.agents.indicators.ml_trend_channel_agent import MLTrendChannelAgent
from backend.agents.indicators.clustered_sr_agent import ClusteredSupportResistanceAgent
from datetime import datetime

AGENT_CLASSES = [
    SentimentAgent,
    RSIAgent,
    MACDAgent,
    IVRankAgent,
    AdaptiveOscillatorAgent,
    KNNTrendAgent,
    MLTrendChannelAgent,
    ClusteredSupportResistanceAgent,
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--date", type=str, required=False, help="Historical date (YYYY-MM-DD)")
    args = parser.parse_args()
    agents = [AgentClass(args.symbol) for AgentClass in AGENT_CLASSES]
    manager = AgentManager(agents)
    print(f"\nRunning agents for {args.symbol} on {args.date or datetime.now().date()}...")
    for agent in agents:
        result = agent.run()
        print(f"[{agent.__class__.__name__}] {result}")
    consensus = manager.vote()
    print("\nFinal Consensus Signal:")
    print(consensus)

if __name__ == "__main__":
    main()
