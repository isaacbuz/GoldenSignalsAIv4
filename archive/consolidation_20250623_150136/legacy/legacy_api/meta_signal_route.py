from fastapi import APIRouter, Request
from archive.legacy_backend_agents.registry import AgentRegistry
from archive.legacy_backend_agents.meta.meta_ensembler import MetaEnsembler
from archive.legacy_backend_agents.meta.feedback_reweighter import FeedbackReweighter
from archive.legacy_backend_agents.meta.news_override_agent import NewsOverrideAgent
from archive.legacy_backend_agents.meta.regime_detector_agent import RegimeDetectorAgent
from archive.legacy_backend_agents.edge.agent_memory_manager import AgentMemoryManager
from archive.legacy_backend_agents.edge.explanation_engine import ExplanationEngine
from archive.legacy_backend_agents.edge.smart_money_overlay_agent import SmartMoneyOverlayAgent

router = APIRouter()

@router.post("/api/meta_signal")
async def get_meta_signal(request: Request):
    payload = await request.json()
    market_data = payload.get("ohlcv", {})
    sentiment_score = payload.get("sentiment_score", 0.0)
    options_flow = payload.get("options_flow", {})
    symbol = payload.get("symbol", "UNKNOWN")

    # 1. Run all agents
    registry = AgentRegistry()
    agent_outputs = registry.run_all(market_data)

    # Track memory globally
    feedback = FeedbackReweighter()
    feedback_weights = {name: feedback.get_weight(name) for name in agent_outputs}

    # Personal tuning based on feedback + preferences
    auto_weights = AutoAgentTuner().compute_weights()

    regime = RegimeDetectorAgent().run(market_data)
    override = NewsOverrideAgent().run({"sentiment_score": sentiment_score})
    smart_money = SmartMoneyOverlayAgent().run(options_flow)

    ensembler = MetaEnsembler(agent_weights=auto_weights)
    result = ensembler.aggregate(agent_outputs)
    # Add strategy_bucket to meta_signal result if possible
    if result.get('sources'):
        # Use the most common bucket among contributing agents
        buckets = [agent_outputs[src].get('strategy_bucket', 'other') for src in result['sources']]
        from collections import Counter
        bucket = Counter(buckets).most_common(1)[0][0] if buckets else 'other'
        result['strategy_bucket'] = bucket
    else:
        result['strategy_bucket'] = 'other'

    gate = SignalGatekeeper(min_confidence=0.65, allowed_regimes=["normal", "low_volatility"])
    gate_result = gate.evaluate(result, {
        "regime": regime.get("regime"),
        "smart_money": smart_money,
        "override": override
    })

    # 8. Memory update (if outcome tracking is available)
    memory = AgentMemoryManager()
    for agent_name, output in agent_outputs.items():
        if "signal" in output and "outcome" in output:
            memory.record(agent_name, symbol, output["signal"], output["outcome"])

    # 9. Generate full explainability tree
    explainer = ExplanationEngine()
    full_explanation = explainer.generate(agent_outputs, result)

    return {
        "meta_signal": result,
        "regime": regime,
        "smart_money": smart_money,
        "override": override,
        "weights": weights,
        "explanation": full_explanation,
        "raw_agent_outputs": agent_outputs
    }
