from agents.agent_registry import AgentRegistry
from agents.meta.meta_ensembler import MetaEnsembler


class MetaSignalOrchestrator:
    def __init__(self, mode="swing"):
        self.registry = AgentRegistry(mode=mode)
        self.mode = mode

    def generate(self, symbol, market_data, sentiment_score, options_flow, strategy_profile="swing"):
        agents = self.registry

        # Agent runs
        base_outputs = agents.run_all(market_data)
        smart_money = agents.get("SmartMoneyAgent").run({"trades": options_flow})
        meta_input = {
            "signal": "bullish",  # placeholder
            "confidence": 0.8
        }

        # Ensemble
        ensembler = MetaEnsembler()
        result = ensembler.aggregate({
            name: output for name, output in base_outputs.items()
            if isinstance(output, dict) and "signal" in output
        })

        # Approval
        approval = agents.get("ApprovalAgent").run(result, {
            "regime": market_data.get("regime"),
            "smart_money": smart_money,
            "override": {"override": None}
        })

        # Notification
        notify = agents.get("NotifyAgent").run(symbol, result, {
            "source_agents": list(base_outputs.keys())
        })

        return {
            "meta_signal": result,
            "approval": approval,
            "smart_money": smart_money,
            "notified": notify,
            "agent_outputs": base_outputs,
            "strategy_profile": strategy_profile
        }
