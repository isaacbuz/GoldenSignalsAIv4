# backend/agents/blender_agent.py

class BlenderAgent:
    def blend(self, agent_outputs):
        scores = []
        details = []

        for name, agent in agent_outputs.items():
            if "confidence" in agent and agent.get("signal") not in ["neutral", "unknown"]:
                scores.append(agent["confidence"])
            if agent.get("explanation"):
                details.append(f"{name}: {agent['explanation']}")

        avg_conf = sum(scores) / len(scores) if scores else 50

        return {
            "entry": 148,
            "exit": 160,
            "confidence": int(avg_conf),
            "strategy": "Vertical Call Spread" if avg_conf > 60 else "Straddle",
            "explanation": " | ".join(details),
            "legs": [
                { "type": "call", "strike": 150, "action": "buy" },
                { "type": "call", "strike": 160, "action": "sell" }
            ]
        }
