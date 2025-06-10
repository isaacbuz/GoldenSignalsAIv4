class ExplanationEngine:
    def generate(self, agent_outputs: dict, meta_signal: dict) -> dict:
        explanation = {
            "final_signal": meta_signal["signal"],
            "confidence": meta_signal["confidence"],
            "details": []
        }

        for name, output in agent_outputs.items():
            if "signal" in output:
                explanation["details"].append({
                    "agent": name,
                    "signal": output["signal"],
                    "confidence": output.get("confidence", 0.5),
                    "explanation": output.get("explanation", "")
                })

        explanation["details"] = sorted(explanation["details"], key=lambda x: -x["confidence"])
        return explanation
