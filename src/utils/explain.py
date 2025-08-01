from typing import Any, Dict


class ExplanationEngine:
    """Generates an explanation tree for a consensus signal.

    The structure is intentionally simple JSON so it can be broadcast over
    WebSocket and rendered by the React dashboard without additional parsing
    logic on the backend.
    """

    def generate(
        self, agent_outputs: Dict[str, Dict[str, Any]], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        explanation = {
            "final_signal": meta.get("signal_type") or meta.get("signal"),
            "confidence": meta.get("confidence"),
            "details": [],
        }

        for name, output in agent_outputs.items():
            if "signal" in output:
                explanation["details"].append(
                    {
                        "agent": name,
                        "signal": output.get("signal"),
                        "confidence": output.get("confidence", 0.5),
                        "strength": output.get("strength"),
                        "explanation": output.get("explanation", ""),
                    }
                )

        explanation["details"] = sorted(
            explanation["details"], key=lambda x: x.get("confidence", 0), reverse=True
        )
        return explanation
