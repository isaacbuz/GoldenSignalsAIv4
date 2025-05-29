import React, { useState } from "react";

interface AgentRetrainPanelProps {
  agent: string;
}

export default function AgentRetrainPanel({ agent }: AgentRetrainPanelProps) {
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleRetrain = async () => {
    setLoading(true);
    setStatus("");
    try {
      const res = await fetch("/api/agents/admin/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent }),
      });
      const data = await res.json();
      setStatus(data.status || "Retrain triggered");
    } catch (e) {
      setStatus("Failed to trigger retrain");
    }
    setLoading(false);
  };

  return (
    <div className="agent-retrain-panel">
      <h3>Retrain Agent: {agent}</h3>
      <button onClick={handleRetrain} disabled={loading}>
        {loading ? "Retraining..." : "Trigger Retrain"}
      </button>
      {status && <div className="status">{status}</div>}
    </div>
  );
}
