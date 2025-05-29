import React, { useState, useEffect } from "react";

interface AgentSignal {
  symbol: string;
  action: string;
  confidence: number;
  agent: string;
  explanation?: string;
}

interface Consensus {
  symbol: string;
  action: string;
  confidence: number;
  sources: string[];
}

export default function AgentConsensusPanel({ symbol }: { symbol: string }) {
  const [signals, setSignals] = useState<AgentSignal[]>([]);
  const [consensus, setConsensus] = useState<Consensus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    setError("");
    fetch(`/api/agents/signals?symbol=${symbol}`)
      .then((res) => res.json())
      .then((data) => {
        setSignals(data.signals || []);
        setConsensus(data.consensus || null);
        setLoading(false);
      })
      .catch((e) => {
        setError("Failed to load agent signals.");
        setLoading(false);
      });
  }, [symbol]);

  if (!symbol) return null;

  return (
    <div className="agent-consensus-panel">
      <h3>Agent Consensus & Explanations</h3>
      {loading ? (
        <div>Loading...</div>
      ) : error ? (
        <div className="error">{error}</div>
      ) : (
        <>
          {consensus && (
            <div className="consensus-summary">
              <strong>Consensus:</strong> {consensus.action} ({consensus.confidence}%)
              <br />
              <span>Sources: {consensus.sources.join(", ")}</span>
            </div>
          )}
          <ul className="agent-signal-list">
            {signals.map((s) => (
              <li key={s.agent}>
                <strong>{s.agent}:</strong> {s.action} ({s.confidence}%)
                {s.explanation && (
                  <div className="explanation">{s.explanation}</div>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
