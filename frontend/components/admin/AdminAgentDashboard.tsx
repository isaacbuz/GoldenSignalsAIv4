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

import AgentConfigPanel from './AgentConfigPanel';
import AgentRetrainPanel from './AgentRetrainPanel';

function Toast({ message, type, onClose }: { message: string; type: 'success' | 'error'; onClose: () => void }) {
  return (
    <div style={{
      position: 'fixed', bottom: 32, right: 32, zIndex: 2000, background: type === 'success' ? '#4caf50' : '#d32f2f', color: 'white', padding: '16px 32px', borderRadius: 8, boxShadow: '0 4px 16px #0002', fontWeight: 600
    }}>
      {message}
      <button onClick={onClose} style={{ marginLeft: 16, background: 'none', color: 'white', border: 'none', fontWeight: 700, cursor: 'pointer' }}>×</button>
    </div>
  );
}

function useAgentEventNotifications(onEvent: (event: any) => void) {
  React.useEffect(() => {
    const ws = new WebSocket(
      (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws/agent-events'
    );
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data);
        onEvent(event);
      } catch {}
    };
    return () => ws.close();
  }, [onEvent]);
}

export default function AdminAgentDashboard() {
  const [symbol, setSymbol] = useState("AAPL");
  const [signals, setSignals] = useState<AgentSignal[]>([]);
  const [consensus, setConsensus] = useState<Consensus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [retrainAgent, setRetrainAgent] = useState<string | null>(null);
  const [retrainOutput, setRetrainOutput] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string, type: 'success' | 'error' } | null>(null);

  // Demo: agent config as JSON (replace with backend fetch in future)
  const agentConfigs: Record<string, any> = {
    NewsHeadlineAgent: { threshold: 0.5, source: "newsapi" },
    TradingViewSignalAgent: { interval: "1h", symbol },
    UserFeedbackAgent: { minVotes: 3 },
  };

  // Real-time retrain notifications
  useAgentEventNotifications((event) => {
    if (event.type === 'retrain') {
      setToast({
        message: `Retrain event: ${event.agent} — ${event.status}`,
        type: event.status === 'retrain complete' ? 'success' : 'error',
      });
    }
  });

  // Retrain handler
  const handleRetrain = async (agent: string) => {
    setRetrainOutput(null);
    setRetrainAgent(agent);
    try {
      const res = await fetch('/api/agents/admin/retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent })
      });
      const data = await res.json();
      if (data.status === 'retrain complete') {
        setToast({ message: `Retrain complete for ${agent}`, type: 'success' });
      } else if (data.status === 'not supported') {
        setToast({ message: `Retrain not supported for ${agent}`, type: 'error' });
      } else {
        setToast({ message: `Retrain failed for ${agent}`, type: 'error' });
      }
      setRetrainOutput(data.output || JSON.stringify(data));
    } catch (e) {
      setToast({ message: `Retrain error for ${agent}`, type: 'error' });
      setRetrainOutput(String(e));
    }
  };


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

  return (
    <div className="admin-agent-dashboard">
      <h2>Admin: Agent Dashboard</h2>
      <div style={{ marginBottom: 16 }}>
        <label>Symbol: </label>
        <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} />
      </div>
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
          <table className="agent-signal-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
  <thead>
    <tr style={{ background: '#f0f0f0' }}>
      <th>Agent</th>
      <th>Action</th>
      <th>Confidence</th>
      <th>Explanation</th>
      <th>Admin</th>
    </tr>
  </thead>
  <tbody>
              {signals.map((s) => (
  <tr key={s.agent}>
    <td>{s.agent}</td>
    <td>{s.action}</td>
    <td>{s.confidence}</td>
    <td>{s.explanation || "-"}</td>
    <td>
      <button onClick={() => setSelectedAgent(s.agent)}>Config</button>
      <button onClick={() => handleRetrain(s.agent)}>Retrain</button>
    </td>
  </tr>
))}

            </tbody>
          </table>
        </>
      )}
    {selectedAgent && (
      <div className="modal" style={{ position: 'fixed', top: 60, left: 0, right: 0, background: '#fff', zIndex: 1000, padding: 24, border: '1px solid #aaa' }}>
        <AgentConfigPanel
          agent={selectedAgent}
          initialConfig={agentConfigs[selectedAgent] || {}}
          onConfigUpdated={() => setSelectedAgent(null)}
        />
        <button onClick={() => setSelectedAgent(null)} style={{ marginTop: 8 }}>Close</button>
      </div>
    )}
    {retrainAgent && (
      <div className="modal" style={{ position: 'fixed', top: 60, left: 0, right: 0, background: '#fff', zIndex: 1000, padding: 24, border: '1px solid #aaa' }}>
        <h3>Retrain Output for {retrainAgent}</h3>
        <pre style={{ maxHeight: 300, overflow: 'auto', background: '#f8f8f8', padding: 12, borderRadius: 6 }}>{retrainOutput || 'Retraining...'}</pre>
        <button onClick={() => { setRetrainAgent(null); setRetrainOutput(null); }} style={{ marginTop: 8 }}>Close</button>
      </div>
    )}
    {toast && (
      <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
    )}
  </div>
  );
}
