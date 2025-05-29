import React, { useEffect, useState } from "react";

interface RetrainLog {
  agent: string;
  status: string;
  output: string;
  timestamp: string;
}

export default function AgentAnalyticsPanel() {
  const [logs, setLogs] = useState<RetrainLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    fetch("/api/agents/admin/retrain_logs")
      .then((res) => res.json())
      .then((data) => {
        setLogs(data.logs || []);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to fetch retrain logs");
        setLoading(false);
      });
  }, []);

  // Simple analytics: retrain count per agent, last retrain status, last retrain time
  const agentStats = logs.reduce<Record<string, { count: number; lastStatus: string; lastTime: string }>>((acc, log) => {
    if (!acc[log.agent]) acc[log.agent] = { count: 0, lastStatus: '', lastTime: '' };
    acc[log.agent].count += 1;
    if (!acc[log.agent].lastTime || log.timestamp > acc[log.agent].lastTime) {
      acc[log.agent].lastStatus = log.status;
      acc[log.agent].lastTime = log.timestamp;
    }
    return acc;
  }, {});

  return (
    <div style={{ marginTop: 32 }}>
      <h2>Agent Retrain Analytics</h2>
      {loading ? <div>Loading...</div> : error ? <div className="error">{error}</div> : (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
          <thead>
            <tr style={{ background: '#f0f0f0' }}>
              <th>Agent</th>
              <th>Retrain Count</th>
              <th>Last Status</th>
              <th>Last Retrain</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(agentStats).map(([agent, stat]) => (
              <tr key={agent}>
                <td>{agent}</td>
                <td>{stat.count}</td>
                <td>{stat.lastStatus}</td>
                <td>{new Date(stat.lastTime).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <h3 style={{ marginTop: 32 }}>Retrain History (Recent)</h3>
      <div style={{ maxHeight: 300, overflow: 'auto', background: '#f8f8f8', border: '1px solid #eee', borderRadius: 6, padding: 8 }}>
        {logs.slice().reverse().slice(0, 20).map((log, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <strong>{log.agent}</strong> [{log.status}] at {new Date(log.timestamp).toLocaleString()}
            <pre style={{ background: '#fff', padding: 8, borderRadius: 4, margin: 0 }}>{log.output.slice(0, 400) + (log.output.length > 400 ? '...': '')}</pre>
          </div>
        ))}
      </div>
    </div>
  );
}
