import React, { useEffect, useState } from "react";
import "./AdminPanel.css";

function statusColor(status) {
  if (status === "active") return "#6be6c1";
  if (status === "inactive") return "#ff5252";
  return "#f8b400";
}

function AdminAgentHealth() {
  const [health, setHealth] = useState({});

  useEffect(() => {
    fetch("/api/admin/agents/health")
      .then((res) => res.json())
      .then(setHealth);
    const interval = setInterval(() => {
      fetch("/api/admin/agents/health")
        .then((res) => res.json())
        .then(setHealth);
    }, 5000); // update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const agents = Object.entries(health);

  if (!agents.length) return <p>Loading agent health...</p>;

  return (
    <div className="agent-health">
      <h4>Agent Health & Heartbeat</h4>
      <table className="agent-health-table">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Status</th>
            <th>Last Heartbeat</th>
            <th>Latency (s)</th>
            <th>Error Rate</th>
          </tr>
        </thead>
        <tbody>
          {agents.map(([name, info]) => (
            <tr key={name}>
              <td>{name}</td>
              <td style={{ color: statusColor(info.status) }}>{info.status}</td>
              <td>{new Date(info.last_heartbeat * 1000).toLocaleTimeString()}</td>
              <td>{info.latency}</td>
              <td>{(info.error_rate * 100).toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default AdminAgentHealth;
