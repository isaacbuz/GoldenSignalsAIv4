// AdminAgentHealth.js
// Purpose: Displays the health and heartbeat status of all agents in GoldenSignalsAI for admin users. Polls the backend for up-to-date agent health, including status, last heartbeat, latency, and error rate. Designed for real-time monitoring and troubleshooting by administrators.

import React, { useEffect, useState } from "react";
import "./AdminPanel.css";

// Utility function to determine color based on agent status
function statusColor(status) {
  if (status === "active") return "#6be6c1";
  if (status === "inactive") return "#ff5252";
  return "#f8b400";
}

function AdminAgentHealth() {
  // State for storing agent health data from the backend
  const [health, setHealth] = useState({});

  useEffect(() => {
    // Fetch agent health once on mount
    fetch("/api/admin/agents/health")
      .then((res) => res.json())
      .then(setHealth);
    // Set up polling to fetch agent health every 5 seconds
    const interval = setInterval(() => {
      fetch("/api/admin/agents/health")
        .then((res) => res.json())
        .then(setHealth);
    }, 5000); // update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  // Convert health object to array for rendering
  const agents = Object.entries(health);

  // Show loading state if no agent data yet
  if (!agents.length) return <p>Loading agent health...</p>;

  // Render a table of agent health and heartbeat info
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
              {/* Status color-coded for quick visual diagnosis */}
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
