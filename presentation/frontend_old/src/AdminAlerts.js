// AdminAlerts.js
// Purpose: Displays real-time system and agent alerts for GoldenSignalsAI administrators. Aggregates and presents alerts based on agent health, queue status, and anomaly detection. Polls backend endpoints for up-to-date alerts and allows editing of alert thresholds. Designed to keep admins aware of critical issues and system health.

import API_URL from './config';
import React, { useEffect, useState } from 'react';
import "./AdminPanel.css";

// Generate alerts based on agent health data
function getAgentAlerts(health) {
  const alerts = [];
  Object.entries(health).forEach(([name, info]) => {
    if (info.status !== "active") {
      alerts.push({
        type: "danger",
        message: `Agent ${name} is ${info.status}`,
      });
    } else if (info.error_rate > 0.05) {
      alerts.push({
        type: "warning",
        message: `Agent ${name} has high error rate (${(info.error_rate * 100).toFixed(2)}%)`,
      });
    }
  });
  return alerts;
}

// Generate alerts based on queue status
function getQueueAlerts(queue) {
  const alerts = [];
  if (queue && queue.depth > 10) {
    alerts.push({
      type: "warning",
      message: `Queue depth is high (${queue.depth})`,
    });
  }
  if (queue && queue.active < 1) {
    alerts.push({
      type: "danger",
      message: `No active workers!`,
    });
  }
  return alerts;
}

function AdminAlerts() {
  // State for agent health, queue status, and alert messages
  const [health, setHealth] = useState({});
  const [queue, setQueue] = useState(null);
  const [alerts, setAlerts] = useState([]);
  // State for alert thresholds and anomaly alerts
  const [thresholds, setThresholds] = useState({});
  const [anomalyAlerts, setAnomalyAlerts] = useState([]);
  const [editing, setEditing] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");

  // Poll agent health and queue status every 5 seconds
  useEffect(() => {
    function fetchData() {
      fetch(`${API_URL}/api/admin/agents/health`)
        .then((res) => res.json())
        .then(setHealth);
      fetch(`${API_URL}/api/admin/queue`)
        .then((res) => res.json())
        .then(setQueue);
    }
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch alert thresholds once on mount
  useEffect(() => {
    fetch(`${API_URL}/api/admin/alert_thresholds`)
      .then(res => res.json())
      .then(setThresholds);
  }, []);

  // Poll anomaly alerts every 10 seconds
  useEffect(() => {
    let mounted = true;
    const fetchAnomalyAlerts = () => {
      fetch(`${API_URL}/api/admin/anomaly_check`)
        .then((res) => res.json())
        .then((data) => {
          if (mounted && data && data.alerts) setAnomalyAlerts(data.alerts);
        })
        .catch(() => {});
    };
    fetchAnomalyAlerts();
    const interval = setInterval(fetchAnomalyAlerts, 10000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  // Aggregate all alerts into a single array for display
  useEffect(() => {
    const agentAlerts = getAgentAlerts(health);
    const queueAlerts = getQueueAlerts(queue);
    setAlerts([...anomalyAlerts, ...agentAlerts, ...queueAlerts]);
  }, [health, queue, anomalyAlerts]);

  // Handlers for editing and saving alert thresholds
  const handleEdit = () => setEditing(true);
  const handleChange = (k, v) => setThresholds({ ...thresholds, [k]: v });
  const handleSave = async () => {
    setSaveMsg("");
    await fetch(`${API_URL}/api/admin/alert_thresholds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(thresholds)
    });
    setEditing(false);
    setSaveMsg("Thresholds updated.");
  };

  // If no alerts, render nothing
  if (!alerts.length) return null;

  // Render all current alerts as styled boxes
  return (
    <div className="admin-alerts">
      {alerts.map((alert, idx) => (
        <div
          key={idx}
          className={`alert alert-${alert.type || "warning"}`}
          style={{
            background: (alert.type || "warning") === "danger" ? "#ff5252" : "#f8b400",
            color: "#fff",
            margin: "1rem 0",
            padding: "1rem",
            borderRadius: "8px",
            fontWeight: "bold",
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          }}
        >
          {alert.message}
        </div>
      ))}
    </div>
  );
}

export default AdminAlerts;
