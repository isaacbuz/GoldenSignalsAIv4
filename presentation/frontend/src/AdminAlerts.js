import React, { useEffect, useState } from "react";
import "./AdminPanel.css";

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
  const [health, setHealth] = useState({});
  const [queue, setQueue] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    function fetchData() {
      fetch("/api/admin/agents/health")
        .then((res) => res.json())
        .then(setHealth);
      fetch("/api/admin/queue")
        .then((res) => res.json())
        .then(setQueue);
    }
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetch("/api/admin/alert_thresholds")
      .then(res => res.json())
      .then(setThresholds);
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchAnomalyAlerts = () => {
      fetch("/api/admin/anomaly_check")
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

  useEffect(() => {
    const agentAlerts = getAgentAlerts(health);
    const queueAlerts = getQueueAlerts(queue);
    setAlerts([...anomalyAlerts, ...agentAlerts, ...queueAlerts]);
  }, [health, queue, anomalyAlerts]);

  const handleEdit = () => setEditing(true);
  const handleChange = (k, v) => setThresholds({ ...thresholds, [k]: v });
  const handleSave = async () => {
    setSaveMsg("");
    await fetch("/api/admin/alert_thresholds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(thresholds)
    });
    setEditing(false);
    setSaveMsg("Thresholds updated.");
  };

  if (!alerts.length) return null;

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
