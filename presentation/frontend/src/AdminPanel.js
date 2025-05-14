import React, { useEffect, useState } from "react";
import { auth } from "./firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";
import Login from "./Login";
import AdminCharts from "./AdminCharts";
import AdminAgentHealth from "./AdminAgentHealth";
import AdminQueueStatus from "./AdminQueueStatus";
import AdminAlerts from "./AdminAlerts";
import AdminAgentControls from "./AdminAgentControls";
import AdminUserManagement from "./AdminUserManagement";
import AdminOnboardingModal from "./AdminOnboardingModal";
import "./AdminPanel.css";

function AdminPanel() {
  const [performance, setPerformance] = useState(null);
  const [logs, setLogs] = useState("");
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentDetails, setAgentDetails] = useState(null);

  // Fetch application performance metrics
  useEffect(() => {
    if (!user) return;
    fetch("/api/admin/performance")
      .then((res) => res.json())
      .then(setPerformance);
  }, [user]);

  // Fetch real-time logs (polling)
  useEffect(() => {
    if (!user) return;
    const interval = setInterval(() => {
      fetch("/api/admin/logs")
        .then((res) => res.text())
        .then(setLogs);
    }, 2000);
    return () => clearInterval(interval);
  }, [user]);

  // Fetch agent list
  useEffect(() => {
    if (!user) return;
    fetch("/api/admin/agents")
      .then((res) => res.json())
      .then(setAgents);
  }, [user]);

  // Fetch selected agent details
  useEffect(() => {
    if (!user || !selectedAgent) return;
    fetch(`/api/admin/agents/${selectedAgent}`)
      .then((res) => res.json())
      .then(setAgentDetails);
  }, [selectedAgent, user]);

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      setUser(user);
    });
  }, []);

  const handleLogout = () => {
    signOut(auth);
  };

  if (!user) {
    return <Login onLogin={setUser} />;
  }

  return (
    <div className="admin-panel">
      <AdminOnboardingModal />
      <AdminAlerts />
      <div className="admin-header">
        <h2>Admin Panel</h2>
        <div className="admin-user-info">
          <span>Signed in as: {user.email || user.displayName}</span>
          <button onClick={handleLogout}>Logout</button>
        </div>
        <div className="admin-tabs">
          <button className={tab === "metrics" ? "active" : ""} onClick={() => setTab("metrics")}>Metrics</button>
          <button className={tab === "agents" ? "active" : ""} onClick={() => setTab("agents")}>Agents</button>
          <button className={tab === "queue" ? "active" : ""} onClick={() => setTab("queue")}>Queue</button>
          <button className={tab === "users" ? "active" : ""} onClick={() => setTab("users")}>Users</button>
          <button className={tab === "analytics" ? "active" : ""} onClick={() => setTab("analytics")}>Analytics</button>
        </div>
      </div>
      <div className="admin-content">
        {tab === "metrics" && (
          <div className="admin-sections">
            <section>
              <AdminCharts />
            </section>
            <section>
              <h3>Application Performance</h3>
              {performance ? (
                <ul>
                  <li>CPU Usage: {performance.cpu}%</li>
                  <li>Memory Usage: {performance.memory} MB</li>
                  <li>Uptime: {performance.uptime} s</li>
                  <li>Active Requests: {performance.activeRequests}</li>
                </ul>
              ) : (
                <p>Loading...</p>
              )}
            </section>
            <section>
              <h3>Real-Time Logs</h3>
              <pre className="logs-box">{logs || "Loading..."}</pre>
            </section>
            </ul>
          ) : (
            <p>Loading...</p>
          )}
        </section>
        <section>
          <h3>Real-Time Logs</h3>
          <pre className="logs-box">{logs || "Loading..."}</pre>
        </section>
        <section>
          <h3>Agents</h3>
          <div className="agents-list">
            {agents.length === 0 ? (
              <p>No agents found.</p>
            ) : (
              <ul>
                {agents.map((agent) => (
                  <li
                    key={agent.name}
                    className={selectedAgent === agent.name ? "selected" : ""}
                    onClick={() => setSelectedAgent(agent.name)}
                  >
                    {agent.name} ({agent.status})
                  </li>
                ))}
              </ul>
            )}
          </div>
          {agentDetails && (
            <div className="agent-details">
              <h4>Agent: {agentDetails.name}</h4>
              <ul>
                <li>Status: {agentDetails.status}</li>
                <li>Type: {agentDetails.type}</li>
                <li>Last Activity: {agentDetails.lastActivity}</li>
                <li>Current Task: {agentDetails.currentTask}</li>
                <li>Success Rate: {agentDetails.successRate}%</li>
                <li>Errors: {agentDetails.errors}</li>
              </ul>
              <h5>Recent Work</h5>
              <ul>
                {agentDetails.recentWork.map((work, idx) => (
                  <li key={idx}>{work}</li>
                ))}
              </ul>
              {/* Agent Controls */}
              <div style={{ marginTop: "1rem" }}>
                {agentDetails.name && (
                  <AdminAgentControls agentName={agentDetails.name} />
                )}
              </div>
            </div>
          )}
        </section>
      {user && user.role === "admin" && (
        <section>
          <AdminUserManagement />
        </section>
      )}
      </div>
    </div>
  );
}

export default AdminPanel;
