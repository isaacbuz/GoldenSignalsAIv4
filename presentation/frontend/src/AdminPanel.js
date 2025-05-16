// AdminPanel.js
// Purpose: Provides the main administrative interface for GoldenSignalsAI. This panel allows administrators to monitor system health, view and manage agents, review logs, manage users, and access analytics. It is strictly separated from user-facing features and is only accessible to users with admin privileges. All sensitive actions are audit-logged and the panel is designed for real-time operational awareness and control.

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
  // State for application performance metrics
  // (CPU, memory, uptime, active requests)
  const [performance, setPerformance] = useState(null);
  // State for real-time logs
  const [logs, setLogs] = useState("");
  // State for list of agents
  const [agents, setAgents] = useState([]);
  // State for currently selected agent
  const [selectedAgent, setSelectedAgent] = useState(null);
  // State for details of selected agent
  const [agentDetails, setAgentDetails] = useState(null);
  // State for logged-in user
  const [user, setUser] = useState(null);
  // State for active tab in the panel
  const [tab, setTab] = useState("metrics");

  // Fetch application performance metrics on mount or when the user changes
  useEffect(() => {
    if (!user) return;
    fetch("/api/admin/performance")
      .then((res) => res.json())
      .then(setPerformance);
  }, [user]);

  // Poll real-time logs every 2 seconds for up-to-date system activity
  useEffect(() => {
    if (!user) return;
    const interval = setInterval(() => {
      fetch("/api/admin/logs")
        .then((res) => res.text())
        .then(setLogs);
    }, 2000);
    return () => clearInterval(interval);
  }, [user]);

  // Fetch the list of all registered agents for admin management
  useEffect(() => {
    if (!user) return;
    fetch("/api/admin/agents")
      .then((res) => res.json())
      .then(setAgents);
  }, [user]);

  // Fetch details for the currently selected agent whenever selection changes
  useEffect(() => {
    if (!user || !selectedAgent) return;
    fetch(`/api/admin/agents/${selectedAgent}`)
      .then((res) => res.json())
      .then(setAgentDetails);
  }, [selectedAgent, user]);

  // Listen for authentication state changes (login/logout)
  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      setUser(user);
    });
  }, []);

  // Handle user logout action
  const handleLogout = () => {
    signOut(auth);
  };

  // If not authenticated, show the login screen
  if (!user) {
    return <Login onLogin={setUser} />;
  }

  // Render the main admin panel UI, including onboarding, alerts, tabs, and tab content
  return (
    <div className="admin-panel">
      {/* Onboarding modal for new admins */}
      <AdminOnboardingModal />
      {/* Real-time system alerts and warnings */}
      <AdminAlerts />
      <div className="admin-header">
        <h2>Admin Panel</h2>
        <div className="admin-user-info">
          <span>Signed in as: {user.email || user.displayName}</span>
          <button onClick={handleLogout}>Logout</button>
        </div>
        {/* Tab navigation for different admin sections */}
        <div className="admin-tabs">
          <button className={tab === "metrics" ? "active" : ""} onClick={() => setTab("metrics")}>Metrics</button>
          <button className={tab === "agents" ? "active" : ""} onClick={() => setTab("agents")}>Agents</button>
          <button className={tab === "queue" ? "active" : ""} onClick={() => setTab("queue")}>Queue</button>
          <button className={tab === "users" ? "active" : ""} onClick={() => setTab("users")}>Users</button>
          <button className={tab === "analytics" ? "active" : ""} onClick={() => setTab("analytics")}>Analytics</button>
        </div>
      </div>
      <div className="admin-content">
        {/* Metrics Tab: Performance charts and logs */}
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
          </div>
        )}
        {/* Agents Tab: List and manage agents */}
        {tab === "agents" && (
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
                {/* Agent Controls (restart/disable) */}
                <div style={{ marginTop: "1rem" }}>
                  {agentDetails.name && (
                    <AdminAgentControls agentName={agentDetails.name} />
                  )}
                </div>
              </div>
            )}
          </section>
        )}
        {/* Queue Tab: Show task queue status */}
        {tab === "queue" && (
          <section>
            <AdminQueueStatus />
          </section>
        )}
        {/* Users Tab: User and role management */}
        {tab === "users" && (
          <section>
            <AdminUserManagement />
          </section>
        )}
        {/* Analytics Tab: Placeholder for future analytics features */}
        {tab === "analytics" && (
          <section>
            <p>Analytics coming soon...</p>
          </section>
        )}
      </div>
    </div>
  );
}

export default AdminPanel;
