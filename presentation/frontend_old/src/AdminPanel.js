// AdminPanel.js
// Purpose: Provides the main administrative interface for GoldenSignalsAI. This panel allows administrators to monitor system health, view and manage agents, review logs, manage users, and access analytics. It is strictly separated from user-facing features and is only accessible to users with admin privileges. All sensitive actions are audit-logged and the panel is designed for real-time operational awareness and control.

import API_URL from './config';
import React, { useState, useEffect, Suspense } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Avatar from '@mui/material/Avatar';
import Divider from '@mui/material/Divider';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import AdminUserManagement from './AdminUserManagement';
import AdminAgentHealth from './AdminAgentHealth';
import AdminAlerts from './AdminAlerts';
import AdminCharts from './AdminCharts';
import AdminBacktest from './AdminBacktest';
import AdminQueueStatus from './AdminQueueStatus';
import { auth } from "./firebase";
import { onAuthStateChanged, signOut } from "firebase/auth";
import Login from "./Login";


import AdminAgentControls from "./AdminAgentControls";

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
    fetch(`${API_URL}/api/admin/performance`)
      .then((res) => res.json())
      .then(setPerformance);
  }, [user]);

  // Poll real-time logs every 2 seconds for up-to-date system activity
  useEffect(() => {
    if (!user) return;
    const interval = setInterval(() => {
      fetch(`${API_URL}/api/admin/logs`)
        .then((res) => res.text())
        .then(setLogs);
    }, 2000);
    return () => clearInterval(interval);
  }, [user]);

  // Fetch the list of all registered agents for admin management
  useEffect(() => {
    if (!user) return;
    fetch(`${API_URL}/api/admin/agents`)
      .then((res) => res.json())
      .then(setAgents);
  }, [user]);

  // Fetch details for the currently selected agent whenever selection changes
  useEffect(() => {
    if (!user || !selectedAgent) return;
    fetch(`${API_URL}/api/admin/agents/${selectedAgent}`)
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
  <Box className="admin-panel" sx={{ bgcolor: 'background.default', minHeight: '100vh', p: { xs: 1, md: 3 } }}>
    <AdminOnboardingModal />
    <AdminAlerts />
    <Box className="admin-header" sx={{ mb: 3, display: 'flex', flexDirection: { xs: 'column', md: 'row' }, alignItems: { md: 'center' }, justifyContent: 'space-between' }}>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: { xs: 2, md: 0 } }}>Admin Panel</Typography>
      <Box className="admin-user-info" sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>{(user.displayName || user.email || '?')[0]}</Avatar>
        <Typography variant="subtitle1">{user.email || user.displayName}</Typography>
        <Button onClick={handleLogout} variant="outlined" color="secondary" size="small">Logout</Button>
      </Box>
    </Box>
    <Divider sx={{ mb: 2 }} />
    <Box className="admin-tabs" sx={{ mb: 3 }}>
      <Tabs
        value={tab}
        onChange={(_, value) => setTab(value)}
        indicatorColor="primary"
        textColor="primary"
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab label="Metrics" value="metrics" />
        <Tab label="Agents" value="agents" />
        <Tab label="Queue" value="queue" />
        <Tab label="Users" value="users" />
        <Tab label="Backtest" value="backtest" />
        <Tab label="Analytics" value="analytics" />
      </Tabs>
    </Box>
    <Box className="admin-content">
      {tab === "metrics" && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Application Performance</Typography>
                {performance ? (
                  <List>
                    <ListItem>CPU Usage: {performance.cpu}%</ListItem>
                    <ListItem>Memory Usage: {performance.memory} MB</ListItem>
                    <ListItem>Uptime: {performance.uptime} s</ListItem>
                    <ListItem>Active Requests: {performance.activeRequests}</ListItem>
                  </List>
                ) : (
                  <Typography color="text.secondary">Loading...</Typography>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Real-Time Logs</Typography>
                <Box sx={{ maxHeight: 240, overflow: 'auto', bgcolor: 'grey.900', color: 'lime', p: 2, borderRadius: 1 }}>
                  <pre style={{ margin: 0, fontSize: 13 }}>{logs || "Loading..."}</pre>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <AdminCharts />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      {tab === "agents" && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Agents</Typography>
            {agents.length === 0 ? (
              <Typography>No agents found.</Typography>
            ) : (
              <List>
                {agents.map((agent) => (
                  <ListItem
                    button
                    key={agent.name}
                    selected={selectedAgent === agent.name}
                    onClick={() => setSelectedAgent(agent.name)}
                  >
                    <ListItemText primary={`${agent.name} (${agent.status})`} />
                  </ListItem>
                ))}
              </List>
            )}
            {agentDetails && (
              <Box className="agent-details" sx={{ mt: 2 }}>
                <Typography variant="subtitle1">Agent: {agentDetails.name}</Typography>
                <List>
                  <ListItem>Status: {agentDetails.status}</ListItem>
                  <ListItem>Type: {agentDetails.type}</ListItem>
                  <ListItem>Last Activity: {agentDetails.lastActivity}</ListItem>
                  <ListItem>Current Task: {agentDetails.currentTask}</ListItem>
                  <ListItem>Success Rate: {agentDetails.successRate}%</ListItem>
                  <ListItem>Errors: {agentDetails.errors}</ListItem>
                </List>
                <Typography variant="subtitle2" sx={{ mt: 2 }}>Recent Work</Typography>
                <List>
                  {agentDetails.recentWork.map((work, idx) => (
                    <ListItem key={idx}>{work}</ListItem>
                  ))}
                </List>
                <Box sx={{ mt: 2 }}>
                  {agentDetails.name && (
                    <AdminAgentControls agentName={agentDetails.name} />
                  )}
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      )}
      {tab === "queue" && (
        <Card>
          <CardContent>
            <AdminQueueStatus />
          </CardContent>
        </Card>
      )}
      {tab === "users" && (
        <Card>
          <CardContent>
            <AdminUserManagement />
          </CardContent>
        </Card>
      )}
      {tab === "backtest" && (
        <Card>
          <CardContent>
            <AdminBacktest />
          </CardContent>
        </Card>
      )}
      {tab === "analytics" && (
        <Card>
          <CardContent>
            <Typography>Analytics coming soon...</Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  </Box>
);
}

export default AdminPanel;
