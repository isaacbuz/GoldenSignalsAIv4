import React from "react";
import AdminAgentDashboard from "./AdminAgentDashboard";
import FeedbackReviewPanel from "./FeedbackReviewPanel";
import AgentAnalyticsPanel from "./AgentAnalyticsPanel";
import AgentPerformanceChart from "./AgentPerformanceChart";

export default function AdminDashboard() {
  return (
    <div className="admin-dashboard" style={{ padding: 24 }}>
      <h1>Admin Dashboard</h1>
      <div style={{ display: "flex", gap: 48, alignItems: "flex-start" }}>
        <div style={{ flex: 2 }}>
          <AdminAgentDashboard />
        </div>
        <div style={{ flex: 1 }}>
          <FeedbackReviewPanel />
        </div>
      </div>
      <AgentAnalyticsPanel />
      <AgentPerformanceChart />
    </div>
  );
}
