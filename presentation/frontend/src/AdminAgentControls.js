// AdminAgentControls.js
// Purpose: Provides controls for admin users to manage agents in GoldenSignalsAI. Allows restarting or disabling agents via API calls. Displays action status and feedback messages for each operation.

import React, { useState } from "react";

function AdminAgentControls({ agentName, onAction }) {
  // State to indicate loading status for async actions
  const [loading, setLoading] = useState(false);
  // State for feedback message to display action results
  const [msg, setMsg] = useState("");

  // Handle admin actions (restart/disable) for the agent
  const handleAction = async (action) => {
    setLoading(true);
    setMsg("");
    try {
      // Send POST request to backend API for the selected action
      const res = await fetch(`/api/admin/agents/${agentName}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      setMsg(data.message); // Show feedback message
      if (onAction) onAction(action, data); // Notify parent if callback provided
    } catch (e) {
      setMsg("Error performing action");
    }
    setLoading(false);
  };

  // Render action buttons for restart and disable, and show feedback message
  return (
    <div style={{ margin: "1rem 0" }}>
      <button
        onClick={() => handleAction("restart")}
        disabled={loading}
        style={{ marginRight: "1rem", background: "#4fc3a1", color: "#fff", border: "none", borderRadius: 6, padding: "0.5rem 1rem", cursor: "pointer" }}
      >
        Restart Agent
      </button>
      <button
        onClick={() => handleAction("disable")}
        disabled={loading}
        style={{ background: "#ff5252", color: "#fff", border: "none", borderRadius: 6, padding: "0.5rem 1rem", cursor: "pointer" }}
      >
        Disable Agent
      </button>
      {/* Show feedback message if present */}
      {msg && <span style={{ marginLeft: "1rem", fontWeight: "bold" }}>{msg}</span>}
    </div>
  );
}

export default AdminAgentControls;
