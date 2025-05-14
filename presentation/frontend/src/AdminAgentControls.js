import React, { useState } from "react";

function AdminAgentControls({ agentName, onAction }) {
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const handleAction = async (action) => {
    setLoading(true);
    setMsg("");
    try {
      const res = await fetch(`/api/admin/agents/${agentName}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      setMsg(data.message);
      if (onAction) onAction(action, data);
    } catch (e) {
      setMsg("Error performing action");
    }
    setLoading(false);
  };

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
      {msg && <span style={{ marginLeft: "1rem", fontWeight: "bold" }}>{msg}</span>}
    </div>
  );
}

export default AdminAgentControls;
