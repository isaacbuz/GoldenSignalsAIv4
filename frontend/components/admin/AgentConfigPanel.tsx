import React, { useState } from "react";

interface AgentConfigPanelProps {
  agent: string;
  initialConfig: Record<string, any>;
  onConfigUpdated?: (config: Record<string, any>) => void;
}

export default function AgentConfigPanel({ agent, initialConfig, onConfigUpdated }: AgentConfigPanelProps) {
  const [config, setConfig] = useState<Record<string, any>>(initialConfig);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (key: string, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setSuccess(false);
    try {
      const res = await fetch("/api/agents/admin/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent, config }),
      });
      if (!res.ok) throw new Error("Failed to update config");
      setSuccess(true);
      if (onConfigUpdated) onConfigUpdated(config);
    } catch (e) {
      setError("Error updating config");
    }
    setSaving(false);
  };

  return (
    <form className="agent-config-panel" onSubmit={handleSubmit}>
      <h3>Configure Agent: {agent}</h3>
      {Object.entries(config).map(([key, value]) => (
        <div key={key}>
          <label>{key}: </label>
          <input
            value={value}
            onChange={e => handleChange(key, e.target.value)}
            disabled={saving}
          />
        </div>
      ))}
      <button type="submit" disabled={saving}>Save Config</button>
      {success && <span className="success">Config updated!</span>}
      {error && <span className="error">{error}</span>}
    </form>
  );
}
