import React from "react";

interface AgentHealth {
  id: string;
  name: string;
  online: boolean;
  uptime: number;
  ping: number | null;
  errors: number;
}

const AgentHealthMonitor: React.FC<{ agents: AgentHealth[] }> = ({ agents }) => {
  const formatUptime = (s: number) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return `${h}h ${m}m`;
  };

  return (
    <div className="mt-10">
      <h2 className="text-lg font-bold mb-4">📡 Agent Health Monitor</h2>
      <table className="w-full text-sm border border-gray-800">
        <thead className="text-gray-400 bg-gray-800 border-b border-gray-700">
          <tr>
            <th className="px-3 py-2 text-left">Agent</th>
            <th className="px-3 py-2 text-left">Status</th>
            <th className="px-3 py-2 text-left">Uptime</th>
            <th className="px-3 py-2 text-left">Ping</th>
            <th className="px-3 py-2 text-left">Errors</th>
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.id} className="border-b border-gray-700">
              <td className="px-3 py-2 font-semibold">{a.name}</td>
              <td className="px-3 py-2">
                <span className={`font-bold ${a.online ? "text-green-400" : "text-red-500"}`}>
                  {a.online ? "Online" : "Offline"}
                </span>
              </td>
              <td className="px-3 py-2">{formatUptime(a.uptime)}</td>
              <td className="px-3 py-2">{a.ping ? `${a.ping} ms` : "-"}</td>
              <td className="px-3 py-2">{a.errors}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AgentHealthMonitor;
