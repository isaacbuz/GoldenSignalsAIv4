import React from "react";

interface Signal {
  symbol: string;
  type: "buy" | "sell";
  confidence: number;
  timestamp: number;
  agents: string[];
}

const AdminRecentSignalsTable: React.FC<{ data: Signal[] }> = ({ data }) => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <div className="p-4 text-lg font-semibold border-b border-gray-800">📝 Recent Signals</div>
      <table className="w-full text-sm text-left">
        <thead className="text-gray-400 bg-gray-800 border-b border-gray-700">
          <tr>
            <th className="px-4 py-2">Symbol</th>
            <th className="px-4 py-2">Type</th>
            <th className="px-4 py-2">Confidence</th>
            <th className="px-4 py-2">Time</th>
            <th className="px-4 py-2">Agents</th>
          </tr>
        </thead>
        <tbody>
          {data.map((s, i) => (
            <tr key={i} className="hover:bg-gray-800/40 transition border-b border-gray-800">
              <td className="px-4 py-2 font-bold">{s.symbol}</td>
              <td className={`px-4 py-2 ${s.type === "buy" ? "text-green-400" : "text-red-400"}`}>
                {s.type.toUpperCase()}
              </td>
              <td className="px-4 py-2">{(s.confidence * 100).toFixed(1)}%</td>
              <td className="px-4 py-2">{new Date(s.timestamp).toLocaleTimeString()}</td>
              <td className="px-4 py-2 text-xs text-gray-300">{s.agents.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AdminRecentSignalsTable;
