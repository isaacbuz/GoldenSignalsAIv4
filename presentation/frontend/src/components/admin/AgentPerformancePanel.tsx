import React from "react";

interface AgentStats {
  name: string;
  signals: number;
  winRate: number; // decimal (0.87)
  avgConfidence: number; // decimal
}

const AgentPerformancePanel: React.FC<{ data: AgentStats[] }> = ({ data }) => {
  return (
    <div className="mt-8">
      <h2 className="text-lg font-bold mb-4">🧠 Agent Performance</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border border-gray-800">
          <thead className="text-gray-400 bg-gray-800 border-b border-gray-700">
            <tr>
              <th className="text-left py-2 px-3">Agent</th>
              <th className="text-left py-2 px-3">Signals</th>
              <th className="text-left py-2 px-3">Win Rate</th>
              <th className="text-left py-2 px-3">Avg Confidence</th>
            </tr>
          </thead>
          <tbody>
            {data.map((a, i) => (
              <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/30">
                <td className="px-3 py-2 font-semibold">{a.name}</td>
                <td className="px-3 py-2">{a.signals}</td>
                <td className="px-3 py-2 text-green-400">{(a.winRate * 100).toFixed(1)}%</td>
                <td className="px-3 py-2 text-yellow-300">{(a.avgConfidence * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AgentPerformancePanel;
