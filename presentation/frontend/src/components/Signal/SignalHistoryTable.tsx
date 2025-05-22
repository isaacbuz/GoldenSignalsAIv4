import React, { useMemo, useState } from "react";
import { useWebSocket } from "../../context/WebSocketContext";

const SignalHistoryTable: React.FC = () => {
  const { signals } = useWebSocket();
  const [filter, setFilter] = useState("");
  const [sortBy, setSortBy] = useState<"confidence" | "timestamp">("timestamp");

  const rows = useMemo(() => {
    return Object.entries(signals)
      .flatMap(([symbol, list]) =>
        list.map((s) => ({ ...s, symbol }))
      )
      .filter((r) => r.symbol.toLowerCase().includes(filter.toLowerCase()))
      .sort((a, b) =>
        sortBy === "confidence"
          ? b.confidence - a.confidence
          : b.timestamp - a.timestamp
      );
  }, [signals, filter, sortBy]);

  return (
    <div className="bg-gray-900 text-white rounded-xl p-4 shadow-lg">
      <div className="flex justify-between mb-4">
        <input
          type="text"
          placeholder="Filter by symbol..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-3 py-1 rounded bg-gray-800 text-sm border border-gray-700 w-1/2"
        />
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
          className="px-3 py-1 rounded bg-gray-800 text-sm border border-gray-700"
        >
          <option value="timestamp">Newest</option>
          <option value="confidence">Confidence</option>
        </select>
      </div>

      <table className="w-full text-sm table-auto">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left py-2">Symbol</th>
            <th className="text-left py-2">Type</th>
            <th className="text-left py-2">Confidence</th>
            <th className="text-left py-2">Time</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={r.id + i}
              className="border-b border-gray-800 hover:bg-gray-800/50 transition"
            >
              <td className="py-1 font-semibold">{r.symbol}</td>
              <td className={`py-1 ${r.type === "buy" ? "text-green-400" : "text-red-400"}`}>
                {r.type.toUpperCase()}
              </td>
              <td className="py-1">{(r.confidence * 100).toFixed(1)}%</td>
              <td className="py-1">
                {new Date(r.timestamp).toLocaleTimeString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default SignalHistoryTable;
