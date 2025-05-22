import React from "react";
import { useWebSocket } from "../../context/WebSocketContext";

function timeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const mins = Math.floor(seconds / 60);
  return `${mins}m ago`;
}

const SignalSidebar: React.FC = () => {
  const { signals } = useWebSocket();

  const flattened = Object.entries(signals)
    .flatMap(([symbol, list]) => list.map((s) => ({ ...s, symbol })))
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 8); // Show only 8 most recent

  return (
    <div className="w-full bg-gray-900 p-4 rounded-xl shadow-lg text-white">
      <h2 className="text-lg font-bold mb-4">🔔 Live Signals</h2>
      {flattened.map((sig, idx) => (
        <div
          key={sig.id + idx}
          className={`mb-3 p-3 rounded-lg border-l-4 ${
            sig.type === "buy" ? "border-green-400 bg-green-900/10" : "border-red-400 bg-red-900/10"
          }`}
        >
          <div className="flex justify-between items-center">
            <span className="font-semibold">{sig.symbol}</span>
            <span className="text-sm opacity-70">{timeAgo(sig.timestamp)}</span>
          </div>
          <div className="text-sm">
            <strong>{sig.type.toUpperCase()}</strong> •{" "}
            <span className="text-yellow-300">{(sig.confidence * 100).toFixed(0)}%</span>
          </div>
          {sig.source && (
            <div className="text-xs mt-1 text-gray-400 italic">{sig.source}</div>
          )}
        </div>
      ))}
    </div>
  );
};

export default SignalSidebar;
