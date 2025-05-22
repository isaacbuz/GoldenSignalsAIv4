import React from "react";

export default function SignalCard({ signal }) {
  return (
    <div className="bg-white p-6 rounded shadow-md">
      <h2 className="text-xl font-semibold mb-2">{signal.symbol}</h2>
      <p><strong>Action:</strong> {signal.action}</p>
      <p><strong>Confidence:</strong> {signal.confidence}</p>
      <p><strong>Entry Price:</strong> ${signal.entry_price}</p>
      <p><strong>Exit Price:</strong> ${signal.exit_price}</p>
      <p className="mt-4"><strong>Reasoning:</strong></p>
      <pre className="text-sm bg-gray-50 p-3 rounded overflow-auto">{JSON.stringify(signal.reasoning, null, 2)}</pre>
    </div>
  );
}
