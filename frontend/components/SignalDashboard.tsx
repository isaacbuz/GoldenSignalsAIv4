import React from "react";

/**
 * SignalDashboard displays a summary of current strategy performance and consensus metrics.
 */
export default function SignalDashboard() {
  return (
    <div className="p-4 text-white font-sans bg-bgPanel rounded-lg" aria-label="Signal dashboard summary">
      <h2 className="text-2xl font-bold mb-4 font-sans" aria-label="Signal Summary"> Signal Summary</h2>
      <p>View current strategy performance, blended signals, and agent consensus metrics.</p>
    </div>
  );
}
