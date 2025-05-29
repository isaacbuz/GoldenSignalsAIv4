import React from 'react';

export default function SignalExplainModal({ open, onClose, signal }: { open: boolean, onClose: () => void, signal: any }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60">
      <div className="bg-white dark:bg-zinc-900 p-6 rounded shadow-lg max-w-lg w-full">
        <h3 className="text-lg font-bold mb-2">AI Signal Explanation</h3>
        <div className="mb-4">
          {signal ? (
            <>
              <div><b>Symbol:</b> {signal.symbol}</div>
              <div><b>Action:</b> {signal.action}</div>
              <div><b>Confidence:</b> {signal.confidence}%</div>
              <div><b>Reason:</b> {signal.reason}</div>
              {/* Add more details as needed */}
            </>
          ) : (
            <div>No signal selected.</div>
          )}
        </div>
        <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
