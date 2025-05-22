import React from "react";
import { Dialog } from "@headlessui/react";
import { XIcon } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
  signal: {
    symbol: string;
    type: string;
    confidence: number;
    timestamp: number;
    indicators?: Record<string, number>;
    agents?: string[];
    headlines?: string[];
  };
}

const SignalDetailsModal: React.FC<Props> = ({ open, onClose, signal }) => {
  const date = new Date(signal.timestamp).toLocaleString();

  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-md rounded-xl bg-gray-900 p-6 shadow-xl text-white border border-gray-700">
          <div className="flex justify-between items-center mb-4">
            <Dialog.Title className="text-xl font-bold">
              {signal.symbol} • {signal.type.toUpperCase()} • {Math.round(signal.confidence * 100)}%
            </Dialog.Title>
            <button onClick={onClose}>
              <XIcon className="h-5 w-5 text-gray-400 hover:text-white" />
            </button>
          </div>

          <p className="text-sm text-gray-400 mb-2">🗓 {date}</p>

          {signal.indicators && (
            <>
              <h3 className="font-semibold mt-4">💡 Indicators</h3>
              <ul className="text-sm list-disc pl-5">
                {Object.entries(signal.indicators).map(([k, v]) => (
                  <li key={k}>
                    {k}: {v}
                  </li>
                ))}
              </ul>
            </>
          )}

          {signal.agents && (
            <>
              <h3 className="font-semibold mt-4">🧠 Agents</h3>
              <p className="text-sm text-gray-300">{signal.agents.join(", ")}</p>
            </>
          )}

          {signal.headlines && (
            <>
              <h3 className="font-semibold mt-4">📰 News</h3>
              <ul className="text-sm list-disc pl-5">
                {signal.headlines.map((h, i) => (
                  <li key={i}>{h}</li>
                ))}
              </ul>
            </>
          )}
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default SignalDetailsModal;
