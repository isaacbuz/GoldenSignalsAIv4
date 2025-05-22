import React from "react";
import { Dialog } from "@headlessui/react";
import { X } from "lucide-react";

interface TickerSignal {
  symbol: string;
  lastSignal: "buy" | "sell" | "hold";
  confidence: number;
  agents: string[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  sector: string;
  tickers: TickerSignal[];
}

const SectorTickerDrilldown: React.FC<Props> = ({ open, onClose, sector, tickers }) => {
  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-2xl rounded-xl bg-gray-900 p-6 shadow-xl text-white border border-gray-700">
          <div className="flex justify-between items-center mb-4">
            <Dialog.Title className="text-xl font-bold">
              {sector} — Active Tickers
            </Dialog.Title>
            <button onClick={onClose}><X className="h-5 w-5 text-gray-400" /></button>
          </div>
          <div className="overflow-y-auto max-h-[50vh]">
            <table className="w-full text-sm">
              <thead className="text-gray-400 border-b border-gray-700">
                <tr>
                  <th className="text-left py-2">Ticker</th>
                  <th className="text-left py-2">Signal</th>
                  <th className="text-left py-2">Confidence</th>
                  <th className="text-left py-2">Agents</th>
                </tr>
              </thead>
              <tbody>
                {tickers.map((t, i) => (
                  <tr key={i} className="border-b border-gray-800">
                    <td className="py-2 font-bold">{t.symbol}</td>
                    <td className={`py-2 ${t.lastSignal === "buy" ? "text-green-400" : t.lastSignal === "sell" ? "text-red-400" : "text-yellow-400"}`}>
                      {t.lastSignal.toUpperCase()}
                    </td>
                    <td className="py-2">{(t.confidence * 100).toFixed(1)}%</td>
                    <td className="py-2 text-xs text-gray-300">{t.agents.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default SectorTickerDrilldown;
