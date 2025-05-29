import React from "react";
import type { Signal } from '@/hooks/useLiveSignalFeed';

type Props = {
  signals: Signal[];
  symbol?: string;
};

const getSignalColor = (signal: string) => {
  if (signal === 'buy') return 'text-green-400';
  if (signal === 'sell') return 'text-red-400';
  return 'text-yellow-300';
};

const SignalLogPanel: React.FC<Props> = ({ signals, symbol }) => (
  <div className="bg-[#232c3b] rounded-lg shadow-md p-4 w-full max-w-3xl">
    <h3 className="text-lg font-semibold text-accentBlue mb-2">
      Signal Log{symbol ? ` for ${symbol}` : ''}
    </h3>
    <ul className="divide-y divide-gray-700">
      {signals.length === 0 && <li className="text-gray-400 py-2">No signals yet.</li>}
      {signals.map((s, i) => (
        <li key={i} className="flex flex-col md:flex-row md:items-center py-2 gap-1 md:gap-4">
          <span className="font-semibold text-accentPink">{s.name || s.source}</span>
          <span className={`font-mono ${getSignalColor(s.signal)} font-bold`}>{s.signal}</span>
          {typeof s.confidence === 'number' && (
            <span className="text-xs text-gray-400 ml-2">Confidence: {s.confidence}%</span>
          )}
          {s.explanation && (
            <span className="text-xs text-gray-500 ml-2 italic">{s.explanation}</span>
          )}
        </li>
      ))}
    </ul>
  </div>
);

export default SignalLogPanel;
