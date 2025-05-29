import React, { useEffect, useState } from "react";
import type { Signal } from '@/hooks/useLiveSignalFeed';

type Props = {
  signal: Signal | null;
  open: boolean;
  onClose: () => void;
};

const getToastColor = (signal: string) => {
  if (signal === 'buy') return 'bg-green-600';
  if (signal === 'sell') return 'bg-red-600';
  return 'bg-yellow-600';
};

const LiveSignalToast: React.FC<Props> = ({ signal, open, onClose }) => {
  useEffect(() => {
    if (open) {
      const timer = setTimeout(onClose, 3500);
      return () => clearTimeout(timer);
    }
  }, [open, onClose]);

  if (!open || !signal) return null;

  return (
    <div className={`fixed bottom-6 right-6 z-50 px-6 py-4 rounded shadow-lg text-white ${getToastColor(signal.signal)} animate-fadeIn`}
      role="alert"
      aria-live="assertive"
    >
      <div className="font-bold text-lg mb-1">New Signal: {signal.name || signal.source}</div>
      <div className="flex items-center gap-2">
        <span className="font-mono text-xl">{signal.signal.toUpperCase()}</span>
        {typeof signal.confidence === 'number' && (
          <span className="text-xs ml-2">Confidence: {signal.confidence}%</span>
        )}
      </div>
      {signal.explanation && (
        <div className="text-xs text-gray-200 mt-1">{signal.explanation}</div>
      )}
    </div>
  );
};

export default LiveSignalToast;
