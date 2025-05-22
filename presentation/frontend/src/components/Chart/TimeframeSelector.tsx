import React from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
}

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"];

const TimeframeSelector: React.FC<Props> = ({ value, onChange }) => {
  return (
    <div className="inline-flex items-center space-x-1 bg-gray-800 rounded-lg p-1 shadow-inner">
      {TIMEFRAMES.map((tf) => (
        <button
          key={tf}
          onClick={() => onChange(tf)}
          title={`Switch to ${tf}`}
          className={`px-3 py-1 rounded-md text-sm font-medium transition ${
            value === tf
              ? "bg-green-500 text-black shadow"
              : "text-gray-300 hover:bg-gray-700"
          }`}
        >
          {tf}
        </button>
      ))}
    </div>
  );
};

export default TimeframeSelector;
