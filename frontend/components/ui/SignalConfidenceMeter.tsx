import React from "react";

type Props = {
  confidence: number;
  label?: string;
};

const getColor = (confidence: number) => {
  if (confidence >= 85) return "bg-green-500";
  if (confidence >= 70) return "bg-yellow-400";
  if (confidence >= 50) return "bg-orange-400";
  return "bg-red-500";
};

const SignalConfidenceMeter: React.FC<Props> = ({ confidence, label }) => (
  <div className="w-full flex flex-col items-start gap-1">
    {label && <span className="text-xs text-gray-400 mb-1">{label}</span>}
    <div className="w-full h-4 bg-gray-700 rounded">
      <div
        className={`h-4 rounded ${getColor(confidence)}`}
        style={{ width: `${confidence}%`, transition: 'width 0.5s' }}
      />
    </div>
    <span className="text-xs text-white ml-1">{confidence}%</span>
  </div>
);

export default SignalConfidenceMeter;
