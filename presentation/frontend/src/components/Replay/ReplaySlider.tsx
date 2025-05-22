import React, { useEffect, useState } from "react";

interface Tick {
  timestamp: number;
  price: number;
}

interface Signal {
  timestamp: number;
  type: string;
  confidence: number;
  symbol: string;
}

interface Props {
  ticks: Tick[];
  signals: Signal[];
  onFrameUpdate: (frame: { ticks: Tick[]; signals: Signal[] }) => void;
}

const ReplaySlider: React.FC<Props> = ({ ticks, signals, onFrameUpdate }) => {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // ticks per second

  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setFrameIndex((i) => Math.min(i + speed, ticks.length - 1));
    }, 1000);
    return () => clearInterval(interval);
  }, [playing, speed, ticks]);

  useEffect(() => {
    const currentTime = ticks[frameIndex]?.timestamp;
    const currentTicks = ticks.slice(0, frameIndex + 1);
    const currentSignals = signals.filter((s) => s.timestamp <= currentTime);
    onFrameUpdate({ ticks: currentTicks, signals: currentSignals });
  }, [frameIndex, ticks, signals]);

  const timeLabel = new Date(ticks[frameIndex]?.timestamp || 0).toLocaleTimeString();

  return (
    <div className="p-4 bg-gray-900 text-white rounded-xl mt-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm">{timeLabel}</span>
        <div className="space-x-2">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded"
          >
            {playing ? "Pause ⏸" : "Play ▶️"}
          </button>
          <select
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
            className="bg-gray-800 px-2 py-1 rounded"
          >
            <option value={1}>1x</option>
            <option value={5}>5x</option>
            <option value={10}>10x</option>
          </select>
          <button
            onClick={() => setFrameIndex(0)}
            className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded"
          >
            Reset 🔄
          </button>
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={ticks.length - 1}
        value={frameIndex}
        onChange={(e) => setFrameIndex(+e.target.value)}
        className="w-full"
      />
    </div>
  );
};

export default ReplaySlider;
