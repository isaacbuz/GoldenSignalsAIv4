import React from 'react';

interface ReplayControlsProps {
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onSpeedChange: (speed: number) => void;
  speed: number;
}

export default function ReplayControls({ isPlaying, onPlay, onPause, onStep, onSpeedChange, speed }: ReplayControlsProps) {
  return (
    <div className="flex items-center gap-4 mb-4">
      <button
        className={`px-4 py-2 rounded text-white ${isPlaying ? 'bg-gray-600' : 'bg-blue-600'}`}
        onClick={isPlaying ? onPause : onPlay}
      >
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <button className="px-4 py-2 rounded bg-yellow-600 text-white" onClick={onStep}>
        Step
      </button>
      <label className="flex items-center gap-2">
        <span className="text-white">Speed:</span>
        <input
          type="range"
          min={0.25}
          max={2}
          step={0.25}
          value={speed}
          onChange={e => onSpeedChange(Number(e.target.value))}
        />
        <span className="text-white">{speed}x</span>
      </label>
    </div>
  );
}
