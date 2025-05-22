import React, { useState } from 'react';

export default function ReplaySession({ candles, signals }) {
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);

  const maxFrame = candles.length;

  const play = () => {
    setPlaying(true);
    let f = frame;
    const interval = setInterval(() => {
      f++;
      if (f >= maxFrame) {
        setPlaying(false);
        clearInterval(interval);
      } else {
        setFrame(f);
      }
    }, 400);
  };

  return (
    <div style={{ marginTop: 12 }}>
      <button onClick={play} disabled={playing} style={{ background: '#FFD700', color: '#232323', border: 'none', borderRadius: 8, padding: '6px 18px', fontWeight: 700 }}>
        {playing ? 'Replaying...' : 'Replay Day'}
      </button>
      <div style={{ marginTop: 10 }}>
        {/* Render a chart up to frame and signals up to frame (to be integrated with chart) */}
        <div style={{ color: '#FFD700' }}>Frame: {frame + 1} / {maxFrame}</div>
      </div>
    </div>
  );
}
