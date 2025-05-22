import React from 'react';

export default function SignalMarkerTooltip({ signal }) {
  return (
    <div style={{ minWidth: 180, color: '#FFD700', background: '#232323', border: '1px solid #FFD700', borderRadius: 8, padding: 10 }}>
      <b>{signal.type === 'entry' ? 'Buy Signal' : 'Sell Signal'}</b>
      <div>Time: {new Date(signal.timestamp || signal.time).toLocaleString()}</div>
      <div>Price: {signal.price}</div>
      <div>RSI: {signal.rsi ?? '--'}</div>
      <div>MACD: {signal.macd ?? '--'}</div>
      <div>Confidence: {signal.confidence ? (signal.confidence * 100).toFixed(1) + '%' : '--'}</div>
      {signal.news && signal.news.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <b>News:</b>
          <ul>
            {signal.news.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
