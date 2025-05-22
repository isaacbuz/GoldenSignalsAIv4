import React from 'react';

export default function DrillInPanel({ signal, onClose }) {
  if (!signal) return null;
  return (
    <div style={{ position: 'fixed', right: 0, top: 0, width: 340, height: '100%', background: '#232323', color: '#FFD700', boxShadow: '-2px 0 16px #FFD70033', zIndex: 1000, padding: 24 }}>
      <button style={{ float: 'right', color: '#FFD700', background: 'none', border: 'none', fontSize: 24 }} onClick={onClose}>×</button>
      <h2>{signal.type === 'entry' ? 'Buy Signal' : 'Sell Signal'}</h2>
      <div><b>Time:</b> {new Date(signal.timestamp || signal.time).toLocaleString()}</div>
      <div><b>Price:</b> {signal.price}</div>
      <div><b>RSI:</b> {signal.rsi ?? '--'}</div>
      <div><b>MACD:</b> {signal.macd ?? '--'}</div>
      <div><b>Confidence:</b> {signal.confidence ? (signal.confidence * 100).toFixed(1) + '%' : '--'}</div>
      <div style={{ marginTop: 16 }}>
        <b>Backtest Performance:</b>
        <div>{signal.backtest ? `${signal.backtest.pnl}% P/L, Sharpe: ${signal.backtest.sharpe}` : '--'}</div>
      </div>
      {signal.news && signal.news.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <b>Relevant News:</b>
          <ul>
            {signal.news.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
