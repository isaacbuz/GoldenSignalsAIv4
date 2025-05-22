import React from 'react';

export default function QuickActionToolbar({ signal, onBacktest, onOrder, onWatchlist }) {
  if (!signal) return null;
  return (
    <div id="quick-actions" style={{ position: 'absolute', left: 40, top: 10, zIndex: 12, display: 'flex', gap: 12 }}>
      <button style={{ background: '#FFD700', color: '#232323', border: 'none', borderRadius: 8, padding: '6px 12px', fontWeight: 700 }}
        onClick={() => onBacktest(signal)}>
        Backtest
      </button>
      <button style={{ background: '#232323', color: '#FFD700', border: '1.5px solid #FFD700', borderRadius: 8, padding: '6px 12px', fontWeight: 700 }}
        onClick={() => onOrder(signal)}>
        Create Order
      </button>
      <button style={{ background: '#FFD700', color: '#232323', border: 'none', borderRadius: 8, padding: '6px 12px', fontWeight: 700 }}
        onClick={() => onWatchlist(signal)}>
        Add to Watchlist
      </button>
    </div>
  );
}
