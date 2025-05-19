// AdminBacktest.js
// Robust backtesting UI for the admin panel
import React, { useState } from 'react';
import API_URL from './config';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

function AdminBacktest() {
  const [symbol, setSymbol] = useState('AAPL');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);

  const handleBacktest = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    setHistory([]);
    try {
      const jwt = localStorage.getItem('jwt_token');
      const res = await fetch(`${API_URL}/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(jwt ? { Authorization: `Bearer ${jwt}` } : {})
        },
        body: JSON.stringify({ symbol })
      });
      if (!res.ok) throw new Error('Backtest failed');
      const data = await res.json();
      setResult(data.backtest_result);
      if (data.backtest_result && data.backtest_result.equity_curve) {
        setHistory(data.backtest_result.equity_curve);
      }
    } catch (err) {
      setError(err.message || 'Error running backtest');
    }
    setLoading(false);
  };

  // Chart data for equity curve
  const chartData = {
    labels: history.map((pt, i) => i),
    datasets: [
      {
        label: 'Equity Curve',
        data: history.map(pt => pt.equity),
        fill: false,
        borderColor: '#1976d2',
        backgroundColor: '#2196f3',
        tension: 0.1,
      },
    ],
  };

  return (
    <div className="admin-backtest" style={{ maxWidth: 600, margin: '0 auto' }}>
      <h3>Backtesting</h3>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <label htmlFor="backtest-symbol">Symbol:</label>
        <input
          id="backtest-symbol"
          type="text"
          value={symbol}
          onChange={e => setSymbol(e.target.value.toUpperCase())}
          style={{ width: 120, padding: 6, borderRadius: 4, border: '1px solid #333' }}
          placeholder="e.g. AAPL"
        />
        <button onClick={handleBacktest} disabled={loading || !symbol} style={{ padding: '6px 18px' }}>
          {loading ? 'Running...' : 'Run Backtest'}
        </button>
      </div>
      {error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 24 }}>
          <h4>Summary</h4>
          <ul>
            <li><b>Total Return:</b> {result.total_return ? (result.total_return * 100).toFixed(2) + '%' : 'N/A'}</li>
            <li><b>Sharpe Ratio:</b> {result.sharpe_ratio ?? 'N/A'}</li>
            <li><b>Max Drawdown:</b> {result.max_drawdown ? (result.max_drawdown * 100).toFixed(2) + '%' : 'N/A'}</li>
            <li><b>Trades:</b> {result.trades ?? 'N/A'}</li>
          </ul>
        </div>
      )}
      {history.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h4>Equity Curve</h4>
          <Line data={chartData} options={{ plugins: { legend: { display: true } } }} />
        </div>
      )}
    </div>
  );
}

export default AdminBacktest;
