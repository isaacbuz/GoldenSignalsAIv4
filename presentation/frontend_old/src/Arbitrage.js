// Arbitrage.js
// Purpose: Displays arbitrage opportunities and allows users to execute trades. Fetches arbitrage data from the backend and manages trade execution state. Designed for user-facing arbitrage discovery and action within GoldenSignalsAI.

import React, { useEffect, useState } from 'react';
import './Arbitrage.css';
import API_URL from './config';

// Arbitrage component: displays arbitrage opportunities and allows users to execute trades
function Arbitrage() {
  // State for storing arbitrage opportunities
  const [opportunities, setOpportunities] = useState([]);
  // State for tracking loading state
  const [loading, setLoading] = useState(true);
  // State for tracking trade execution status
  const [tradeStatus, setTradeStatus] = useState("");
  const [symbol, setSymbol] = useState('AAPL');
  // Optionally, add error state for invalid ticker
  const [tickerError, setTickerError] = useState("");
  const [executing, setExecuting] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch arbitrage opportunities from backend
  const fetchOpportunities = async () => {
    if (tickerError) return;
    try {
      const res = await fetch(`${API_URL}/arbitrage/opportunities`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, min_spread: 0.01 })
      });
      if (!res.ok) {
        let text = await res.text();
        try { text = JSON.parse(text); } catch {}
        setMessage('Error: ' + (text.detail || text || res.status));
        setOpportunities([]);
        return;
      }
      const data = await res.json();
      setOpportunities(data);
      setMessage('');
    } catch (err) {
      setMessage('Error fetching opportunities: ' + (err.message || err));
      setOpportunities([]);
    }
  };

  // Execute arbitrage trade for all opportunities
  const executeArbitrage = async () => {
    if (tickerError) return;
    setExecuting(true);
    setMessage('');
    try {
      const res = await fetch(`${API_URL}/arbitrage/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, min_spread: 0.01 })
      });
      if (!res.ok) {
        let text = await res.text();
        try { text = JSON.parse(text); } catch {}
        setMessage('Error: ' + (text.detail || text || res.status));
        setExecuting(false);
        return;
      }
      const data = await res.json();
      setMessage(`Executed ${data.executed} out of ${data.total} opportunities.`);
    } catch (err) {
      setMessage('Error executing arbitrage: ' + (err.message || err));
    }
    setExecuting(false);
    fetchOpportunities();
  };

  // Fetch arbitrage opportunities on mount and every 15 seconds
  useEffect(() => {
    fetchOpportunities();
    const interval = setInterval(fetchOpportunities, 15000);
    return () => clearInterval(interval);
  }, [symbol]);

  return (
    <>
      <div className="arbitrage-container">
        <h2>AI Arbitrage Opportunities</h2>
        <div className="arbitrage-controls">
          {/* Material UI TextField for ticker input */}
          <label htmlFor="ticker-input">Symbol: </label>
          <input
            id="ticker-input"
            type="text"
            value={symbol}
            onChange={async (e) => {
              const newSymbol = e.target.value.toUpperCase();
              setSymbol(newSymbol);
              setTickerError("");
              if (newSymbol && newSymbol.length >= 1) {
                try {
                  const jwt = localStorage.getItem('jwt_token');
                  const res = await fetch(`${API_URL}/api/tickers/validate`, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                      ...(jwt ? { Authorization: `Bearer ${jwt}` } : {})
                    },
                    body: JSON.stringify({ symbol: newSymbol })
                  });
                  const data = await res.json();
                  if (!data.valid) {
                    setTickerError('Invalid ticker symbol');
                  } else {
                    setTickerError("");
                  }
                } catch (err) {
                  setTickerError('Validation failed');
                }
              } else {
                setTickerError("");
              }
            }}
            placeholder="Enter stock ticker (e.g. AAPL)"
            style={{ width: 120, marginRight: 8, padding: 4, borderRadius: 4, border: '1px solid #333' }}
          />
          {tickerError && <span style={{ color: 'red', marginLeft: 8 }}>{tickerError}</span>}
          <button onClick={fetchOpportunities} disabled={!!tickerError}>Refresh</button>
          <button onClick={executeArbitrage} disabled={executing || !!tickerError} className="execute-btn">
            {executing ? 'Executing...' : 'Execute All'}
          </button>
        </div>
        {message && <div className="arbitrage-message" style={{color: message.startsWith('Error') ? 'red' : undefined}}>{message}</div>}
        <table className="arbitrage-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Buy Venue</th>
              <th>Sell Venue</th>
              <th>Buy Price</th>
              <th>Sell Price</th>
              <th>Spread</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {opportunities.length === 0 ? (
              <tr><td colSpan="7">No opportunities found.</td></tr>
            ) : (
              opportunities.map((opp, idx) => (
                <tr key={idx} className={opp.spread > 0.1 ? 'highlight-row' : ''}>
                  <td>{opp.symbol}</td>
                  <td>{opp.buy_venue}</td>
                  <td>{opp.sell_venue}</td>
                  <td>${opp.buy_price.toFixed(2)}</td>
                  <td>${opp.sell_price.toFixed(2)}</td>
                  <td className="spread-cell">${opp.spread.toFixed(2)}</td>
                  <td>{opp.status}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default Arbitrage;
