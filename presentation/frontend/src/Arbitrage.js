// Arbitrage.js
// Purpose: Displays arbitrage opportunities and allows users to execute trades. Fetches arbitrage data from the backend and manages trade execution state. Designed for user-facing arbitrage discovery and action within GoldenSignalsAI.

import React, { useEffect, useState } from 'react';
import './Arbitrage.css';

// Arbitrage component: displays arbitrage opportunities and allows users to execute trades
function Arbitrage() {
  // State for storing arbitrage opportunities
  const [opportunities, setOpportunities] = useState([]);
  // State for tracking loading state
  const [loading, setLoading] = useState(true);
  // State for tracking trade execution status
  const [tradeStatus, setTradeStatus] = useState("");
  const [symbol, setSymbol] = useState('AAPL');
  const [executing, setExecuting] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch arbitrage opportunities from backend
  const fetchOpportunities = async () => {
    // Send POST request to backend with symbol and min spread
    const res = await fetch('http://localhost:8000/arbitrage/opportunities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, min_spread: 0.01 })
    });
    // Parse response data and update opportunities state
    const data = await res.json();
    setOpportunities(data);
  };

  // Execute arbitrage trade for all opportunities
  const executeArbitrage = async () => {
    // Set executing state to true and clear message
    setExecuting(true);
    setMessage('');
    // Send POST request to backend with symbol and min spread
    const res = await fetch('http://localhost:8000/arbitrage/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, min_spread: 0.01 })
    });
    // Parse response data and update message state
    const data = await res.json();
    setMessage(`Executed ${data.executed} out of ${data.total} opportunities.`);
    // Set executing state to false and refetch opportunities
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
    <div className="arbitrage-container">
      <h2>AI Arbitrage Opportunities</h2>
      <div className="arbitrage-controls">
        <label>Symbol: </label>
        <select value={symbol} onChange={e => setSymbol(e.target.value)}>
          <option value="AAPL">AAPL</option>
          <option value="GOOGL">GOOGL</option>
          <option value="MSFT">MSFT</option>
        </select>
        <button onClick={fetchOpportunities}>Refresh</button>
        <button onClick={executeArbitrage} disabled={executing} className="execute-btn">
          {executing ? 'Executing...' : 'Execute All'}
        </button>
      </div>
      {message && <div className="arbitrage-message">{message}</div>}
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
  );
}

export default Arbitrage;
