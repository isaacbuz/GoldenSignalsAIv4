// presentation/frontend/src/App.js
// Purpose: Implements the React frontend for GoldenSignalsAI, providing a dashboard for
// viewing trading signals, agent activity, and market data, with options trading-specific features.

import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [symbol, setSymbol] = useState('AAPL');
  const [data, setData] = useState(null);
  const [token, setToken] = useState(null);

  useEffect(() => {
    // Login to get token
    fetch('http://localhost:8000/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        'username': 'user1',
        'password': 'password1'
      })
    })
      .then(response => response.json())
      .then(data => setToken(data.access_token))
      .catch(error => console.error('Login failed:', error));
  }, []);

  useEffect(() => {
    if (token) {
      fetch(`http://localhost:8000/dashboard/${symbol}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => response.json())
        .then(data => setData(data))
        .catch(error => console.error('Error fetching dashboard data:', error));
    }
  }, [symbol, token]);

  return (
    <div className="App">
      <h1>GoldenSignalsAI Dashboard</h1>
      <select onChange={(e) => setSymbol(e.target.value)} value={symbol}>
        <option value="AAPL">AAPL</option>
        <option value="GOOGL">GOOGL</option>
        <option value="MSFT">MSFT</option>
      </select>
      {data ? (
        <div>
          <h2>{data.symbol}</h2>
          <p>Price: ${data.price}</p>
          <p>Trend: {data.trend}</p>
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default App;
