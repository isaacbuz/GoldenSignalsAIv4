// presentation/frontend/src/App.js
// Purpose: Implements the React frontend for GoldenSignalsAI, providing a dashboard for
// viewing trading signals, agent activity, and market data, with options trading-specific features.

import React, { useState, useEffect } from 'react';
import './App.css';
import Arbitrage from './Arbitrage';
import './Arbitrage.css';
import AdminPanel from "./AdminPanel";
import ErrorBoundary from "./ErrorBoundary";
import { setupTokenExpiryListener, auth } from "./firebase";
import './AdminPanel.css';

function App() {
  const [activeTab, setActiveTab] = useState('signals');
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
    if (token && activeTab === 'signals') {
      fetch(`http://localhost:8000/dashboard/${symbol}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => response.json())
        .then(data => setData(data))
        .catch(error => console.error('Error fetching dashboard data:', error));
    }
  }, [symbol, token, activeTab]);

  return (
    <div className="App">
      {tokenExpired && (
        <div style={{ background: "#ff5252", color: "#fff", padding: 16, textAlign: "center", position: "fixed", top: 0, left: 0, width: "100%", zIndex: 10000 }}>
          Your session has expired. You will be redirected to login.
        </div>
      )}
      <nav className="tab-nav">
        <button
          className={activeTab === "signals" ? "active" : ""}
          onClick={() => setActiveTab("signals")}
        >
          Trading Signals
        </button>
        <button
          className={activeTab === "arbitrage" ? "active" : ""}
          onClick={() => setActiveTab("arbitrage")}
        >
          Arbitrage
        </button>
        <button
          className={activeTab === "admin" ? "active" : ""}
          onClick={() => setActiveTab("admin")}
        >
          Admin
        </button>
      </nav>
      <main>
        {activeTab === "signals" && (
          <>
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
          </>
        )}
        {activeTab === "arbitrage" && (
          <Arbitrage />
        )}
        {activeTab === "admin" && (
          <ErrorBoundary><AdminPanel /></ErrorBoundary>
        )}
      </main>
    </div>
  );
}

export default App;
