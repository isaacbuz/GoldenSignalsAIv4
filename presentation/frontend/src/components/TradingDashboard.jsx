import React, { useEffect, useState } from 'react';
import { API_URL, WS_URL } from '../config';
import './TradingDashboard.css';

function TradingDashboard() {
  const [apiHealth, setApiHealth] = useState('');
  const [apiHealthLoading, setApiHealthLoading] = useState(true);
  const [apiHealthError, setApiHealthError] = useState(null);
  const [ohlcv, setOhlcv] = useState(null);
  const [ohlcvLoading, setOhlcvLoading] = useState(true);
  const [ohlcvError, setOhlcvError] = useState(null);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [wsMsg, setWsMsg] = useState('');
  const [wsError, setWsError] = useState(null);
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('D');
  const [user, setUser] = useState(null); // Placeholder for auth

  // API health check with loading/error states
  useEffect(() => {
    setApiHealthLoading(true);
    setApiHealthError(null);
    fetch(`${API_URL}/health`)
      .then(r => r.ok ? r.text() : Promise.reject(r.statusText))
      .then(data => {
        setApiHealth(data);
        setApiHealthLoading(false);
      })
      .catch(e => {
        setApiHealth('API not reachable');
        setApiHealthError(e.toString());
        setApiHealthLoading(false);
      });
  }, []);

  // Fetch OHLCV data with loading/error/retry logic
  useEffect(() => {
    let retryCount = 0;
    const fetchOhlcv = () => {
      setOhlcvLoading(true);
      setOhlcvError(null);
      fetch(`${API_URL}/api/ohlcv?symbol=${symbol}&timeframe=${timeframe}`)
        .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
        .then(data => {
          setOhlcv(data);
          setOhlcvLoading(false);
        })
        .catch(e => {
          setOhlcv(null);
          setOhlcvError(e.toString());
          setOhlcvLoading(false);
          if (retryCount < 2) {
            retryCount++;
            setTimeout(fetchOhlcv, 1000 * retryCount); // Exponential backoff
          }
        });
    };
    fetchOhlcv();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, timeframe]);

  // WebSocket for live updates with error handling
  useEffect(() => {
    let ws;
    setWsStatus('connecting');
    setWsError(null);
    try {
      ws = new window.WebSocket(`${WS_URL}/ws/ohlcv?symbol=${symbol}&timeframe=${timeframe}`);
      ws.onopen = () => setWsStatus('connected');
      ws.onmessage = (e) => setWsMsg(e.data);
      ws.onerror = (e) => {
        setWsStatus('error');
        setWsError('WebSocket connection failed');
      };
      ws.onclose = () => setWsStatus('closed');
    } catch (e) {
      setWsStatus('error');
      setWsError(e.toString());
    }
    return () => ws && ws.close();
  }, [symbol, timeframe]);

  // Placeholder for authentication (to be replaced with Firebase auth)
  useEffect(() => {
    setUser({ name: 'Demo User', role: 'admin' });
  }, []);

  return (
    <div className="dashboard-root">
      <header className="dashboard-header">
        <div className="brand">GoldenSignalsAI <span className="brand-ai">AI</span></div>
        <nav className="dashboard-nav">
          <span>Dashboard</span>
          <span>Agents</span>
          <span>Analytics</span>
          <span>Admin</span>
        </nav>
        <div className="dashboard-user">
          {user ? <span>{user.name} ({user.role})</span> : <span>Sign In</span>}
        </div>
      </header>
      <main className="dashboard-main">
        <section className="dashboard-market">
          <div className="market-header">
            <h2>Market Overview</h2>
            <div className="market-symbol-picker">
              <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} placeholder="Symbol" />
              <select value={timeframe} onChange={e => setTimeframe(e.target.value)}>
                <option value="D">1D</option>
                <option value="1h">1H</option>
                <option value="5m">5M</option>
              </select>
            </div>
          </div>
          <div className="market-ohlcv">
            <b>OHLCV Data:</b>
            <pre className="market-ohlcv-data">
              {ohlcvLoading ? 'Loading...' : ohlcvError ? `Failed to load: ${ohlcvError}` : ohlcv ? JSON.stringify(ohlcv, null, 2) : 'No data.'}
            </pre>
            <div className="market-live-status">
              <b>WebSocket:</b> {wsStatus} <span className="market-ws-msg">{wsMsg}</span>
              {wsError && <span className="market-error">WebSocket error: {wsError}</span>}
            </div>
          </div>
        </section>
        <section className="dashboard-agents">
          <h3>Agent Status</h3>
          <div className="agent-status-list">
            {/* Placeholder for agent health, performance, and controls */}
            <div className="agent-status-card">Arbitrage Agent: <span className="agent-status agent-status-ok">Healthy</span></div>
            <div className="agent-status-card">Trading Agent: <span className="agent-status agent-status-warning">Warning</span></div>
          </div>
        </section>
        <section className="dashboard-analytics">
          <h3>Analytics & Performance</h3>
          <div className="analytics-charts">
            {/* Placeholder for charts: CPU, memory, uptime, queue, error rates */}
            <div className="analytics-chart analytics-chart-placeholder">[Charts Coming Soon]</div>
          </div>
        </section>
        <section className="dashboard-alerts">
          <h3>Alerts</h3>
          <div className="alerts-list">
            {/* Placeholder for alerts */}
            <div className="alert alert-ok">All systems operational.</div>
          </div>
        </section>
      </main>
      <footer className="dashboard-footer">
        <div>© {new Date().getFullYear()} GoldenSignalsAI. Inspired by Robinhood & eToro. All rights reserved.</div>
      </footer>
    </div>
  );
}

export default TradingDashboard;
