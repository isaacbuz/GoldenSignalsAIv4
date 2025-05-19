import React, { useEffect, useState } from 'react';
import { API_URL, WS_URL } from '../config';
import './TradingDashboard.css';
import EChartsOptionTradeChart from './EChartsOptionTradeChart';
import WatchlistSidebar from './WatchlistSidebar';
import AlertsFeed from './AlertsFeed';
import ChartOverlayControls from './ChartOverlayControls';
import PerformanceWidgets from './PerformanceWidgets';
import BacktestPanel from './BacktestPanel';
import TradeJournal from './TradeJournal';

function TradingDashboard() {
  const [ohlcv, setOhlcv] = useState(null);
  const [ohlcvLoading, setOhlcvLoading] = useState(true);
  const [ohlcvError, setOhlcvError] = useState(null);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('D');
  const [user, setUser] = useState(null); // Placeholder for auth

  // Fetch OHLCV data with loading/error/retry logic, fallback to mock if failed
  useEffect(() => {
    setOhlcvLoading(true);
    setOhlcvError(null);
    fetch(`${API_URL}/api/ohlcv?symbol=${symbol}&timeframe=${timeframe}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(data => {
        setOhlcv(data);
        setOhlcvLoading(false);
      })
      .catch(() => {
        // Use mock data for demo
        setOhlcv([
          [1716105600000, 100, 105, 98, 104, 50000],
          [1716192000000, 104, 108, 102, 107, 60000],
          [1716278400000, 107, 110, 105, 109, 55000],
          [1716364800000, 109, 112, 108, 111, 58000],
          [1716451200000, 111, 115, 110, 113, 62000],
          [1716537600000, 113, 116, 112, 115, 61000],
        ]);
        setOhlcvError('Using placeholder data (API unavailable)');
        setOhlcvLoading(false);
      });
  }, [symbol, timeframe]);

  // WebSocket for live updates with error handling
  useEffect(() => {
    let ws;
    setWsStatus('connecting');
    try {
      ws = new window.WebSocket(`${WS_URL}/ws/ohlcv?symbol=${symbol}&timeframe=${timeframe}`);
      ws.onopen = () => setWsStatus('connected');
      ws.onmessage = () => {};
      ws.onerror = () => {
        setWsStatus('error');
      };
      ws.onclose = () => setWsStatus('closed');
    } catch {
      setWsStatus('error');
    }
    return () => ws && ws.close();
  }, [symbol, timeframe]);

  // Placeholder for authentication (to be replaced with Firebase auth)
  useEffect(() => {
    setUser({ name: 'Demo User', role: 'admin' });
  }, []);

  // Mock watchlist data for now
  const [watchlist] = useState([
    { symbol: 'AAPL', price: 175.2, change: 0.8, sparkline: '0,10 10,8 20,12 30,7 40,13 50,11 60,15' },
    { symbol: 'TSLA', price: 725.5, change: -2.1, sparkline: '0,12 10,15 20,10 30,8 40,7 50,9 60,6' },
    { symbol: 'GE', price: 102.9, change: 1.5, sparkline: '0,8 10,12 20,10 30,14 40,15 50,13 60,16' },
    { symbol: 'MSFT', price: 320.4, change: 0.2, sparkline: '0,14 10,13 20,12 30,12 40,13 50,14 60,15' },
  ]);

  // Chart overlays state
  const [overlays, setOverlays] = useState({ ma: true, ema: false, rsi: false, macd: false });
  const [backtestResult, setBacktestResult] = useState(null);

  return (
    <div className="dashboard-root">
      <header className="dashboard-header">
        <div className="brand" style={{ fontWeight: 800, fontSize: '2rem', letterSpacing: '0.03em', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', marginRight: 4 }}>
            <svg width="32" height="32" viewBox="0 0 32 32" style={{ verticalAlign: 'middle' }}>
              <defs>
                <radialGradient id="gold-gradient" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#FFFACD"/>
                  <stop offset="60%" stopColor="#FFD700"/>
                  <stop offset="100%" stopColor="#C9A100"/>
                </radialGradient>
              </defs>
              <circle cx="16" cy="16" r="14" fill="url(#gold-gradient)" stroke="#C9A100" strokeWidth="2"/>
              <circle cx="16" cy="16" r="8" fill="none" stroke="#FFFACD" strokeWidth="1.5"/>
            </svg>
          </span>
          <span style={{
            background: '#C9A100',
            color: '#181818',
            fontWeight: 900,
            fontSize: '2rem',
            letterSpacing: '0.01em',
            borderRadius: '14px',
            padding: '0.1em 1em',
            boxShadow: '0 2px 8px #C9A10055',
            display: 'inline-block',
            lineHeight: 1.2
          }}>
            GoldenSignalsAI
          </span>
        </div>
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
      <div className="responsive-main">
        <main className="dashboard-main">
          <div className="dashboard-content-row responsive-row">
            <div className="responsive-sidebar">
              <WatchlistSidebar watchlist={watchlist} selected={symbol} onSelect={setSymbol} />
            </div>
            <div className="dashboard-main-content">
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
                    {ohlcvLoading ? (
                      <span className="loading-spinner">Loading OHLCV...</span>
                    ) : ohlcv ? (
                      JSON.stringify(ohlcv, null, 2)
                    ) : 'No data.'}
                  </pre>
                  <div className="market-live-status">
                    <b>WebSocket:</b> <span className="market-ws-msg">{wsStatus}</span>
                    {ohlcvError && <span className="market-error">{ohlcvError}</span>}
                  </div>
                  {ohlcvError && (
                    <div style={{ color: '#FFD700', fontSize: '0.95em', marginTop: 4 }}>
                      <b>Notice:</b> Placeholder/mock data in use for demo.
                    </div>
                  )}
                </div>
              </section>

              <section className="dashboard-analytics">
                <h3>Analytics & Performance</h3>
                <div className="analytics-top-row" style={{ display: 'flex', flexDirection: 'row', gap: '1.5rem', justifyContent: 'center', alignItems: 'flex-start', marginBottom: '1.2rem', flexWrap: 'wrap' }}>
                  <div style={{ flex: '1 1 260px', minWidth: 180, maxWidth: 320 }}>
                    <PerformanceWidgets metrics={{
                      pnl: 4250.75,
                      winRate: 68,
                      drawdown: 12,
                      sharpe: 1.45,
                      trades: 57
                    }} />
                    {ohlcvError && (
                      <div style={{ color: '#FFD700', fontSize: '0.98em', margin: '0.5em 0 0.7em 0', textAlign: 'center' }}>
                        <b>Notice:</b> Placeholder analytics in use (backend offline)
                      </div>
                    )}
                  </div>
                  <div style={{ flex: '1 1 260px', minWidth: 180, maxWidth: 340 }}>
                    <BacktestPanel
                      initialParams={{ lookback: 60, threshold: 0.2, stop: 5, tp: 12 }}
                      onRunBacktest={() => setBacktestResult({ summary: `Mock: P&L $${(Math.random()*1000+4000).toFixed(2)}, Win ${Math.round(Math.random()*20+60)}%` })}
                      result={backtestResult}
                    />
                  </div>
                </div>
                <div className="analytics-chart-controls" style={{ maxWidth: 900, margin: '0 auto 0.5rem auto', display: 'flex', justifyContent: 'flex-end' }}>
                  <ChartOverlayControls overlays={overlays} onChange={setOverlays} />
                </div>
                <div className="analytics-charts">
                  <div className="analytics-chart chart-section" style={{ background: 'linear-gradient(120deg, #232323 80%, #181818 100%)', borderRadius: '18px', boxShadow: '0 6px 32px #FFD70022, 0 1.5px 12px #C9A10011', padding: '2rem', maxWidth: 900, minWidth: 320, margin: '0 auto' }}>
                    <EChartsOptionTradeChart 
                      ohlcv={ohlcv || []}
                      signals={[
                        { time: ohlcv?.[1]?.[0] || 1716192000000, price: ohlcv?.[1]?.[4] || 107, type: 'entry' },
                        { time: ohlcv?.[4]?.[0] || 1716451200000, price: ohlcv?.[4]?.[4] || 113, type: 'exit' },
                      ]}
                      loading={ohlcvLoading}
                      overlays={overlays}
                    />
                  </div>
                </div>
              </section>

              <div style={{ display: 'flex', justifyContent: 'center', width: '100%', margin: '2.5rem 0 0 0' }}>
                <TradeJournal />
              </div>

              <section className="dashboard-alerts">
                {/* AlertsFeed with mock data */}
                <AlertsFeed alerts={[
                  { id: '1', type: 'entry', symbol: 'AAPL', message: 'Entry signal @ $175.20', time: Date.now() - 60000, severity: 'entry' },
                  { id: '2', type: 'exit', symbol: 'TSLA', message: 'Exit signal @ $725.50', time: Date.now() - 30000, severity: 'exit' },
                  { id: '3', type: 'stop', symbol: 'GE', message: 'Stop-loss triggered @ $100.00', time: Date.now() - 15000, severity: 'stop' },
                  { id: '4', type: 'takeprofit', symbol: 'MSFT', message: 'Take-profit hit @ $322.00', time: Date.now() - 5000, severity: 'takeprofit' },
                  { id: '5', type: 'warning', symbol: 'AAPL', message: 'Unusual volume detected', time: Date.now() - 2000, severity: 'warning' },
                  { id: '6', type: 'error', symbol: 'TSLA', message: 'Signal fetch failed', time: Date.now() - 1000, severity: 'error' },
                ]} />
              </section>
            </div> {/* End dashboard-main-content */}
          </div> {/* End dashboard-content-row */}
        </main>
      </div>
      <footer className="dashboard-footer">
        <div>© {new Date().getFullYear()} GoldenSignalsAI. Inspired by Robinhood & eToro. All rights reserved.</div>
      </footer>
    </div>
  );
}

export default TradingDashboard;
