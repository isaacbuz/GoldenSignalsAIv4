import React, { useEffect, useState, useCallback } from 'react';
import { API_URL, WS_URL } from '../config';
import TickerAutocomplete from './TickerAutocomplete';
import './TradingDashboard.css';
import EChartsOptionTradeChart from './EChartsOptionTradeChart';
import WatchlistSidebar from './WatchlistSidebar';
import AlertsFeed from './AlertsFeed';
import ChartOverlayControls from './ChartOverlayControls';
import PerformanceWidgets from './PerformanceWidgets';
import BacktestPanel from './BacktestPanel';
import TradeJournal from './TradeJournal';
import SignalFilters from './SignalFilters';
import GuidedTour from './GuidedTour';
import QuickActionToolbar from './QuickActionToolbar';
import ReplaySession from './ReplaySession';
import HeatmapOverview from './HeatmapOverview';
import EquityCurveChart from './EquityCurveChart';
import CorrelationMatrix from './CorrelationMatrix';
import NotificationSettings from './NotificationSettings';

// Simple error boundary for dashboard
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  componentDidCatch(error, errorInfo) {
    // You can log error here
  }
  render() {
    if (this.state.hasError) {
      return <div style={{ color: 'red', padding: 20 }}>Something went wrong in the dashboard UI.</div>;
    }
    return this.props.children;
  }
}

// Helper to compute metrics for PerformanceWidgets
function getPerformanceMetrics(historicalSignals) {
  if (!historicalSignals || historicalSignals.length === 0) {
    return { pnl: 0, winRate: 0, drawdown: 0, sharpe: 0, trades: 0 };
  }
  const trades = historicalSignals.length;
  const pnl = historicalSignals.reduce((sum, s) => sum + (s.pnl || 0), 0);
  const wins = historicalSignals.filter(s => s.outcome === 'Win').length;
  const losses = historicalSignals.filter(s => s.outcome === 'Loss').length;
  const winRate = trades > 0 ? Math.round((wins / trades) * 100) : 0;
  const drawdown = 5; // Placeholder, compute if you have equity curve
  const sharpe = 1.2; // Placeholder, compute if you have returns
  return { pnl, winRate, drawdown, sharpe, trades };
}


function TradingDashboard() {
  const [ohlcvData, setOhlcvData] = useState([]);
  const [signals, setSignals] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [historicalSignals, setHistoricalSignals] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [overlays, setOverlays] = useState({ ma: false, ema: false, rsi: false, macd: false });
  // Watchlist as array of objects for WatchlistSidebar
  const [watchlist, setWatchlist] = useState([
    { symbol: 'AAPL', price: 190.12, change: 0.8, sparkline: '5,10,8,12,9,14,10' },
    { symbol: 'MSFT', price: 320.44, change: -0.5, sparkline: '8,12,10,9,13,11,15' },
    { symbol: 'GOOGL', price: 2725.67, change: 1.2, sparkline: '10,9,12,14,13,15,13' }
  ]);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showWelcome, setShowWelcome] = useState(!localStorage.getItem('hasSeenWelcome'));
  const [activeTab, setActiveTab] = useState('alerts');
  const [timeframe, setTimeframe] = useState('daily');
  const [portfolio, setPortfolio] = useState([]); // Store user trades
  const [tradeSuggestion, setTradeSuggestion] = useState(null);
  const [suggestionLoading, setSuggestionLoading] = useState(false);
  const [guidedTourCompleted, setGuidedTourCompleted] = useState(localStorage.getItem('guidedTourCompleted') === 'true');
  const [quickActionToolbarVisible, setQuickActionToolbarVisible] = useState(false);
  const [quickActionToolbarSignal, setQuickActionToolbarSignal] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_URL}/stock_data?symbol=${selectedSymbol}&timeframe=${timeframe}`);
        if (!response.ok) throw new Error('Failed to fetch stock data');
        const data = await response.json();
        setOhlcvData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    const fetchTradeSuggestion = async () => {
      setSuggestionLoading(true);
      try {
        const resp = await fetch(`${API_URL}/signal/trade_suggestion`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: selectedSymbol, timeframe: timeframe === 'daily' ? 'D' : timeframe, risk_profile: 'balanced' })
        });
        if (!resp.ok) throw new Error('Failed to fetch trade suggestion');
        const suggestion = await resp.json();
        setTradeSuggestion(suggestion);
      } catch (err) {
        setTradeSuggestion(null);
      } finally {
        setSuggestionLoading(false);
      }
    };

    const ws = new WebSocket(WS_URL + '/ws/signals');
    ws.onmessage = (event) => {
      const newSignal = JSON.parse(event.data);
      if (newSignal.symbol === selectedSymbol) {
        setSignals((prev) => [...prev, newSignal].slice(-50));
        setHistoricalSignals((prev) => [...prev, { ...newSignal, outcome: 'Pending', timestamp: new Date().toISOString() }].slice(-100));
        if (newSignal.action !== 'hold' && newSignal.confidence > 0.9) {
          const enrichedSignal = {
            id: Date.now(),
            ...newSignal,
            entryPrice: 2.00,
            targetPrice: 2.50,
            stopLoss: 1.50,
            riskReward: '1:1',
            timeHorizon: timeframe === '5m' ? '15 minutes' : timeframe === '1h' ? '1 hour' : '1 day',
            supportingIndicators: 'Price above VWAP, RSI at 65',
            historicalWinRate: 90,
            rationale: 'This signal was generated due to a breakout above the 20-period moving average with high volume, combined with positive news sentiment.',
          };
          setAlerts((prev) => [...prev, enrichedSignal].slice(-20));
        }
      }
    };
    ws.onclose = () => setError('WebSocket connection closed');
    ws.onerror = () => setError('WebSocket error');

    fetchData();
    fetchTradeSuggestion();

    return () => ws.close();
  }, [selectedSymbol, timeframe]);

  useEffect(() => {
    setFilteredAlerts(alerts);
  }, [alerts]);

  const handleFilterChange = (filters) => {
    const filtered = alerts.filter((alert) => {
      const confidenceMatch = (alert.confidence * 100) >= filters.minConfidence;
      const riskRewardMatch = filters.minRiskReward === '1:1' || 
        (filters.minRiskReward === '1:2' && alert.riskReward !== '1:1') ||
        (filters.minRiskReward === '1:3' && alert.riskReward === '1:3');
      const timeHorizonMatch = filters.timeHorizon === 'any' ||
        (filters.timeHorizon === 'scalping' && alert.timeHorizon.includes('minutes')) ||
        (filters.timeHorizon === 'swing' && alert.timeHorizon.includes('days'));
      const optionTypeMatch = filters.optionType === 'both' ||
        (filters.optionType === 'calls' && alert.action.toLowerCase().includes('buy')) ||
        (filters.optionType === 'puts' && alert.action.toLowerCase().includes('sell'));
      return confidenceMatch && riskRewardMatch && timeHorizonMatch && optionTypeMatch;
    });
    setFilteredAlerts(filtered);
  };

  const handleCloseWelcome = () => {
    localStorage.setItem('hasSeenWelcome', 'true');
    setShowWelcome(false);
  };

  const handleAddToPortfolio = (signal) => {
    const trade = {
      id: Date.now(),
      symbol: signal.symbol,
      action: signal.action,
      entryPrice: signal.entryPrice,
      quantity: 1, // Default quantity
      timestamp: new Date().toISOString(),
      status: 'Open',
    };
    setPortfolio((prev) => [...prev, trade]);
  };

  const handleHoverSignal = (signal) => {
    setHoveredSignal(signal);
    setQuickActionToolbarVisible(true);
    setQuickActionToolbarSignal(signal);
  };

  const handleHideQuickActionToolbar = () => {
    setQuickActionToolbarVisible(false);
    setQuickActionToolbarSignal(null);
  };

  return (
    <ErrorBoundary>
      <div className="dashboard-root">
        {showWelcome && (
          <div className="welcome-modal">
            <div className="welcome-content">
              <h2>Welcome to GoldenSignalsAi</h2>
              <p>Explore advanced options trading insights with real-time data and analytics.</p>
              <button onClick={handleCloseWelcome} className="welcome-close">
                Get Started
              </button>
            </div>
          </div>
        )}
        {!guidedTourCompleted && (
          <GuidedTour
            onComplete={() => {
              localStorage.setItem('guidedTourCompleted', 'true');
              setGuidedTourCompleted(true);
            }}
          />
        )}
        <header className="dashboard-header redesigned-header">
          <div className="dashboard-app-brand">
            <span className="app-logo" aria-label="GoldenSignalsAi" title="GoldenSignalsAi">&#11044;</span>
            <span className="app-name">GoldenSignalsAi</span>
          </div>
          <nav className="dashboard-nav">
            <a href="#" className="nav-link active">Dashboard</a>
            <a href="#" className="nav-link">Scanner</a>
            <a href="#" className="nav-link">Journal</a>
            <a href="#" className="nav-link">Settings</a>
          </nav>
          <div className="dashboard-header-actions">
            <button className="profile-btn" aria-label="User Profile">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="14" cy="14" r="14" fill="#FFD700"/>
                <circle cx="14" cy="12" r="5" fill="#232323"/>
                <ellipse cx="14" cy="21" rx="7" ry="3" fill="#232323"/>
              </svg>
            </button>
          </div>
        </header>
        <main className="dashboard-main scalable-layout">
          {error && <div className="error-message">{error}</div>}
          <div className="dashboard-grid">
            <aside className="dashboard-sidebar scalable-sidebar">
              <WatchlistSidebar
                watchlist={watchlist}
                selected={selectedSymbol}
                onSelect={setSelectedSymbol}
              />
            </aside>
            <section className="dashboard-center scalable-center">
              <section className="dashboard-market scalable-card">
                <h2>Market Overview</h2>
                {/* --- Actionable Trade Suggestion Card --- */}
                <div className="trade-suggestion-card" style={{ background: '#f9fafb', border: '1.5px solid #FFD70044', borderRadius: 12, padding: 20, marginBottom: 18, boxShadow: '0 2px 12px #FFD70011' }}>
                  {suggestionLoading ? (
                    <div style={{ color: '#888', fontWeight: 500 }}>Loading trade suggestion...</div>
                  ) : tradeSuggestion ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                        <span style={{ fontWeight: 700, fontSize: '1.15em', color: tradeSuggestion.direction === 'up' ? '#22c55e' : tradeSuggestion.direction === 'down' ? '#f43f5e' : '#64748b' }}>
                          {tradeSuggestion.direction === 'up' ? '↑ Bullish' : tradeSuggestion.direction === 'down' ? '↓ Bearish' : '→ Neutral'}
                        </span>
                        <span style={{ background: '#FFD70022', color: '#FFD700', borderRadius: 6, padding: '2px 10px', fontWeight: 600, fontSize: '0.98em' }}>{tradeSuggestion.action.replace('_', ' ').toUpperCase()}</span>
                        <span style={{ marginLeft: 8, fontWeight: 500, color: '#334155' }}>Confidence: <b>{(tradeSuggestion.confidence * 100).toFixed(1)}%</b></span>
                      </div>
                      <div style={{ fontSize: '1.01em', marginBottom: 4 }}>
                        <b>Entry:</b> ${tradeSuggestion.entry_price.toFixed(2)} &nbsp; | &nbsp; <b>Stop Loss:</b> ${tradeSuggestion.stop_loss.toFixed(2)} &nbsp; | &nbsp; <b>Take Profit:</b> ${tradeSuggestion.take_profit.toFixed(2)}
                      </div>
                      <div style={{ fontSize: '0.97em', color: '#888', marginBottom: 2 }}>
                        <b>Rationale:</b> {tradeSuggestion.rationale && typeof tradeSuggestion.rationale === 'object' ?
                          `Model: ${tradeSuggestion.rationale.direction}, Δ: ${tradeSuggestion.rationale.predicted_change.toFixed(3)}, LSTM: ${tradeSuggestion.rationale.model_scores.lstm.toFixed(3)}, XGB: ${tradeSuggestion.rationale.model_scores.xgboost.toFixed(3)}, LGB: ${tradeSuggestion.rationale.model_scores.lightgbm.toFixed(3)}`
                          : String(tradeSuggestion.rationale)}
                      </div>
                      <div style={{ fontSize: '0.93em', color: '#aaa' }}>Generated: {new Date(tradeSuggestion.timestamp).toLocaleString()}</div>
                    </>
                  ) : (
                    <div style={{ color: '#f43f5e', fontWeight: 500 }}>No trade suggestion available for this symbol/timeframe.</div>
                  )}
                </div>
                <div className="market-symbol-picker" style={{ display: 'flex', alignItems: 'center', gap: '1.2em', marginBottom: 16 }}>
                  <label htmlFor="symbol-input">Ticker Symbol: </label>
                  <TickerAutocomplete 
                    id="symbol-input"
                    value={selectedSymbol}
                    onChange={setSelectedSymbol}
                    onSelect={setSelectedSymbol}
                    placeholder="AAPL"
                    style={{ width: 90, padding: '6px 8px', border: '1px solid #e5e7eb', borderRadius: 6, fontSize: '1rem', marginRight: 8 }}
                    aria-label="Enter ticker symbol"
                  />
                  <label htmlFor="timeframe">Select Timeframe: </label>
                  <select
                    id="timeframe"
                    value={timeframe}
                    onChange={(e) => setTimeframe(e.target.value)}
                  >
                    <option value="5m">5 Minutes</option>
                    <option value="1h">1 Hour</option>
                    <option value="daily">Daily</option>
                  </select>
                </div>
                <div className="analytics-charts">
                  <EChartsOptionTradeChart ohlcv={ohlcvData} signals={signals} loading={loading} onHoverSignal={handleHoverSignal} />
                </div>
                <ChartOverlayControls overlays={overlays} onChange={setOverlays} />
              </section>
              <section className="scalable-row">
                <PerformanceWidgets metrics={getPerformanceMetrics(historicalSignals)} />
                <SignalFilters onFilterChange={handleFilterChange} />
              </section>
              <section className="scalable-row" aria-label="Equity Curve Section">
                <EquityCurveChart trades={portfolio} />
              </section>
              <section className="scalable-row" aria-label="Correlation Matrix Section">
                <CorrelationMatrix 
  symbols={watchlist.map(w => w.symbol)}
  fetchCorrelation={async (symbols) => {
    // Prevent duplicate requests
    if (window.__correlationLoading) return;
    window.__correlationLoading = true;
    let error = null;
    let warning = null;
    let loadingToast;
    try {
      // Show loading toast or inline UI
      if (window.toast) loadingToast = window.toast('Calculating correlation...', { type: 'info', autoClose: false });
      // Helper to fetch close prices for a symbol
      async function getCloses(symbol) {
        try {
          if (symbol === selectedSymbol && ohlcvData && ohlcvData.length > 0) {
            return ohlcvData.map(row => row[4]);
          }
          const resp = await fetch(`${API_URL}/stock_data?symbol=${symbol}&timeframe=${timeframe}`);
          if (!resp.ok) throw new Error('Failed to fetch data for ' + symbol);
          const data = await resp.json();
          return data.map(row => row[4]);
        } catch (e) {
          error = `Failed to fetch data for ${symbol}`;
          return [];
        }
      }
      // Fetch all in parallel
      const priceData = {};
      await Promise.all(symbols.map(async s => {
        priceData[s] = await getCloses(s);
      }));
      // Align lengths
      const minLen = Math.min(...Object.values(priceData).map(arr => arr.length).filter(len => len > 0));
      if (minLen < 20) {
        warning = 'Not enough data for correlation analysis.';
      }
      Object.keys(priceData).forEach(s => {
        if (priceData[s].length > minLen) priceData[s] = priceData[s].slice(-minLen);
        if (priceData[s].length < minLen) priceData[s] = Array(minLen - priceData[s].length).fill(null).concat(priceData[s]);
      });
      const resp = await fetch(`${API_URL}/correlation/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: priceData })
      });
      if (!resp.ok) throw new Error('Failed to fetch correlation');
      const result = await resp.json();
      if (loadingToast && window.toast) window.toast.dismiss(loadingToast);
      if (warning && window.toast) window.toast(warning, { type: 'warning' });
      if (error && window.toast) window.toast(error, { type: 'error' });
      window.__correlationLoading = false;
      return result;
    } catch (e) {
      if (loadingToast && window.toast) window.toast.dismiss(loadingToast);
      if (window.toast) window.toast(e.message || 'Correlation calculation failed', { type: 'error' });
      window.__correlationLoading = false;
      throw e;
    }
  }}
/>
              </section>
              <section className="scalable-row" aria-label="Notification Settings Section">
                <NotificationSettings onSave={async (settings) => {
                  try {
                    const resp = await fetch(`${API_URL}/user/notifications`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(settings),
                      credentials: 'include',
                    });
                    if (!resp.ok) throw new Error('Failed to save notification settings');
                    if (window.toast) window.toast('Notification settings saved!', { type: 'success' });
                  } catch (e) {
                    if (window.toast) window.toast(e.message || 'Failed to save notification settings', { type: 'error' });
                  }
                }} />
              </section>
              <section className="scalable-row">
                <BacktestPanel 
                  onRunBacktest={(params) => { /* TODO: implement backtest logic */ }}
                  initialParams={{ lookback: 30, threshold: 0.5, stop: 10, tp: 20 }}
                  result={undefined}
                />
                <AlertsFeed alerts={filteredAlerts} onAddToPortfolio={handleAddToPortfolio} onHoverSignal={handleHoverSignal} />
              </section>
              <section className="scalable-row">
                <TradeJournal />
              </section>
              {quickActionToolbarVisible && (
                <QuickActionToolbar
                  signal={quickActionToolbarSignal}
                  onHide={handleHideQuickActionToolbar}
                />
              )}
              <ReplaySession />
              <HeatmapOverview />
            </section>
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
}

export default TradingDashboard;
