import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import TradingDashboard from './components/TradingDashboard';
import NewsSentiment from './pages/NewsSentiment';
import Watchlist from './pages/Watchlist';
import AgentPerformance from './pages/AgentPerformance';
import SearchBar from './components/SearchBar';
import SignalCard from './components/SignalCard';
import { sendSignalNotification, requestNotificationPermission } from './utils/Notifications';

function App() {
  const [signal, setSignal] = useState(null);

  useEffect(() => {
    requestNotificationPermission();

    const listener = (e) => {
      const detail = e.detail;
      sendSignalNotification({
        symbol: detail.symbol,
        type: detail.type,
        confidence: detail.confidence,
        price: detail.price,
      });
    };
    document.addEventListener('signal:new', listener);
    return () => document.removeEventListener('signal:new', listener);
  }, []);

  const handleSearch = async (symbol) => {
    try {
      const response = await fetch(`/signal/${symbol}`);
      const data = await response.json();
      setSignal(data);
    } catch (error) {
      console.error('Failed to fetch signal:', error);
    }
  };

  return (
    <Router>
      <nav style={{ display: 'flex', gap: 16, padding: 16, background: '#f7f7f7' }}>
        <Link to="/">Dashboard</Link>
        <Link to="/news">News & Sentiment</Link>
        <Link to="/watchlist">Watchlist</Link>
        <Link to="/agent-performance">Agent Performance</Link>
      </nav>
      <Routes>
        <Route
          path="/"
          element={
            <div className="max-w-xl mx-auto my-8">
              <h1 className="text-3xl font-bold mb-4">GoldenSignalsAI Dashboard</h1>
              <SearchBar onSearch={handleSearch} />
              {signal && <SignalCard signal={signal} />}
            </div>
          }
        />
        <Route path="/news" element={<NewsSentiment />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/agent-performance" element={<AgentPerformance />} />
      </Routes>
    </Router>
  );
}

export default App;
