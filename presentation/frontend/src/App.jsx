import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import TradingDashboard from './components/TradingDashboard';
import NewsSentiment from './pages/NewsSentiment';
import Watchlist from './pages/Watchlist';
import AgentPerformance from './pages/AgentPerformance';

function App() {
  return (
    <Router>
      <nav style={{ display: 'flex', gap: 16, padding: 16, background: '#f7f7f7' }}>
        <Link to="/">Dashboard</Link>
        <Link to="/news">News & Sentiment</Link>
        <Link to="/watchlist">Watchlist</Link>
        <Link to="/agent-performance">Agent Performance</Link>
      </nav>
      <Routes>
        <Route path="/" element={<TradingDashboard />} />
        <Route path="/news" element={<NewsSentiment />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/agent-performance" element={<AgentPerformance />} />
      </Routes>
    </Router>
  );
}

export default App;
