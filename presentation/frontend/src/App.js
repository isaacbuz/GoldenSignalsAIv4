// presentation/frontend/src/App.js
// Purpose: Implements the React frontend for GoldenSignalsAI, providing a dashboard for
// viewing trading signals, agent activity, and market data, with options trading-specific features.

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import './App.css';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import Arbitrage from './pages/Arbitrage';
import Admin from './pages/Admin';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="main-nav">
          <ul>
            <li><Link to="/">Dashboard</Link></li>
            <li><Link to="/scanner">Scanner</Link></li>
            <li><Link to="/arbitrage">Arbitrage</Link></li>
            <li><Link to="/admin">Admin</Link></li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/arbitrage" element={<Arbitrage />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
