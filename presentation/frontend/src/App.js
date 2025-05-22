// presentation/frontend/src/App.js
// Purpose: Implements the React frontend for GoldenSignalsAI, providing a dashboard for
// viewing trading signals, agent activity, and market data, with options trading-specific features.

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import './App.css';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import Arbitrage from './pages/Arbitrage';
import Admin from './pages/Admin';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(true);

  const toggleTheme = () => {
    setIsDarkMode((prev) => !prev);
    document.body.classList.toggle('light-mode', !isDarkMode);
  };

  return (
    <Router>
      <div className={`App ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
        <header className="sticky-header">
          <div className="header-content">
            <div className="logo">
              <Link to="/">
                <h1>GoldenSignalsAI</h1>
              </Link>
            </div>
            <nav className="main-nav">
              <ul>
                <li><Link to="/">Dashboard</Link></li>
                <li><Link to="/scanner">Scanner</Link></li>
                <li><Link to="/arbitrage">Arbitrage</Link></li>
                <li><Link to="/admin">Admin</Link></li>
              </ul>
            </nav>
            <div className="user-controls">
              <button onClick={toggleTheme} className="theme-toggle">
                {isDarkMode ? 'Light Mode' : 'Dark Mode'}
              </button>
              <div className="user-profile">
                <span>User</span>
                <div className="dropdown">
                  <Link to="/profile">Profile</Link>
                  <Link to="/logout">Logout</Link>
                </div>
              </div>
            </div>
          </div>
        </header>
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
