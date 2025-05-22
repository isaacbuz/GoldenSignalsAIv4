import React from 'react';
import './PerformanceSummary.css';
import { FaChartLine, FaTrophy, FaArrowUp, FaArrowDown } from 'react-icons/fa';

export default function PerformanceSummary({ stats }) {
  // stats: { winRate, totalSignals, avgPL, bestSignal, worstSignal }
  return (
    <div className="performance-summary card">
      <div className="perf-header">
        <FaChartLine className="perf-icon" />
        <span>Performance Summary</span>
      </div>
      <div className="perf-stats">
        <div className="perf-stat">
          <span className="stat-label">Win Rate</span>
          <span className="stat-value win-rate">{stats.winRate}%</span>
        </div>
        <div className="perf-stat">
          <span className="stat-label">Total Signals</span>
          <span className="stat-value">{stats.totalSignals}</span>
        </div>
        <div className="perf-stat">
          <span className="stat-label">Avg P/L</span>
          <span className={`stat-value ${stats.avgPL >= 0 ? 'pl-pos' : 'pl-neg'}`}>{stats.avgPL}%</span>
        </div>
        <div className="perf-stat best">
          <FaTrophy className="best-icon" />
          <span className="stat-label">Best</span>
          <span className="stat-value">{stats.bestSignal}</span>
        </div>
        <div className="perf-stat worst">
          <FaArrowDown className="worst-icon" />
          <span className="stat-label">Worst</span>
          <span className="stat-value">{stats.worstSignal}</span>
        </div>
      </div>
    </div>
  );
}
