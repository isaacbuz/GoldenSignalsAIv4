import React from 'react';
import PropTypes from 'prop-types';
import './PerformanceWidgets.css';

/**
 * PerformanceWidgets displays P&L, win rate, drawdown, Sharpe ratio, and more.
 * @param {object} metrics - { pnl, winRate, drawdown, sharpe, trades, ... }
 */
import { motion, AnimatePresence } from 'framer-motion';

export default function PerformanceWidgets({ metrics }) {
  return (
    <motion.div
      className="performance-widgets single-card"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 24 }}
      transition={{ duration: 0.45, type: 'spring', stiffness: 60 }}
      tabIndex={0}
      aria-label="Performance Metrics"
    >
      <div className="perf-metrics-grid">
        <div className="perf-metric perf-pnl">
          <span className="perf-label" tabIndex={0} aria-label="P&L">P&amp;L
            <span className="perf-tooltip">Profit &amp; Loss: Net gain or loss over the period.</span>
          </span>
          <span className="perf-value">{metrics.pnl > 0 ? '+' : ''}{metrics.pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
        </div>
        <div className="perf-metric perf-winrate">
          <span className="perf-label" tabIndex={0} aria-label="Win Rate">Win Rate
            <span className="perf-tooltip">Win Rate: Percentage of winning trades.</span>
          </span>
          <span className="perf-value">{metrics.winRate}%</span>
        </div>
        <div className="perf-metric perf-drawdown">
          <span className="perf-label" tabIndex={0} aria-label="Drawdown">Drawdown
            <span className="perf-tooltip">Drawdown: Maximum observed loss from a peak.</span>
          </span>
          <span className="perf-value">{metrics.drawdown}%</span>
        </div>
        <div className="perf-metric perf-sharpe">
          <span className="perf-label" tabIndex={0} aria-label="Sharpe">Sharpe
            <span className="perf-tooltip">Sharpe Ratio: Risk-adjusted return.</span>
          </span>
          <span className="perf-value">{metrics.sharpe}</span>
        </div>
        <div className="perf-metric perf-trades">
          <span className="perf-label" tabIndex={0} aria-label="Trades">Trades
            <span className="perf-tooltip">Total number of trades.</span>
          </span>
          <span className="perf-value">{metrics.trades}</span>
        </div>
      </div>
    </motion.div>
  );
}

PerformanceWidgets.propTypes = {
  metrics: PropTypes.shape({
    pnl: PropTypes.number.isRequired,
    winRate: PropTypes.number.isRequired,
    drawdown: PropTypes.number.isRequired,
    sharpe: PropTypes.number.isRequired,
    trades: PropTypes.number.isRequired,
  }).isRequired,
};
