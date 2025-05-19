import React from 'react';
import PropTypes from 'prop-types';
import './PerformanceWidgets.css';

/**
 * PerformanceWidgets displays P&L, win rate, drawdown, Sharpe ratio, and more.
 * @param {object} metrics - { pnl, winRate, drawdown, sharpe, trades, ... }
 */
export default function PerformanceWidgets({ metrics }) {
  return (
    <div className="performance-widgets">
      <div className="perf-widget perf-pnl">
        <div className="perf-label">P&L</div>
        <div className="perf-value">${metrics.pnl > 0 ? '+' : ''}{metrics.pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
      </div>
      <div className="perf-widget perf-winrate">
        <div className="perf-label">Win Rate</div>
        <div className="perf-value">{metrics.winRate}%</div>
      </div>
      <div className="perf-widget perf-drawdown">
        <div className="perf-label">Drawdown</div>
        <div className="perf-value">{metrics.drawdown}%</div>
      </div>
      <div className="perf-widget perf-sharpe">
        <div className="perf-label">Sharpe</div>
        <div className="perf-value">{metrics.sharpe}</div>
      </div>
      <div className="perf-widget perf-trades">
        <div className="perf-label">Trades</div>
        <div className="perf-value">{metrics.trades}</div>
      </div>
    </div>
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
