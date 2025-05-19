import React from 'react';
import PropTypes from 'prop-types';
import './AlertsFeed.css';

/**
 * AlertsFeed displays a real-time feed of trading alerts and notifications.
 * @param {Array} alerts - Array of { id, type, symbol, message, time, severity }
 */
export default function AlertsFeed({ alerts }) {
  return (
    <aside className="alerts-feed">
      <h3>Alerts & Notifications</h3>
      <ul>
        {alerts.length === 0 && <li className="alert-empty">No alerts.</li>}
        {alerts.map(alert => {
  // Demo lookup tables for tooltips
  const symbolNames = {
    AAPL: 'Apple Inc.',
    TSLA: 'Tesla, Inc.',
    GE: 'General Electric',
    MSFT: 'Microsoft Corp.'
  };
  const typeDescriptions = {
    entry: 'Entry signal: Opportunity to enter a trade',
    exit: 'Exit signal: Opportunity to close a trade',
    stop: 'Stop-loss triggered',
    takeprofit: 'Take-profit hit',
    warning: 'Warning: Check this alert',
    error: 'Error in signal or data'
  };
  return (
    <li key={alert.id} className={`alert-item alert-${alert.severity || alert.type}`}>
      <div
        className="alert-symbol"
        title={symbolNames[alert.symbol] || alert.symbol}
      >
        {alert.symbol}
      </div>
      <div className="alert-message">{alert.message}</div>
      <div className="alert-meta">
        <span
          className="alert-type"
          title={typeDescriptions[alert.type] || alert.type}
        >
          {alert.type}
        </span>
        <span className="alert-time">{formatTime(alert.time)}</span>
      </div>
    </li>
  );
})}
      </ul>
    </aside>
  );
}

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

AlertsFeed.propTypes = {
  alerts: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    symbol: PropTypes.string,
    message: PropTypes.string.isRequired,
    time: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    severity: PropTypes.string,
  })).isRequired,
};
