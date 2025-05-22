import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './AlertsFeed.css';
import { motion, AnimatePresence } from 'framer-motion';

const Tooltip = ({ text, children }) => (
  <span className="tooltip-container">
    {children}
    <span className="tooltip-text">{text}</span>
  </span>
);

/**
 * AlertsFeed displays a real-time feed of trading alerts and notifications.
 * @param {Array} alerts - Array of { id, type, symbol, message, time, severity }
 */

export default function AlertsFeed({ alerts, onAddToPortfolio }) {
  const [selectedSignal, setSelectedSignal] = useState(null);
  const [visibleAlerts, setVisibleAlerts] = useState(alerts);

  const getSignalStrength = (confidence, historicalWinRate) => {
    if (confidence > 90) {
      return { emoji: '🟢', strength: 'Strong' };
    } else if (confidence > 70) {
      return { emoji: '🟡', strength: 'Moderate' };
    } else {
      return { emoji: '🔴', strength: 'Weak' };
    }
  };

  const handleDismiss = (id) => {
    setVisibleAlerts(visibleAlerts.filter((alert) => alert.id !== id));
  };

  const handlePushNotification = (alert) => {
    // Placeholder for push notification logic
    console.log(`Push notification sent for signal: ${alert.action} ${alert.symbol}`);
    // In a real implementation, this would integrate with a backend service to send push notifications
  };

  return (
    <aside className="alerts-feed card">
      <h3>Alerts & Notifications</h3>
      {selectedSignal && (
        <div className="signal-rationale-modal">
          <div className="signal-rationale-content">
            <h4>Why This Signal?</h4>
            <p>{selectedSignal.rationale || 'This signal was generated based on a combination of technical indicators and market sentiment.'}</p>
            <button onClick={() => setSelectedSignal(null)} className="close-btn">
              Close
            </button>
          </div>
        </div>
      )}
      <AnimatePresence>
        {visibleAlerts.map((alert) => {
          const strength = getSignalStrength(alert.confidence * 100, alert.historicalWinRate || 80);
          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="alert-item"
            >
              <div className="alert-content">
                <div className="alert-header">
                  <span className="signal-strength">{strength.emoji} {strength.strength}</span>
                  <span>{alert.action} {alert.symbol} - Confidence: {(alert.confidence * 100).toFixed(2)}%</span>
                </div>
                <div className="alert-details">
                  <p>
                    Entry: ${alert.entryPrice?.toFixed(2)}
                    <span className="tooltip">
                      ℹ️
                      <span className="tooltip-text">The suggested price to enter the trade.</span>
                    </span>
                  </p>
                  <p>
                    Target: ${alert.targetPrice?.toFixed(2)} ({((alert.targetPrice / alert.entryPrice - 1) * 100).toFixed(2)}%)
                    <span className="tooltip">
                      ℹ️
                      <span className="tooltip-text">The price to exit for profit.</span>
                    </span>
                  </p>
                  <p>
                    Stop Loss: ${alert.stopLoss?.toFixed(2)} ({((alert.stopLoss / alert.entryPrice - 1) * 100).toFixed(2)}%)
                    <span className="tooltip">
                      ℹ️
                      <span className="tooltip-text">The price to exit to limit losses.</span>
                    </span>
                  </p>
                  <p>
                    Risk-Reward: {alert.riskReward || '1:1'}
                  </p>
                  <p>
                    Time Horizon: {alert.timeHorizon || '30 minutes'}
                  </p>
                  <p>
                    Indicators: {alert.supportingIndicators || 'Price above VWAP, RSI at 65'}
                  </p>
                  <p>
                    Confidence: {(alert.confidence * 100).toFixed(2)}% (Win Rate: {alert.historicalWinRate || 80}%)
                  </p>
                  <button onClick={(e) => { e.stopPropagation(); setSelectedSignal(alert); }} className="rationale-btn" aria-label="View signal rationale">
                    Why This Signal?
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); handleDismiss(alert.id); }} className="dismiss-btn" aria-label={`Dismiss alert for ${alert.symbol}`}>
                    ✕
                  </button>
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </aside>
  );
}

AlertsFeed.propTypes = {
  alerts: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    type: PropTypes.string,
    symbol: PropTypes.string,
    action: PropTypes.string,
    message: PropTypes.string,
    time: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    severity: PropTypes.string,
    confidence: PropTypes.number.isRequired,
    historicalWinRate: PropTypes.number,
    entryPrice: PropTypes.number,
    targetPrice: PropTypes.number,
    stopLoss: PropTypes.number,
    riskReward: PropTypes.string,
    timeHorizon: PropTypes.string,
    supportingIndicators: PropTypes.string,
    rationale: PropTypes.string,
    isLive: PropTypes.bool,
  })).isRequired,
  onAddToPortfolio: PropTypes.func.isRequired,
};
