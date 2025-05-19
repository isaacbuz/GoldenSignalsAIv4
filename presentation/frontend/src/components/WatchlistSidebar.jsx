import React from 'react';
import PropTypes from 'prop-types';
import './WatchlistSidebar.css';

/**
 * WatchlistSidebar displays a list of tracked symbols with live price and mini-charts.
 * @param {Array} watchlist - Array of { symbol, price, change, sparkline }
 * @param {Function} onSelect - Function to call when a symbol is clicked
 */
export default function WatchlistSidebar({ watchlist, selected, onSelect }) {
  return (
    <aside className="watchlist-sidebar">
      <h3>Watchlist</h3>
      <ul>
        {watchlist.map(item => (
          <li
            key={item.symbol}
            className={selected === item.symbol ? 'selected' : ''}
            onClick={() => onSelect(item.symbol)}
          >
            <div
              className="watchlist-symbol"
              title={{
                AAPL: 'Apple Inc.',
                TSLA: 'Tesla, Inc.',
                GE: 'General Electric',
                MSFT: 'Microsoft Corp.'
              }[item.symbol] || item.symbol}
            >
              {item.symbol}
            </div>
            <div className="watchlist-price">
              <span className={item.change > 0 ? 'up' : item.change < 0 ? 'down' : ''}>
                {item.price}
              </span>
              <span className="watchlist-change">
                {item.change > 0 ? '+' : ''}{item.change}%
              </span>
            </div>
            <div className="watchlist-sparkline">
              {/* Placeholder for mini-chart/sparkline */}
              <svg width="60" height="18"><polyline points={item.sparkline} fill="none" stroke="#FFD700" strokeWidth="2" /></svg>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}

WatchlistSidebar.propTypes = {
  watchlist: PropTypes.arrayOf(PropTypes.shape({
    symbol: PropTypes.string.isRequired,
    price: PropTypes.number.isRequired,
    change: PropTypes.number.isRequired,
    sparkline: PropTypes.string.isRequired,
  })).isRequired,
  selected: PropTypes.string,
  onSelect: PropTypes.func.isRequired,
};
