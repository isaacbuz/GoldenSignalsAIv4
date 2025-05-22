import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './SignalFilters.css';

export default function SignalFilters({ onFilterChange }) {
  const [filters, setFilters] = useState({
    minConfidence: 80,
    minRiskReward: '1:1',
    timeHorizon: 'any',
    optionType: 'both',
    excludeChoppy: false,
    prePostMarket: false,
    minVolume: 1000000,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newFilters = {
      ...filters,
      [name]: type === 'checkbox' ? checked : value
    };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  return (
    <div className="signal-filters card">
      <h3>Filter Signals</h3>
      <div className="filter-group">
        <label htmlFor="minConfidence">Minimum Confidence (%):</label>
        <input
          type="number"
          id="minConfidence"
          name="minConfidence"
          value={filters.minConfidence}
          onChange={handleChange}
          min="0"
          max="100"
        />
      </div>
      <div className="filter-group">
        <label htmlFor="minRiskReward">Minimum Risk-Reward Ratio:</label>
        <select id="minRiskReward" name="minRiskReward" value={filters.minRiskReward} onChange={handleChange}>
          <option value="1:1">1:1</option>
          <option value="1:2">1:2</option>
          <option value="1:3">1:3</option>
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="timeHorizon">Time Horizon:</label>
        <select id="timeHorizon" name="timeHorizon" value={filters.timeHorizon} onChange={handleChange}>
          <option value="any">Any</option>
          <option value="scalping">Scalping (&lt;1 hour)</option>
          <option value="swing">Swing (1-3 days)</option>
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="optionType">Option Type:</label>
        <select id="optionType" name="optionType" value={filters.optionType} onChange={handleChange}>
          <option value="both">Both</option>
          <option value="calls">Calls Only</option>
          <option value="puts">Puts Only</option>
        </select>
      </div>
      <div className="filter-group">
        <label>
          <input
            type="checkbox"
            name="excludeChoppy"
            checked={filters.excludeChoppy}
            onChange={handleChange}
          />
          Exclude Choppy Market (Lunch Hours)
        </label>
      </div>
      <div className="filter-group">
        <label>
          <input
            type="checkbox"
            name="prePostMarket"
            checked={filters.prePostMarket}
            onChange={handleChange}
          />
          Include Pre/Post-Market Signals
        </label>
      </div>
      <div className="filter-group">
        <label htmlFor="minVolume">Min Avg Daily Volume (shares):</label>
        <input
          type="number"
          id="minVolume"
          name="minVolume"
          value={filters.minVolume}
          onChange={handleChange}
          min="0"
        />
      </div>
    </div>
  );
}

SignalFilters.propTypes = {
  onFilterChange: PropTypes.func.isRequired,
};
