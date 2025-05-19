import React from 'react';
import PropTypes from 'prop-types';
import './ChartOverlayControls.css';

/**
 * ChartOverlayControls provides toggles for technical indicators on the chart.
 * @param {object} overlays - { ma: boolean, ema: boolean, rsi: boolean, macd: boolean }
 * @param {function} onChange - (overlays) => void
 */
export default function ChartOverlayControls({ overlays, onChange }) {
  return (
    <div className="chart-overlay-controls">
      <label title="Moving Average: Smooths price data for trend analysis">
        <input type="checkbox" checked={overlays.ma} onChange={e => onChange({ ...overlays, ma: e.target.checked })} /> MA
      </label>
      <label title="Exponential Moving Average: Gives more weight to recent prices">
        <input type="checkbox" checked={overlays.ema} onChange={e => onChange({ ...overlays, ema: e.target.checked })} /> EMA
      </label>
      <label title="Relative Strength Index: Measures momentum and overbought/oversold">
        <input type="checkbox" checked={overlays.rsi} onChange={e => onChange({ ...overlays, rsi: e.target.checked })} /> RSI
      </label>
      <label title="MACD: Moving Average Convergence Divergence, trend and momentum">
        <input type="checkbox" checked={overlays.macd} onChange={e => onChange({ ...overlays, macd: e.target.checked })} /> MACD
      </label>
    </div>
  );
}

ChartOverlayControls.propTypes = {
  overlays: PropTypes.shape({
    ma: PropTypes.bool,
    ema: PropTypes.bool,
    rsi: PropTypes.bool,
    macd: PropTypes.bool,
  }).isRequired,
  onChange: PropTypes.func.isRequired,
};
