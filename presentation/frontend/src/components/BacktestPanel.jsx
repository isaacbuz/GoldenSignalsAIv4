import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './BacktestPanel.css';

/**
 * BacktestPanel allows users to run strategy backtests and tune parameters.
 * @param {function} onRunBacktest - (params) => void
 * @param {object} initialParams - default strategy parameters
 * @param {object} result - backtest result summary (optional)
 */
export default function BacktestPanel({ onRunBacktest, initialParams, result }) {
  const [params, setParams] = useState(initialParams);
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setParams(p => ({ ...p, [name]: type === 'number' ? +value : value }));
  };
  return (
    <div className="backtest-panel">
      <h3>Strategy Backtest</h3>
      <form onSubmit={e => { e.preventDefault(); onRunBacktest(params); }}>
        <div className="backtest-fields">
          <label>
            Lookback (days)
            <input type="number" name="lookback" min={10} max={365} value={params.lookback} onChange={handleChange} />
          </label>
          <label>
            Threshold
            <input type="number" name="threshold" min={0} max={1} step={0.01} value={params.threshold} onChange={handleChange} />
          </label>
          <label>
            Stop Loss (%)
            <input type="number" name="stop" min={0} max={50} step={0.1} value={params.stop} onChange={handleChange} />
          </label>
          <label>
            Take Profit (%)
            <input type="number" name="tp" min={0} max={50} step={0.1} value={params.tp} onChange={handleChange} />
          </label>
        </div>
        <button className="backtest-run-btn" type="submit">Run Backtest</button>
      </form>
      {result && (
        <div className="backtest-result">
          <b>Result:</b> {result.summary}
        </div>
      )}
    </div>
  );
}

BacktestPanel.propTypes = {
  onRunBacktest: PropTypes.func.isRequired,
  initialParams: PropTypes.object.isRequired,
  result: PropTypes.object,
};
