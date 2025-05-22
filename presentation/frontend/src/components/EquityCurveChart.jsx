import React from 'react';
import PropTypes from 'prop-types';
import { Line } from 'react-chartjs-2';

function computeEquityCurve(trades) {
  let curve = [];
  let equity = 0;
  trades.forEach((trade, idx) => {
    equity += trade.pnl || 0;
    curve.push({ x: trade.timestamp || idx, y: equity });
  });
  return curve;
}

export default function EquityCurveChart({ trades }) {
  const data = {
    labels: trades.map((t, i) => new Date(t.timestamp).toLocaleDateString()),
    datasets: [
      {
        label: 'Equity Curve',
        data: computeEquityCurve(trades).map(pt => pt.y),
        fill: true,
        borderColor: '#FFD700',
        backgroundColor: 'rgba(255,215,0,0.1)',
        tension: 0.2,
        pointRadius: 0,
      }
    ]
  };
  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false }
    },
    scales: {
      x: { display: false },
      y: { ticks: { color: '#FFD700' }, grid: { color: '#444' } }
    }
  };
  return (
    <div style={{ background: '#232323', borderRadius: 10, padding: 16, margin: '12px 0' }}>
      <div style={{ color: '#FFD700', fontWeight: 600, marginBottom: 6 }}>Equity Curve</div>
      <Line data={data} options={options} height={60} />
    </div>
  );
}

EquityCurveChart.propTypes = {
  trades: PropTypes.array.isRequired
};
