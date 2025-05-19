import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

const mockSignal = {
  action: 'BUY CALL',
  confidence: 92,
  entry: 192.45,
  exit: 196.00,
  reason: 'Momentum divergence detected with high volume and bullish options flow.',
  time: '2025-05-19 13:45',
};

const mockLog = [
  { time: '13:45', action: 'BUY CALL', price: 192.45, confidence: 92 },
  { time: '11:20', action: 'SELL PUT', price: 191.00, confidence: 86 },
  { time: '10:05', action: 'HOLD', price: 190.50, confidence: 70 },
];

const mockOHLCV = {
  labels: ['2025-05-19 10:00', '2025-05-19 11:00', '2025-05-19 12:00', '2025-05-19 13:00', '2025-05-19 14:00'],
  datasets: [
    {
      label: 'Price',
      data: [
        { x: '2025-05-19 10:00', y: 190.50 },
        { x: '2025-05-19 11:00', y: 191.00 },
        { x: '2025-05-19 12:00', y: 192.00 },
        { x: '2025-05-19 13:00', y: 192.45 },
        { x: '2025-05-19 14:00', y: 196.00 },
      ],
      borderColor: 'rgba(255, 215, 0, 1)',
      backgroundColor: 'rgba(255, 215, 0, 0.2)',
      pointBorderColor: 'rgba(255, 215, 0, 1)',
      pointBackgroundColor: 'rgba(255, 215, 0, 0.2)',
    },
  ],
};

function Dashboard() {
  const [ticker, setTicker] = useState('AAPL');
  const [signal, setSignal] = useState(mockSignal);
  const [log, setLog] = useState(mockLog);
  const [chartData, setChartData] = useState(mockOHLCV);

  useEffect(() => {
    // Fetch live signal/log from backend if available, else fallback to mock
    fetch('/api/signal')
      .then(response => response.json())
      .then(data => {
        setSignal(data.signal);
        setLog(data.log);
      })
      .catch(error => console.error(error));
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    // In real app, fetch signal for ticker here
    setSignal({ ...mockSignal, entry: Math.random() * 10 + 190 });
    setLog([
      { time: new Date().toLocaleTimeString(), action: 'BUY CALL', price: Math.random() * 10 + 190, confidence: 90 + Math.floor(Math.random() * 10) },
      ...log.slice(0, 2),
    ]);
  };

  return (
    <div className="dashboard-page" style={{ background: '#181818', minHeight: '100vh', color: '#F2E9C9', padding: '2rem' }}>
      <h1 style={{ color: '#FFD700', fontWeight: 700, fontSize: '2.5rem', marginBottom: '1rem' }}>
        <span role="img" aria-label="money">💰</span> GoldenSignalsAI Options Dashboard
      </h1>
      <form onSubmit={handleSearch} style={{ display: 'flex', alignItems: 'center', marginBottom: '2rem' }}>
        <input
          type="text"
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter ticker (e.g. AAPL)"
          style={{
            padding: '0.75rem 1rem',
            fontSize: '1.2rem',
            borderRadius: '8px 0 0 8px',
            border: 'none',
            outline: 'none',
            background: '#232323',
            color: '#FFD700',
            fontWeight: 600,
            letterSpacing: '0.1em',
            width: '180px',
          }}
        />
        <button
          type="submit"
          style={{
            padding: '0.75rem 1.5rem',
            background: 'linear-gradient(90deg, #FFD700 60%, #F2E9C9 100%)',
            color: '#181818',
            fontWeight: 700,
            fontSize: '1.2rem',
            border: 'none',
            borderRadius: '0 8px 8px 0',
            cursor: 'pointer',
            boxShadow: '0 2px 8px #FFD70044',
          }}
        >
          Search
        </button>
      </form>
      <div className="signal-section" style={{ background: '#232323', borderRadius: '16px', padding: '2rem', marginBottom: '2rem', boxShadow: '0 4px 16px #FFD70022', maxWidth: 600 }}>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: '#FFD700', marginBottom: '0.5rem' }}>{signal.action}</div>
        <div style={{ fontSize: '1.2rem', color: signal.action.includes('BUY') ? '#00FF99' : (signal.action.includes('SELL') ? '#FF5555' : '#F2E9C9'), fontWeight: 500 }}>
          Confidence: {signal.confidence}%
        </div>
        <div style={{ margin: '1rem 0', fontSize: '1.1rem' }}>
          Entry: <span style={{ color: '#FFD700', fontWeight: 600 }}>${signal.entry.toFixed(2)}</span> &nbsp;|
          Exit: <span style={{ color: '#FFD700', fontWeight: 600 }}>${signal.exit.toFixed(2)}</span>
        </div>
        <div style={{ fontStyle: 'italic', color: '#F2E9C9BB', fontSize: '1rem' }}>
          {signal.reason}
        </div>
        <div style={{ fontSize: '0.9rem', color: '#F2E9C988', marginTop: '0.5rem' }}>
          Last updated: {signal.time}
        </div>
      </div>
      <div className="chart-section" style={{ background: '#232323', borderRadius: '16px', padding: '2rem', marginBottom: '2rem', boxShadow: '0 2px 8px #FFD70022', maxWidth: 900 }}>
        <Line data={chartData} options={{
          responsive: true,
          title: {
            display: true,
            text: 'Price Chart',
          },
          scales: {
            yAxes: [{
              ticks: {
                beginAtZero: true,
              },
            }],
          },
        }} />
      </div>
      <div className="log-section" style={{ background: '#232323', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 2px 8px #FFD70022', maxWidth: 600 }}>
        <div style={{ color: '#FFD700', fontWeight: 600, marginBottom: '1rem', fontSize: '1.1rem' }}>Recent Signals</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', color: '#F2E9C9' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #FFD70044' }}>
              <th style={{ textAlign: 'left', padding: '0.5rem' }}>Time</th>
              <th style={{ textAlign: 'left', padding: '0.5rem' }}>Action</th>
              <th style={{ textAlign: 'left', padding: '0.5rem' }}>Price</th>
              <th style={{ textAlign: 'left', padding: '0.5rem' }}>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {log.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid #FFD70022' }}>
                <td style={{ padding: '0.5rem', color: '#FFD700' }}>{row.time}</td>
                <td style={{ padding: '0.5rem', color: row.action.includes('BUY') ? '#00FF99' : (row.action.includes('SELL') ? '#FF5555' : '#F2E9C9') }}>{row.action}</td>
                <td style={{ padding: '0.5rem' }}>${row.price.toFixed(2)}</td>
                <td style={{ padding: '0.5rem' }}>{row.confidence}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Dashboard;
