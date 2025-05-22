import React, { useState } from 'react';
import PropTypes from 'prop-types';

export default function CorrelationMatrix({ symbols, fetchCorrelation }) {
  const [matrix, setMatrix] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleCalculate() {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchCorrelation(symbols);
      setMatrix(result);
    } catch (e) {
      setError('Failed to fetch correlation');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ background: '#232323', borderRadius: 10, padding: 16, margin: '12px 0', color: '#FFD700' }}>
      <div style={{ fontWeight: 600, marginBottom: 10 }}>Correlation Matrix</div>
      <button onClick={handleCalculate} disabled={loading} style={{ background: '#FFD700', color: '#232323', border: 'none', borderRadius: 8, padding: '4px 10px', fontWeight: 700, marginBottom: 10 }}>
        {loading ? 'Calculating...' : 'Calculate'}
      </button>
      {error && <div style={{ color: '#f43f5e' }}>{error}</div>}
      {matrix && (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 10 }}>
          <thead>
            <tr>
              <th style={{ border: '1px solid #FFD700', padding: 4 }}>Symbol</th>
              {symbols.map(s => <th key={s} style={{ border: '1px solid #FFD700', padding: 4 }}>{s}</th>)}
            </tr>
          </thead>
          <tbody>
            {symbols.map((row, i) => (
              <tr key={row}>
                <td style={{ border: '1px solid #FFD700', padding: 4 }}>{row}</td>
                {symbols.map((col, j) => (
                  <td key={col} style={{ border: '1px solid #FFD700', padding: 4, background: i === j ? '#FFD70022' : '#232323' }}>
                    {matrix[row] && matrix[row][col] !== undefined ? matrix[row][col].toFixed(2) : '--'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

CorrelationMatrix.propTypes = {
  symbols: PropTypes.array.isRequired,
  fetchCorrelation: PropTypes.func.isRequired
};
