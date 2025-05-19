import React from 'react';
import PropTypes from 'prop-types';
import ReactECharts from 'echarts-for-react';
import ChartSkeleton from './ChartSkeleton';

/**
 * ohlcv: array of [date, open, high, low, close]
 * signals: array of { time: date, price: number, type: 'entry'|'exit'|'stop'|'tp' }
 * loading: boolean
 */
export default function EChartsOptionTradeChart({ ohlcv, signals, loading, overlays = {} }) {
  if (loading) {
    return <ChartSkeleton />;
  }
  if (!ohlcv || ohlcv.length === 0) {
    return <div style={{ color: '#FFD700', textAlign: 'center', padding: '2rem' }}>No data available.</div>;
  }

  // Legend for OHLCV format
  const legendStyle = {
    background: '#232323',
    color: '#FFD700',
    borderRadius: '8px',
    padding: '0.6rem 1rem',
    margin: '0 auto 1rem auto',
    maxWidth: 700,
    fontSize: '1rem',
    boxShadow: '0 2px 8px #FFD70022',
    fontWeight: 500
  };
  const legendText = `OHLCV Format: [Timestamp, Open, High, Low, Close, Volume]`;

  // Pagination state
  const [page, setPage] = React.useState(0);
  const rowsPerPage = 6;
  const totalPages = Math.ceil(ohlcv.length / rowsPerPage);
  const pagedOhlcv = ohlcv.slice(page * rowsPerPage, (page + 1) * rowsPerPage);

  // CSV Export
  function exportCSV() {
    const header = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'];
    const rows = pagedOhlcv.map(row => [dateFormat(row[0]), row[1], row[2], row[3], row[4], row[5]]);
    const csv = [header, ...rows].map(e => e.join(",")).join("\n");
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ohlcv_data.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  // Table for OHLCV data with tooltips
  const tableStyle = {
    width: '100%',
    maxWidth: 700,
    margin: '1rem auto 0 auto',
    background: '#191919',
    color: '#FFD700',
    borderRadius: 12,
    boxShadow: '0 2px 8px #FFD70022',
    overflow: 'hidden',
    fontSize: '0.98rem',
    borderCollapse: 'collapse',
  };
  const thStyle = {
    background: '#232323',
    color: '#FFD700',
    fontWeight: 700,
    padding: '0.5rem 0.75rem',
    borderBottom: '2px solid #FFD700',
    textAlign: 'center',
    position: 'relative',
    cursor: 'help',
    whiteSpace: 'nowrap',
  };
  const tdStyle = {
    padding: '0.4rem 0.75rem',
    borderBottom: '1px solid #333',
    textAlign: 'center',
    fontWeight: 400,
  };
  const tooltipStyle = {
    position: 'absolute',
    bottom: '110%',
    left: '50%',
    transform: 'translateX(-50%)',
    background: '#232323',
    color: '#FFD700',
    padding: '0.4rem 0.8rem',
    borderRadius: 8,
    fontSize: '0.92rem',
    boxShadow: '0 2px 8px #FFD70022',
    zIndex: 10,
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
    opacity: 0.95,
  };
  const [tooltip, setTooltip] = React.useState({ col: null, show: false });
  const tooltips = {
    Open: 'The price at market open for the day',
    High: 'The highest price reached during the day',
    Low: 'The lowest price reached during the day',
    Close: 'The price at market close for the day',
    Volume: 'The total number of shares/contracts traded',
  };
  const dateFormat = ts => {
    const d = new Date(ts);
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  };

  const ohlcvTable = (
    <div style={{ position: 'relative' }}>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={thStyle}>Date</th>
            <th style={thStyle} onMouseEnter={() => setTooltip({ col: 'Open', show: true })} onMouseLeave={() => setTooltip({ col: null, show: false })}>
              Open
              {tooltip.show && tooltip.col === 'Open' && (
                <div style={tooltipStyle}>{tooltips.Open}</div>
              )}
            </th>
            <th style={thStyle} onMouseEnter={() => setTooltip({ col: 'High', show: true })} onMouseLeave={() => setTooltip({ col: null, show: false })}>
              High
              {tooltip.show && tooltip.col === 'High' && (
                <div style={tooltipStyle}>{tooltips.High}</div>
              )}
            </th>
            <th style={thStyle} onMouseEnter={() => setTooltip({ col: 'Low', show: true })} onMouseLeave={() => setTooltip({ col: null, show: false })}>
              Low
              {tooltip.show && tooltip.col === 'Low' && (
                <div style={tooltipStyle}>{tooltips.Low}</div>
              )}
            </th>
            <th style={thStyle} onMouseEnter={() => setTooltip({ col: 'Close', show: true })} onMouseLeave={() => setTooltip({ col: null, show: false })}>
              Close
              {tooltip.show && tooltip.col === 'Close' && (
                <div style={tooltipStyle}>{tooltips.Close}</div>
              )}
            </th>
            <th style={thStyle} onMouseEnter={() => setTooltip({ col: 'Volume', show: true })} onMouseLeave={() => setTooltip({ col: null, show: false })}>
              Volume
              {tooltip.show && tooltip.col === 'Volume' && (
                <div style={tooltipStyle}>{tooltips.Volume}</div>
              )}
            </th>
          </tr>
        </thead>
        <tbody>
          {pagedOhlcv.map((row, i) => (
            <tr key={i}>
              <td style={tdStyle}>{dateFormat(row[0])}</td>
              <td style={tdStyle}>{row[1]}</td>
              <td style={tdStyle}>{row[2]}</td>
              <td style={tdStyle}>{row[3]}</td>
              <td style={tdStyle}>{row[4]}</td>
              <td style={tdStyle}>{row[5]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {/* Pagination controls */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', margin: '0.5rem 0 0.5rem 0' }}>
          <button
            style={{
              background: '#232323', color: '#FFD700', border: 'none', borderRadius: 6, padding: '0.4rem 1.1rem', marginRight: 8, fontWeight: 600, cursor: page > 0 ? 'pointer' : 'not-allowed', opacity: page > 0 ? 1 : 0.5, boxShadow: '0 2px 6px #FFD70022' }}
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
          >Previous</button>
          <span style={{ color: '#FFD700', fontWeight: 500, fontSize: '1rem', margin: '0 8px' }}>Page {page + 1} of {totalPages}</span>
          <button
            style={{
              background: '#232323', color: '#FFD700', border: 'none', borderRadius: 6, padding: '0.4rem 1.1rem', marginLeft: 8, fontWeight: 600, cursor: page < totalPages - 1 ? 'pointer' : 'not-allowed', opacity: page < totalPages - 1 ? 1 : 0.5, boxShadow: '0 2px 6px #FFD70022' }}
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page === totalPages - 1}
          >Next</button>
        </div>
      )}
      {/* Export CSV button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
        <button
          style={{ background: '#FFD700', color: '#232323', border: 'none', borderRadius: 6, padding: '0.5rem 1.2rem', fontWeight: 700, cursor: 'pointer', boxShadow: '0 2px 6px #FFD70022', fontSize: '1rem' }}
          onClick={exportCSV}
        >Export CSV</button>
      </div>
    </div>
  );

  const dates = ohlcv.map(item => item[0]);
  // Helper: Simple moving average (window=3 for demo)
  function calcMA(data, window = 3) {
    if (!data || data.length < window) return [];
    return data.map((d, i, arr) => {
      if (i < window - 1) return null;
      const sum = arr.slice(i - window + 1, i + 1).reduce((acc, v) => acc + v[4], 0);
      return sum / window;
    });
  }
  // Helper: Exponential moving average (window=3 for demo)
  function calcEMA(data, window = 3) {
    if (!data || data.length < window) return [];
    let k = 2 / (window + 1);
    let emaArr = [];
    let prev = data[0][4];
    for (let i = 0; i < data.length; i++) {
      let price = data[i][4];
      let ema = i === 0 ? price : price * k + prev * (1 - k);
      emaArr.push(ema);
      prev = ema;
    }
    return emaArr;
  }
  // Helper: RSI (window=5 for demo)
  function calcRSI(data, window = 5) {
    if (!data || data.length < window) return [];
    let rsiArr = [];
    for (let i = 0; i < data.length; i++) {
      if (i < window) { rsiArr.push(null); continue; }
      let gains = 0, losses = 0;
      for (let j = i - window + 1; j <= i; j++) {
        let diff = data[j][4] - data[j - 1][4];
        if (diff > 0) gains += diff;
        else losses -= diff;
      }
      let rs = gains / (losses || 1);
      rsiArr.push(100 - 100 / (1 + rs));
    }
    return rsiArr;
  }
  // Helper: MACD (12,26,9 for demo)
  function calcMACD(data) {
    if (!data || data.length < 26) return { macd: [], signal: [] };
    let ema12 = calcEMA(data, 12);
    let ema26 = calcEMA(data, 26);
    let macd = ema12.map((v, i) => v - (ema26[i] || 0));
    let signal = calcEMA(macd.map(v => [0,0,0,0,v]), 9); // fudge for demo
    return { macd, signal };
  }

  const entryMarkers = (signals||[]).filter(s => s.type === 'entry').map(s => ({
    name: 'Entry',
    coord: [s.time, s.price],
    value: 'Entry',
    itemStyle: { color: '#00FF99' },
    symbol: 'arrow',
    symbolSize: 18,
    label: { formatter: 'Entry', position: 'top', color: '#00FF99', fontWeight: 'bold' }
  }));
  const exitMarkers = (signals||[]).filter(s => s.type === 'exit').map(s => ({
    name: 'Exit',
    coord: [s.time, s.price],
    value: 'Exit',
    itemStyle: { color: '#FF5555' },
    symbol: 'arrow',
    symbolRotate: 180,
    symbolSize: 18,
    label: { formatter: 'Exit', position: 'bottom', color: '#FF5555', fontWeight: 'bold' }
  }));
  // Add more marker types as needed (stop-loss, take-profit)
  // Overlays
  const ma = overlays.ma ? calcMA(ohlcv) : [];
  const ema = overlays.ema ? calcEMA(ohlcv) : [];
  const rsi = overlays.rsi ? calcRSI(ohlcv) : [];
  const macdObj = overlays.macd ? calcMACD(ohlcv) : { macd: [], signal: [] };

  const option = {
    backgroundColor: '#232323',
    title: {
      text: 'Option Trade Chart',
      left: 'center',
      textStyle: { color: '#FFD700', fontWeight: 'bold', fontSize: 20 }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: '#232323',
      borderColor: '#FFD700',
      textStyle: { color: '#FFD700' }
    },
    grid: {
      left: '5%', right: '5%', top: 60, bottom: 40
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#FFD700' } },
      axisLabel: { color: '#FFD700', fontWeight: 'bold' }
    },
    yAxis: {
      scale: true,
      axisLine: { lineStyle: { color: '#FFD700' } },
      splitLine: { lineStyle: { color: '#333' } },
      axisLabel: { color: '#FFD700', fontWeight: 'bold' }
    },
    series: [
      {
        type: 'candlestick',
        name: 'Price',
        data: ohlcv.map(item => item.slice(1)),
        itemStyle: {
          color: '#FFD700',
          color0: '#C9A100',
          borderColor: '#FFD700',
          borderColor0: '#C9A100',
          shadowColor: 'rgba(255,215,0,0.4)',
          shadowBlur: 8
        },
        markPoint: {
          data: [...entryMarkers, ...exitMarkers],
          symbolKeepAspect: true,
          label: { fontWeight: 'bold', fontSize: 12 }
        }
      },
      ...(overlays.ma && ma.length ? [{
        type: 'line',
        name: 'MA',
        data: ma,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#00BFFF', width: 2, type: 'solid' },
        emphasis: { focus: 'series' },
        tooltip: { valueFormatter: v => v && v.toFixed(2) }
      }] : []),
      ...(overlays.ema && ema.length ? [{
        type: 'line',
        name: 'EMA',
        data: ema,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#FF9900', width: 2, type: 'dashed' },
        emphasis: { focus: 'series' },
        tooltip: { valueFormatter: v => v && v.toFixed(2) }
      }] : []),
      // RSI and MACD are typically in subcharts, but for demo, overlay as lines
      ...(overlays.rsi && rsi.length ? [{
        type: 'line',
        name: 'RSI',
        data: rsi,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#00FF99', width: 1, type: 'dotted' },
        emphasis: { focus: 'series' },
        tooltip: { valueFormatter: v => v && v.toFixed(2) }
      }] : []),
      ...(overlays.macd && macdObj.macd.length ? [{
        type: 'line',
        name: 'MACD',
        data: macdObj.macd,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#FF5555', width: 1, type: 'solid' },
        emphasis: { focus: 'series' },
        tooltip: { valueFormatter: v => v && v.toFixed(2) }
      }, {
        type: 'line',
        name: 'MACD Signal',
        data: macdObj.signal,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#FFD700', width: 1, type: 'dashed' },
        emphasis: { focus: 'series' },
        tooltip: { valueFormatter: v => v && v.toFixed(2) }
      }] : [])
    ]
  };
  return (
    <div style={{ width: '100%', maxWidth: 700, margin: '0 auto', background: '#232323', borderRadius: 16, boxShadow: '0 2px 12px #FFD70033', padding: 16 }}>
      {/* Legend for OHLCV array format */}
      <div style={legendStyle}>{legendText}</div>
      {/* Main Chart */}
      <ReactECharts option={option} style={{ height: 380, width: '100%' }} />
      {/* Table: Human-readable OHLCV data */}
      {ohlcvTable}
    </div>
  );
}
EChartsOptionTradeChart.propTypes = {
  ohlcv: PropTypes.array,
  signals: PropTypes.array,
  loading: PropTypes.bool,
  overlays: PropTypes.object
};
