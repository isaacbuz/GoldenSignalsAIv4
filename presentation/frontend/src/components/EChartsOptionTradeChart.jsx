import React from 'react';
import PropTypes from 'prop-types';
import ReactECharts from 'echarts-for-react';

/**
 * ohlcv: array of [date, open, high, low, close]
 * signals: array of { time: date, price: number, type: 'entry'|'exit'|'stop'|'tp' }
 * loading: boolean
 */
export default function EChartsOptionTradeChart({ ohlcv, signals, loading, overlays = {} }) {
  if (loading) {
    return <div>Loading...</div>;
  }
  if (!ohlcv || ohlcv.length === 0) {
    return <div>No data available.</div>;
  }

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
      {/* Main Chart Only - no legend or OHLCV table */}
      <ReactECharts option={option} style={{ height: 380, width: '100%' }} />
    </div>
  );
}
EChartsOptionTradeChart.propTypes = {
  ohlcv: PropTypes.array,
  signals: PropTypes.array,
  loading: PropTypes.bool,
  overlays: PropTypes.object
};
