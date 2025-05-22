import React, { useRef, useEffect, useState } from 'react';
import { createChart } from 'lightweight-charts';
import SignalMarkerTooltip from './SignalMarkerTooltip';
import DrillInPanel from './DrillInPanel';

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d'];

// Example: Replace with real API/WebSocket data
const mockCandles = {
  '1m': [
    { time: '2023-05-13', open: 150, high: 154, low: 148, close: 152 },
    { time: '2023-05-14', open: 152, high: 158, low: 151, close: 157 },
    { time: '2023-05-15', open: 157, high: 159, low: 155, close: 158 },
    { time: '2023-05-16', open: 158, high: 162, low: 157, close: 161 },
    { time: '2023-05-17', open: 161, high: 163, low: 158, close: 160 },
    { time: '2023-05-18', open: 160, high: 162, low: 157, close: 158 },
    { time: '2023-05-19', open: 158, high: 160, low: 155, close: 156 },
  ],
  '5m': [ /* ... */ ],
  '15m': [ /* ... */ ],
  '1h': [ /* ... */ ],
  '1d': [ /* ... */ ],
};
const mockSignals = [
  { time: '2023-05-14', price: 153, type: 'entry', rsi: 28, macd: 0.5, confidence: 0.91, timestamp: Date.now() - 1000000, news: ['AAPL beats earnings expectations'] },
  { time: '2023-05-17', price: 161, type: 'exit', rsi: 71, macd: -0.3, confidence: 0.78, timestamp: Date.now() - 500000, news: ['Fed announces rate hike'] },
];

export default function OptionTradeChart() {
  const chartContainerRef = useRef();
  const [error, setError] = useState(null);
  const [timeframe, setTimeframe] = useState('1m');
  const [candles, setCandles] = useState(mockCandles['1m']);
  const [signals, setSignals] = useState(mockSignals);
  const [hoveredSignal, setHoveredSignal] = useState(null);
  const [drillSignal, setDrillSignal] = useState(null);
  const [livePrice, setLivePrice] = useState(null);
  const [markerPositions, setMarkerPositions] = useState([]);

  // Simulate real-time price ticker
  useEffect(() => {
    const interval = setInterval(() => {
      setLivePrice(prev => prev ? prev + (Math.random() - 0.5) : 160);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    setCandles(mockCandles[timeframe] || mockCandles['1m']);
    setSignals(mockSignals);
  }, [timeframe]);

  // Chart rendering and interactive markers
  useEffect(() => {
    if (!chartContainerRef.current) return;
    chartContainerRef.current.innerHTML = '';
    let chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.offsetWidth,
      height: 340,
      layout: {
        background: { type: 'solid', color: '#232323' },
        textColor: '#FFD700',
      },
      grid: {
        vertLines: { color: '#333' },
        horzLines: { color: '#333' },
      },
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: '#FFD700',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: { borderColor: '#FFD700' },
    });
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#FFD700',
      downColor: '#C9A100',
      borderUpColor: '#FFD700',
      borderDownColor: '#C9A100',
      wickUpColor: '#FFD700',
      wickDownColor: '#C9A100',
    });
    candleSeries.setData(candles);

    // Interactive signal markers (approximate x/y placement)
    const positions = signals.map((signal, idx) => {
      // Find the x index of the candle for this signal
      const candleIdx = candles.findIndex(c => c.time === signal.time);
      if (candleIdx === -1) return null;
      return {
        left: 60 + candleIdx * 60,
        top: 120 + (signal.type === 'entry' ? -20 : 20),
        signal,
      };
    }).filter(Boolean);
    setMarkerPositions(positions);

    // Responsive resize
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current.offsetWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chart) chart.remove();
    };
  }, [candles, signals]);

  if (error) {
    return (
      <div style={{ color: '#FF5555', background: '#232323', border: '1.5px solid #FF5555', borderRadius: 12, padding: 24, textAlign: 'center', margin: '2rem auto' }}>
        <b>Chart Error</b><br />
        {error}
      </div>
    );
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: 380 }}>
      {/* Timeframe Toggle */}
      <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 2 }}>
        {TIMEFRAMES.map(tf => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            style={{
              marginRight: 6,
              padding: '2px 10px',
              borderRadius: 8,
              background: timeframe === tf ? '#FFD700' : '#333',
              color: timeframe === tf ? '#232323' : '#FFD700',
              border: 'none',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {tf}
          </button>
        ))}
      </div>
      {/* Live Price Ticker */}
      <div style={{ position: 'absolute', top: 8, right: 16, zIndex: 2, fontWeight: 700, color: '#FFD700', fontSize: 18 }}>
        {livePrice !== null && <span>Live: ${livePrice.toFixed(2)}</span>}
      </div>
      {/* Chart Canvas */}
      <div
        ref={chartContainerRef}
        style={{ width: '100%', height: 340, borderRadius: 16, boxShadow: '0 2px 12px #FFD70033', background: '#232323', margin: '0 auto' }}
      />
      {/* Signal Markers */}
      {markerPositions.map((pos, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: pos.left,
            top: pos.top,
            width: 20,
            height: 20,
            background: pos.signal.type === 'entry' ? '#00FF99' : '#FF5555',
            borderRadius: '50%',
            boxShadow: '0 0 6px #FFD700AA',
            border: '2px solid #FFD700',
            cursor: 'pointer',
            zIndex: 5,
          }}
          onMouseEnter={() => setHoveredSignal(pos.signal)}
          onMouseLeave={() => setHoveredSignal(null)}
          onClick={() => setDrillSignal(pos.signal)}
        />
      ))}
      {/* Hover Tooltip */}
      {hoveredSignal && (
        <div style={{ position: 'absolute', left: 120, top: 60, zIndex: 10 }}>
          <SignalMarkerTooltip signal={hoveredSignal} />
        </div>
      )}
      {/* Drill-in Panel */}
      <DrillInPanel signal={drillSignal} onClose={() => setDrillSignal(null)} />
    </div>
  );
}
