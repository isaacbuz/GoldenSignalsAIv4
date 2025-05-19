import React, { useRef, useEffect } from 'react';
import { createChart } from 'lightweight-charts';

// Sample candlestick data (mock)
const candles = [
  { time: '2023-05-13', open: 150, high: 154, low: 148, close: 152 },
  { time: '2023-05-14', open: 152, high: 158, low: 151, close: 157 },
  { time: '2023-05-15', open: 157, high: 159, low: 155, close: 158 },
  { time: '2023-05-16', open: 158, high: 162, low: 157, close: 161 },
  { time: '2023-05-17', open: 161, high: 163, low: 158, close: 160 },
  { time: '2023-05-18', open: 160, high: 162, low: 157, close: 158 },
  { time: '2023-05-19', open: 158, high: 160, low: 155, close: 156 },
];

// Entry and exit points (mock)
const signals = [
  { time: '2023-05-14', price: 153, type: 'entry' },
  { time: '2023-05-17', price: 161, type: 'exit' },
];

export default function OptionTradeChart() {
  const chartContainerRef = useRef();
  const [error, setError] = React.useState(null);
  useEffect(() => {
    let chart;
    let handleResize;
    try {
      chart = createChart(chartContainerRef.current, {
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
      crosshair: {
        mode: 0,
      },
      timeScale: {
        borderColor: '#FFD700',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#FFD700',
      },
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

    // Overlay entry/exit markers
    signals.forEach(signal => {
      candleSeries.createPriceLine({
        price: signal.price,
        color: signal.type === 'entry' ? '#00FF99' : '#FF5555',
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: signal.type === 'entry' ? 'Entry' : 'Exit',
      });
      // NOTE: The open-source lightweight-charts does NOT support addShape (arrows, icons). For more advanced overlays, use the full Charting Library or another charting solution.

    });

    // Responsive resize
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current.offsetWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      handleResize = () => {
        chart.applyOptions({ width: chartContainerRef.current.offsetWidth });
      };
      window.addEventListener('resize', handleResize);
    } catch (err) {
      setError('Chart rendering error: ' + (err?.message || err));
    }
    return () => {
      if (handleResize) window.removeEventListener('resize', handleResize);
      if (chart) chart.remove();
    };
  }, []);

  if (error) {
    return (
      <div style={{ color: '#FF5555', background: '#232323', border: '1.5px solid #FF5555', borderRadius: 12, padding: 24, textAlign: 'center', margin: '2rem auto' }}>
        <b>Chart Error</b><br />
        {error}
      </div>
    );
  }

  return (
    <div
      ref={chartContainerRef}
      style={{ width: '100%', height: 340, borderRadius: 16, boxShadow: '0 2px 12px #FFD70033', background: '#232323', margin: '0 auto' }}
    />
  );
}

