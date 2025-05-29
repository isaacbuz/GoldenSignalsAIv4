
import React, { useEffect, useRef } from 'react';
import { AISignal } from '../hooks/useSignalWebSocket';

// Lightweight Charts (https://github.com/tradingview/lightweight-charts)
import { createChart } from 'lightweight-charts';

interface SignalChartPanelProps {
  symbol: string;
  signals: AISignal[];
}

export default function SignalChartPanel({ symbol, signals }: SignalChartPanelProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }
    const chart = createChart(chartContainerRef.current, { width: 600, height: 320 });
    chartRef.current = chart;
    const lineSeries = chart.addLineSeries();
    // Fetch historical price data for the symbol
    fetch(`/api/price-history?symbol=${symbol}`)
      .then(res => res.json())
      .then(data => {
        lineSeries.setData(data.prices); // [{ time, value }]
        // Overlay signals
        signals.forEach(sig => {
          lineSeries.createPriceLine({
            price: sig.action === 'BUY' ? data.prices.find((p: any) => p.time === sig.time)?.value : undefined,
            color: sig.action === 'BUY' ? 'green' : 'red',
            lineWidth: 2,
            lineStyle: 2, // 2 = Dotted in lightweight-charts
            title: `${sig.action} (${sig.confidence}%)`
          });
        });
      });
    return () => {
      chart.remove();
    };
  }, [symbol, signals]);

  return (
    <div ref={chartContainerRef} className="rounded shadow bg-white dark:bg-zinc-900" />
  );
}
