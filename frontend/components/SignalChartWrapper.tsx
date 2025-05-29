import React, { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";
import type { PricePoint, ForecastTrendPoint } from "./signalTypes";

interface ChartPanelProps {
  prices: PricePoint[];
  entryPoint: { time: number | string };
  exitPoint: { time: number | string };
  forecastTrend?: ForecastTrendPoint[];
}

export default function ChartPanel({ prices, entryPoint, exitPoint, forecastTrend = [] }: ChartPanelProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current || !prices || prices.length === 0) return;

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 300,
      layout: {
        background: { color: "#12141b" },
        textColor: "#c9d1d9"
      },
      grid: {
        vertLines: { color: "#2a2e39" },
        horzLines: { color: "#2a2e39" }
      },
      rightPriceScale: { borderColor: "#444" },
      timeScale: { borderColor: "#444" }
    });

    const priceSeries = chart.addLineSeries({ color: "#00d97e", lineWidth: 2 });
    priceSeries.setData(prices);

    priceSeries.setMarkers([
      {
        time: entryPoint.time,
        position: "belowBar",
        color: "#00ff00",
        shape: "arrowUp",
        text: "Entry"
      },
      {
        time: exitPoint.time,
        position: "aboveBar",
        color: "#ff3b3b",
        shape: "arrowDown",
        text: "Exit"
      }
    ]);

    if (forecastTrend.length > 0) {
      const forecastSeries = chart.addLineSeries({
        color: "#fbbf24", // gold/yellow
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false
      });
      forecastSeries.setData(forecastTrend);
    }

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [prices, entryPoint, exitPoint, forecastTrend]);

  return (
    <div className="bg-bgPanel p-4 rounded-lg border border-borderSoft font-sans" aria-label="Signal chart with forecast">
      {/*
        SignalChartWrapper displays a trading chart with entry/exit markers and optional forecast overlay.
        Uses lightweight-charts for visualization. Accessible and responsive.
      */}
      <h3 className="text-lg font-semibold text-white mb-2 font-sans" aria-label="Signal Chart and Forecast"> Signal Chart + Forecast</h3>
      <div ref={chartRef} className="h-[300px]" aria-label="Trading chart" />
    </div>
  );
}
