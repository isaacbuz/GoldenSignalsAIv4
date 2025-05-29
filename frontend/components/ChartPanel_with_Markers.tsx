import React, { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";
import type { PricePoint, ForecastTrendPoint } from "./signalTypes";

/**
 * ChartPanel renders a trading chart with entry/exit markers and optional forecast overlay.
 * @param prices - Array of price points for the main chart.
 * @param entryPoint - Object with entry time for marker.
 * @param exitPoint - Object with exit time for marker.
 * @param forecastTrend - Optional array of forecast trend points for overlay.
 */
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
    <div className="bg-bgPanel p-4 md:p-6 rounded-xl border border-borderSoft font-sans" aria-label="Trading chart with entry and exit markers">
      <h3 className="text-lg md:text-xl font-bold text-accentBlue mb-2 font-sans">📈 Signal Chart + Forecast</h3>
      <div ref={chartRef} className="h-[300px] w-full" />
    </div>
  );
}
