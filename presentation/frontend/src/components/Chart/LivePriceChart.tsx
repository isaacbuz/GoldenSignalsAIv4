import React, { useEffect, useRef } from "react";
import { createChart, CrosshairMode, IChartApi } from "lightweight-charts";
import { useWebSocket } from "../../context/WebSocketContext";

interface Props {
  symbol: string;
  timeframe: string; // e.g., "1m", "5m"
}

const LivePriceChart: React.FC<Props> = ({ symbol, timeframe }) => {
  const { ticks, signals } = useWebSocket();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    // Initialize chart
    chartInstance.current = createChart(chartRef.current, {
      layout: {
        background: { color: "#111" },
        textColor: "#ccc",
      },
      grid: {
        vertLines: { color: "#222" },
        horzLines: { color: "#222" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      width: chartRef.current.clientWidth,
      height: 400,
    });

    const lineSeries = chartInstance.current.addLineSeries({
      color: "#4ade80",
      lineWidth: 2,
    });

    const symbolTicks = ticks[symbol] || [];
    lineSeries.setData(
      symbolTicks.map((tick) => ({
        time: Math.floor(tick.timestamp / 1000),
        value: tick.price,
      }))
    );

    const signalData = signals[symbol] || [];
    signalData.forEach((sig) => {
      chartInstance.current?.addAnnotation?.({
        time: Math.floor(sig.timestamp / 1000),
        position: "aboveBar",
        color: sig.type === "buy" ? "#10b981" : "#ef4444",
        shape: sig.type === "buy" ? "arrowUp" : "arrowDown",
        text: `${sig.type.toUpperCase()} (${(sig.confidence * 100).toFixed(0)}%)`,
      });
    });

    // Resize listener
    const handleResize = () => {
      chartInstance.current?.resize(chartRef.current!.clientWidth, 400);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      chartInstance.current?.remove();
      window.removeEventListener("resize", handleResize);
    };
  }, [symbol, ticks, signals]);

  return (
    <div className="rounded-xl overflow-hidden border border-gray-700 shadow" ref={chartRef} />
  );
};

export default LivePriceChart;
