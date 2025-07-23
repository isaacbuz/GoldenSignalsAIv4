/**
 * RSI Chart Component
 * Displays RSI indicator below the main chart
 */

import React, { useEffect, useRef } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineData,
  Time,
  ColorType,
} from 'lightweight-charts';
import { Box } from '@mui/material';
import { styled } from '@mui/material/styles';

const ChartContainer = styled(Box)({
  position: 'relative',
  width: '100%',
  height: '150px',
  backgroundColor: '#000000',
  borderRadius: '8px',
});

interface RSIChartProps {
  values: number[];
  timestamps?: number[];
}

export const RSIChart: React.FC<RSIChartProps> = ({ values, timestamps }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || values.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 150,
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
        visible: false,
      },
    });

    // Add RSI line
    const rsiSeries = chart.addLineSeries({
      color: '#ff9800',
      lineWidth: 2,
      crosshairMarkerVisible: false,
    });

    // Add horizontal lines for 70 and 30
    const overboughtLine = chart.addLineSeries({
      color: 'rgba(255, 255, 255, 0.3)',
      lineWidth: 1,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    const oversoldLine = chart.addLineSeries({
      color: 'rgba(255, 255, 255, 0.3)',
      lineWidth: 1,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Generate time data
    const currentTime = Math.floor(Date.now() / 1000);
    const timeData = timestamps || values.map((_, i) => currentTime - (values.length - i - 1) * 300);

    // Set RSI data
    const rsiData: LineData[] = values.map((value, i) => ({
      time: timeData[i] as Time,
      value: value,
    }));
    rsiSeries.setData(rsiData);

    // Set overbought/oversold lines
    const lineData = timeData.map(time => [
      { time: time as Time, value: 70 },
      { time: time as Time, value: 30 },
    ]);
    overboughtLine.setData(lineData.map(d => d[0]));
    oversoldLine.setData(lineData.map(d => d[1]));

    // Add watermark
    chart.applyOptions({
      watermark: {
        visible: true,
        fontSize: 24,
        horzAlign: 'left',
        vertAlign: 'top',
        color: 'rgba(255, 255, 255, 0.2)',
        text: 'RSI(14)',
      },
    });

    // Store references
    chartRef.current = chart;
    rsiSeriesRef.current = rsiSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [values, timestamps]);

  return <ChartContainer ref={chartContainerRef} />;
};
