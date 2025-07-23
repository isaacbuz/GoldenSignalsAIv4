/**
 * MACD Chart Component
 * Displays MACD indicator with histogram below the main chart
 */

import React, { useEffect, useRef } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineData,
  HistogramData,
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

interface MACDData {
  macd: number;
  signal: number;
  hist: number;
}

interface MACDChartProps {
  values: MACDData[];
  timestamps?: number[];
}

export const MACDChart: React.FC<MACDChartProps> = ({ values, timestamps }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const macdSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const signalSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const histogramSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

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

    // Add histogram series first (so it's behind the lines)
    const histogramSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'price',
      },
    });

    // Add MACD line
    const macdSeries = chart.addLineSeries({
      color: '#3f51b5',
      lineWidth: 2,
      crosshairMarkerVisible: false,
    });

    // Add signal line
    const signalSeries = chart.addLineSeries({
      color: '#f44336',
      lineWidth: 2,
      crosshairMarkerVisible: false,
    });

    // Generate time data
    const currentTime = Math.floor(Date.now() / 1000);
    const timeData = timestamps || values.map((_, i) => currentTime - (values.length - i - 1) * 300);

    // Set MACD data
    const macdData: LineData[] = values.map((value, i) => ({
      time: timeData[i] as Time,
      value: value.macd,
    }));
    macdSeries.setData(macdData);

    // Set signal data
    const signalData: LineData[] = values.map((value, i) => ({
      time: timeData[i] as Time,
      value: value.signal,
    }));
    signalSeries.setData(signalData);

    // Set histogram data
    const histogramData: HistogramData[] = values.map((value, i) => ({
      time: timeData[i] as Time,
      value: value.hist,
      color: value.hist >= 0 ? '#00D964' : '#FF3B30',
    }));
    histogramSeries.setData(histogramData);

    // Add zero line
    const zeroLine = chart.addLineSeries({
      color: 'rgba(255, 255, 255, 0.3)',
      lineWidth: 1,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });
    zeroLine.setData(timeData.map(time => ({ time: time as Time, value: 0 })));

    // Add watermark
    chart.applyOptions({
      watermark: {
        visible: true,
        fontSize: 24,
        horzAlign: 'left',
        vertAlign: 'top',
        color: 'rgba(255, 255, 255, 0.2)',
        text: 'MACD(12,26,9)',
      },
    });

    // Store references
    chartRef.current = chart;
    macdSeriesRef.current = macdSeries;
    signalSeriesRef.current = signalSeries;
    histogramSeriesRef.current = histogramSeries;

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
