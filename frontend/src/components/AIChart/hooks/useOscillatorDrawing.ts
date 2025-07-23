/**
 * useOscillatorDrawing Hook
 *
 * Handles the drawing of oscillator-type indicators (RSI, MACD, Stochastic, etc.)
 * in separate panels below the main chart.
 */

import { useCallback } from 'react';
import { ChartDataPoint } from '../components/ChartCanvas/types';
import {
  calculateRSI,
  calculateMACD,
  calculateStochastic,
  calculateATR,
  calculateADX
} from '../../../utils/technicalIndicators';

interface OscillatorPanelParams {
  ctx: CanvasRenderingContext2D;
  data: ChartDataPoint[];
  indicator: string;
  x: number;
  y: number;
  width: number;
  height: number;
  theme: any;
}

interface DrawOscillatorParams {
  values: number[];
  timestamps: number[];
  ctx: CanvasRenderingContext2D;
  x: number;
  y: number;
  width: number;
  height: number;
  min: number;
  max: number;
  color: string;
  lineWidth?: number;
}

export const useOscillatorDrawing = () => {
  /**
   * Draw oscillator line within panel bounds
   */
  const drawOscillatorLine = useCallback((params: DrawOscillatorParams) => {
    const { values, timestamps, ctx, x, y, width, height, min, max, color, lineWidth = 2 } = params;

    if (values.length < 2) return;

    const range = max - min;
    const xScale = width / (timestamps.length - 1);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;

    // Clip to panel bounds
    ctx.beginPath();
    ctx.rect(x, y, width, height);
    ctx.clip();

    // Draw line
    ctx.beginPath();
    let started = false;

    for (let i = 0; i < values.length; i++) {
      if (isNaN(values[i])) continue;

      const xPos = x + i * xScale;
      const yPos = y + height - ((values[i] - min) / range) * height;

      if (!started) {
        ctx.moveTo(xPos, yPos);
        started = true;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    }

    ctx.stroke();
    ctx.restore();
  }, []);

  /**
   * Draw horizontal reference lines (e.g., overbought/oversold)
   */
  const drawReferenceLine = useCallback((
    ctx: CanvasRenderingContext2D,
    value: number,
    x: number,
    y: number,
    width: number,
    height: number,
    min: number,
    max: number,
    color: string,
    label?: string
  ) => {
    const range = max - min;
    const yPos = y + height - ((value - min) / range) * height;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    ctx.beginPath();
    ctx.moveTo(x, yPos);
    ctx.lineTo(x + width, yPos);
    ctx.stroke();

    // Draw label if provided
    if (label) {
      ctx.fillStyle = color;
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x - 5, yPos);
    }

    ctx.restore();
  }, []);

  /**
   * Draw RSI indicator
   */
  const drawRSI = useCallback((params: OscillatorPanelParams) => {
    const { ctx, data, x, y, width, height, theme } = params;
    const closes = data.map(d => d.close);
    const timestamps = data.map(d => d.time);
    const rsi = calculateRSI(closes);

    // Draw background
    ctx.fillStyle = theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, 0.02)'
      : 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(x, y, width, height);

    // Draw reference lines
    drawReferenceLine(ctx, 70, x, y, width, height, 0, 100, '#FF4444', '70');
    drawReferenceLine(ctx, 50, x, y, width, height, 0, 100, '#666666', '50');
    drawReferenceLine(ctx, 30, x, y, width, height, 0, 100, '#00FF88', '30');

    // Draw RSI line
    drawOscillatorLine({
      values: rsi,
      timestamps,
      ctx,
      x,
      y,
      width,
      height,
      min: 0,
      max: 100,
      color: theme.palette.primary.main,
      lineWidth: 2
    });

    // Draw label
    ctx.fillStyle = theme.palette.text.primary;
    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('RSI (14)', x + 10, y + 5);

    // Draw current value
    const currentRSI = rsi[rsi.length - 1];
    if (!isNaN(currentRSI)) {
      ctx.fillStyle = currentRSI > 70 ? '#FF4444' : currentRSI < 30 ? '#00FF88' : theme.palette.text.primary;
      ctx.fillText(currentRSI.toFixed(2), x + width - 50, y + 5);
    }
  }, [drawOscillatorLine, drawReferenceLine]);

  /**
   * Draw MACD indicator
   */
  const drawMACD = useCallback((params: OscillatorPanelParams) => {
    const { ctx, data, x, y, width, height, theme } = params;
    const closes = data.map(d => d.close);
    const timestamps = data.map(d => d.time);
    const macd = calculateMACD(closes);

    // Find min/max for scaling
    let min = Infinity, max = -Infinity;
    macd.macd.forEach((val, i) => {
      if (!isNaN(val) && !isNaN(macd.signal[i]) && !isNaN(macd.histogram[i])) {
        min = Math.min(min, val, macd.signal[i], macd.histogram[i]);
        max = Math.max(max, val, macd.signal[i], macd.histogram[i]);
      }
    });

    // Add padding
    const range = max - min;
    min -= range * 0.1;
    max += range * 0.1;

    // Draw background
    ctx.fillStyle = theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, 0.02)'
      : 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(x, y, width, height);

    // Draw zero line
    drawReferenceLine(ctx, 0, x, y, width, height, min, max, '#666666');

    // Draw histogram
    const xScale = width / (timestamps.length - 1);
    ctx.save();
    macd.histogram.forEach((val, i) => {
      if (isNaN(val)) return;

      const xPos = x + i * xScale;
      const barWidth = Math.max(1, xScale * 0.8);
      const barHeight = Math.abs(val / (max - min) * height);
      const yPos = y + height - ((Math.max(0, val) - min) / (max - min)) * height;

      ctx.fillStyle = val >= 0 ? '#00FF88' : '#FF4444';
      ctx.fillRect(xPos - barWidth / 2, yPos, barWidth, val >= 0 ? -barHeight : barHeight);
    });
    ctx.restore();

    // Draw MACD line
    drawOscillatorLine({
      values: macd.macd,
      timestamps,
      ctx,
      x,
      y,
      width,
      height,
      min,
      max,
      color: '#2196F3',
      lineWidth: 2
    });

    // Draw signal line
    drawOscillatorLine({
      values: macd.signal,
      timestamps,
      ctx,
      x,
      y,
      width,
      height,
      min,
      max,
      color: '#FF5722',
      lineWidth: 2
    });

    // Draw label
    ctx.fillStyle = theme.palette.text.primary;
    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('MACD (12,26,9)', x + 10, y + 5);
  }, [drawOscillatorLine, drawReferenceLine]);

  /**
   * Draw Stochastic indicator
   */
  const drawStochastic = useCallback((params: OscillatorPanelParams) => {
    const { ctx, data, x, y, width, height, theme } = params;
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);
    const timestamps = data.map(d => d.time);
    const stoch = calculateStochastic(highs, lows, closes);

    // Draw background
    ctx.fillStyle = theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, 0.02)'
      : 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(x, y, width, height);

    // Draw reference lines
    drawReferenceLine(ctx, 80, x, y, width, height, 0, 100, '#FF4444', '80');
    drawReferenceLine(ctx, 50, x, y, width, height, 0, 100, '#666666', '50');
    drawReferenceLine(ctx, 20, x, y, width, height, 0, 100, '#00FF88', '20');

    // Draw %K line
    drawOscillatorLine({
      values: stoch.k,
      timestamps,
      ctx,
      x,
      y,
      width,
      height,
      min: 0,
      max: 100,
      color: '#2196F3',
      lineWidth: 2
    });

    // Draw %D line
    drawOscillatorLine({
      values: stoch.d,
      timestamps,
      ctx,
      x,
      y,
      width,
      height,
      min: 0,
      max: 100,
      color: '#FF5722',
      lineWidth: 2
    });

    // Draw label
    ctx.fillStyle = theme.palette.text.primary;
    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Stochastic (14,3,3)', x + 10, y + 5);
  }, [drawOscillatorLine, drawReferenceLine]);

  /**
   * Draw oscillator panel based on indicator type
   */
  const drawOscillatorPanel = useCallback((params: OscillatorPanelParams) => {
    const { indicator } = params;

    switch (indicator) {
      case 'rsi':
        drawRSI(params);
        break;
      case 'macd':
        drawMACD(params);
        break;
      case 'stochastic':
        drawStochastic(params);
        break;
      // Add more oscillators as needed
    }
  }, [drawRSI, drawMACD, drawStochastic]);

  return {
    drawOscillatorPanel,
  };
};
