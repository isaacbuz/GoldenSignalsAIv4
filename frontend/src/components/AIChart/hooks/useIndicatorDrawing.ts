/**
 * useIndicatorDrawing Hook
 *
 * Handles the drawing of technical indicators and volume bars
 * using the new coordinate system.
 */

import { useCallback } from 'react';
import { CoordinateSystem } from '../utils/coordinateSystem';
import { ChartDataPoint } from '../components/ChartCanvas/types';
import {
  calculateRSI,
  calculateMACD,
  calculateSMA,
  calculateEMA,
  calculateVWAP,
  calculateBollingerBands,
  calculateStochastic,
  calculateATR,
  calculateADX
} from '../../../utils/technicalIndicators';

interface DrawIndicatorsParams {
  ctx: CanvasRenderingContext2D;
  data: ChartDataPoint[];
  coordinates: CoordinateSystem;
  theme: any;
  indicators: string[];
}

interface DrawVolumeParams {
  ctx: CanvasRenderingContext2D;
  data: ChartDataPoint[];
  coordinates: CoordinateSystem;
  theme: any;
  volumeHeight: number;
}

export const useIndicatorDrawing = () => {
  /**
   * Draw moving average line
   */
  const drawMovingAverage = useCallback((
    ctx: CanvasRenderingContext2D,
    values: number[],
    timestamps: number[],
    coordinates: CoordinateSystem,
    color: string,
    lineWidth: number = 2
  ) => {
    if (values.length < 2) return;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.shadowBlur = 5;
    ctx.shadowColor = color;

    ctx.beginPath();
    let started = false;

    for (let i = 0; i < values.length; i++) {
      if (isNaN(values[i])) continue;

      const x = coordinates.timeToX(timestamps[i]);
      const y = coordinates.priceToY(values[i]);

      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
    ctx.restore();
  }, []);

  /**
   * Draw Bollinger Bands
   */
  const drawBollingerBands = useCallback((
    ctx: CanvasRenderingContext2D,
    data: ChartDataPoint[],
    coordinates: CoordinateSystem,
    theme: any
  ) => {
    const closes = data.map(d => d.close);
    const timestamps = data.map(d => d.time);
    const sma20 = calculateSMA(closes, 20);

    // Calculate standard deviation
    const upperBand: number[] = [];
    const lowerBand: number[] = [];

    for (let i = 19; i < closes.length; i++) {
      const slice = closes.slice(i - 19, i + 1);
      const mean = sma20[i];
      if (isNaN(mean)) continue;

      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / 20;
      const stdDev = Math.sqrt(variance);

      upperBand[i] = mean + 2 * stdDev;
      lowerBand[i] = mean - 2 * stdDev;
    }

    // Draw filled area between bands
    ctx.save();
    ctx.fillStyle = theme.palette.mode === 'dark'
      ? 'rgba(255, 215, 0, 0.05)'
      : 'rgba(33, 150, 243, 0.05)';

    ctx.beginPath();
    let started = false;

    // Draw upper band path
    for (let i = 0; i < upperBand.length; i++) {
      if (isNaN(upperBand[i])) continue;

      const x = coordinates.timeToX(timestamps[i]);
      const y = coordinates.priceToY(upperBand[i]);

      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }

    // Draw lower band path in reverse
    for (let i = lowerBand.length - 1; i >= 0; i--) {
      if (isNaN(lowerBand[i])) continue;

      const x = coordinates.timeToX(timestamps[i]);
      const y = coordinates.priceToY(lowerBand[i]);

      ctx.lineTo(x, y);
    }

    ctx.closePath();
    ctx.fill();

    // Draw the bands
    const bandColor = theme.palette.mode === 'dark' ? '#FFD700' : '#2196F3';
    drawMovingAverage(ctx, upperBand, timestamps, coordinates, bandColor, 1);
    drawMovingAverage(ctx, lowerBand, timestamps, coordinates, bandColor, 1);
    drawMovingAverage(ctx, sma20, timestamps, coordinates, bandColor, 2);

    ctx.restore();
  }, [drawMovingAverage]);

  /**
   * Draw volume bars
   */
  const drawVolume = useCallback((params: DrawVolumeParams) => {
    const { ctx, data, coordinates, theme, volumeHeight } = params;

    if (data.length === 0) return;

    const viewport = coordinates.getViewportBounds();
    const volumeTop = viewport.y + viewport.height - volumeHeight;

    // Find max volume for scaling
    const maxVolume = Math.max(...data.map(d => d.volume));
    if (maxVolume === 0) return;

    // Calculate bar width
    const barCount = data.length;
    const barWidth = Math.max(1, (viewport.width / barCount) * 0.8);

    ctx.save();

    // Draw volume background
    ctx.fillStyle = theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, 0.02)'
      : 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(viewport.x, volumeTop, viewport.width, volumeHeight);

    // Draw volume label
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.font = '12px Inter, system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Volume', viewport.x + 10, volumeTop + 5);

    // Draw volume bars
    data.forEach((candle, index) => {
      const x = coordinates.timeToX(candle.time);
      const volumePercent = candle.volume / maxVolume;
      const barHeight = volumePercent * (volumeHeight - 20);

      // Skip bars outside viewport
      if (x < viewport.x - barWidth || x > viewport.x + viewport.width + barWidth) {
        return;
      }

      // Color based on price change
      const isBullish = index === 0 || candle.close >= data[Math.max(0, index - 1)].close;
      const color = isBullish
        ? (theme.palette.mode === 'dark' ? '#00FF8880' : '#4CAF5080')
        : (theme.palette.mode === 'dark' ? '#FF444480' : '#F4433680');

      // Create gradient
      const gradient = ctx.createLinearGradient(0, volumeTop + volumeHeight, 0, volumeTop + volumeHeight - barHeight);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, color.replace('80', 'FF'));

      ctx.fillStyle = gradient;
      ctx.fillRect(
        x - barWidth / 2,
        volumeTop + volumeHeight - barHeight,
        barWidth,
        barHeight
      );
    });

    ctx.restore();
  }, []);

  /**
   * Draw all selected indicators
   */
  const drawIndicators = useCallback((params: DrawIndicatorsParams) => {
    const { ctx, data, coordinates, theme, indicators } = params;

    if (data.length === 0 || indicators.length === 0) return;

    const closes = data.map(d => d.close);
    const timestamps = data.map(d => d.time);

    // Draw each selected indicator
    indicators.forEach(indicator => {
      switch (indicator) {
        // Moving Averages
        case 'sma-20':
          const sma20 = calculateSMA(closes, 20);
          drawMovingAverage(ctx, sma20, timestamps, coordinates, '#FFD700', 2);
          break;

        case 'sma-50':
          const sma50 = calculateSMA(closes, 50);
          drawMovingAverage(ctx, sma50, timestamps, coordinates, '#FF6B6B', 2);
          break;

        case 'sma-200':
          const sma200 = calculateSMA(closes, 200);
          drawMovingAverage(ctx, sma200, timestamps, coordinates, '#9C27B0', 2);
          break;

        case 'ema-12':
          const ema12 = calculateEMA(closes, 12);
          drawMovingAverage(ctx, ema12, timestamps, coordinates, '#00FF88', 2);
          break;

        case 'ema-26':
          const ema26 = calculateEMA(closes, 26);
          drawMovingAverage(ctx, ema26, timestamps, coordinates, '#FF5722', 2);
          break;

        case 'vwap':
          const volumes = data.map(d => d.volume);
          const vwap = calculateVWAP(closes, volumes);
          drawMovingAverage(ctx, vwap, timestamps, coordinates, '#2196F3', 3);
          break;

        case 'bollinger':
          drawBollingerBands(ctx, data, coordinates, theme);
          break;

        // Note: RSI, MACD, Stochastic, ATR, ADX require separate panels
        // They will be drawn in a dedicated oscillator panel
      }
    });
  }, [drawMovingAverage, drawBollingerBands]);

  return {
    drawIndicators,
    drawVolume,
  };
};
