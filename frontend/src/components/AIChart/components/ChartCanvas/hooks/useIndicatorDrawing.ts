/**
 * useIndicatorDrawing Hook
 *
 * Handles the drawing of technical indicators on the chart.
 * Supports various indicators like moving averages, Bollinger Bands,
 * and volume bars.
 *
 * Features:
 * - Multiple indicator types
 * - Smooth line rendering
 * - Gradient fills
 * - Dynamic colors based on theme
 */

import { useCallback } from 'react';
import { IndicatorDrawingParams, LineStyle, VolumeDrawingParams } from '../types';

/**
 * Calculate Simple Moving Average
 */
const calculateSMA = (data: any[], period: number): (number | null)[] => {
  const sma: (number | null)[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      sma.push(null);
    } else {
      const sum = data
        .slice(i - period + 1, i + 1)
        .reduce((acc, candle) => acc + candle.close, 0);
      sma.push(sum / period);
    }
  }

  return sma;
};

/**
 * Calculate Bollinger Bands
 */
const calculateBollingerBands = (data: any[], period: number = 20, stdDev: number = 2) => {
  const sma = calculateSMA(data, period);
  const upperBand: (number | null)[] = [];
  const lowerBand: (number | null)[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1 || sma[i] === null) {
      upperBand.push(null);
      lowerBand.push(null);
    } else {
      const values = data.slice(i - period + 1, i + 1).map(d => d.close);
      const mean = sma[i]!;
      const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
      const stdDeviation = Math.sqrt(variance);

      upperBand.push(mean + stdDev * stdDeviation);
      lowerBand.push(mean - stdDev * stdDeviation);
    }
  }

  return { sma, upperBand, lowerBand };
};

/**
 * Draw a line on the chart
 */
const drawLine = (
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  style: LineStyle
) => {
  if (points.length < 2) return;

  ctx.save();
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.width;

  if (style.dash) {
    ctx.setLineDash(style.dash);
  }

  if (style.shadowBlur) {
    ctx.shadowBlur = style.shadowBlur;
    ctx.shadowColor = style.shadowColor || style.color;
  }

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);

  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }

  ctx.stroke();
  ctx.restore();
};

/**
 * Draw filled area between two lines
 */
const drawFilledArea = (
  ctx: CanvasRenderingContext2D,
  upperPoints: { x: number; y: number }[],
  lowerPoints: { x: number; y: number }[],
  fillColor: string
) => {
  if (upperPoints.length < 2 || lowerPoints.length < 2) return;

  ctx.save();
  ctx.fillStyle = fillColor;

  ctx.beginPath();
  ctx.moveTo(upperPoints[0].x, upperPoints[0].y);

  // Draw upper line
  for (let i = 1; i < upperPoints.length; i++) {
    ctx.lineTo(upperPoints[i].x, upperPoints[i].y);
  }

  // Draw lower line in reverse
  for (let i = lowerPoints.length - 1; i >= 0; i--) {
    ctx.lineTo(lowerPoints[i].x, lowerPoints[i].y);
  }

  ctx.closePath();
  ctx.fill();
  ctx.restore();
};

/**
 * Hook for indicator drawing functionality
 */
export const useIndicatorDrawing = () => {
  /**
   * Draw all selected indicators
   */
  const drawIndicators = useCallback((params: IndicatorDrawingParams) => {
    const { ctx, data, xScale, yScale, indicators, theme, chartWidth } = params;

    if (!ctx || data.length === 0) return;

    // Define indicator styles
    const isDark = theme.palette.mode === 'dark';
    const indicatorStyles = {
      'sma-20': {
        color: isDark ? '#FFD700' : '#FF9800',
        width: 2,
        shadowBlur: isDark ? 5 : 0,
      },
      'sma-50': {
        color: isDark ? '#00BFFF' : '#2196F3',
        width: 2,
        shadowBlur: isDark ? 5 : 0,
      },
      'ema-20': {
        color: isDark ? '#FF69B4' : '#E91E63',
        width: 2,
        shadowBlur: isDark ? 5 : 0,
      },
      'ema-50': {
        color: isDark ? '#9370DB' : '#9C27B0',
        width: 2,
        shadowBlur: isDark ? 5 : 0,
      },
    };

    // Draw each selected indicator
    indicators.forEach(indicator => {
      switch (indicator) {
        case 'sma-20': {
          const sma = calculateSMA(data, 20);
          const points = sma
            .map((value, index) =>
              value !== null ? { x: xScale(data[index].time, index), y: yScale(value) } : null
            )
            .filter(p => p !== null) as { x: number; y: number }[];

          drawLine(ctx, points, indicatorStyles['sma-20']);
          break;
        }

        case 'sma-50': {
          const sma = calculateSMA(data, 50);
          const points = sma
            .map((value, index) =>
              value !== null ? { x: xScale(data[index].time, index), y: yScale(value) } : null
            )
            .filter(p => p !== null) as { x: number; y: number }[];

          drawLine(ctx, points, indicatorStyles['sma-50']);
          break;
        }

        case 'bollinger': {
          const { sma, upperBand, lowerBand } = calculateBollingerBands(data);

          // Convert to points
          const middlePoints = sma
            .map((value, index) =>
              value !== null ? { x: xScale(data[index].time, index), y: yScale(value) } : null
            )
            .filter(p => p !== null) as { x: number; y: number }[];

          const upperPoints = upperBand
            .map((value, index) =>
              value !== null ? { x: xScale(data[index].time, index), y: yScale(value) } : null
            )
            .filter(p => p !== null) as { x: number; y: number }[];

          const lowerPoints = lowerBand
            .map((value, index) =>
              value !== null ? { x: xScale(data[index].time, index), y: yScale(value) } : null
            )
            .filter(p => p !== null) as { x: number; y: number }[];

          // Draw filled area
          const fillColor = isDark
            ? 'rgba(100, 149, 237, 0.1)'
            : 'rgba(100, 149, 237, 0.05)';
          drawFilledArea(ctx, upperPoints, lowerPoints, fillColor);

          // Draw lines
          const bandStyle: LineStyle = {
            color: isDark ? '#6495ED' : '#1976D2',
            width: 1,
            dash: [5, 5],
          };

          drawLine(ctx, upperPoints, bandStyle);
          drawLine(ctx, lowerPoints, bandStyle);
          drawLine(ctx, middlePoints, {
            ...bandStyle,
            width: 2,
            dash: undefined,
          });
          break;
        }
      }
    });
  }, []);

  /**
   * Draw volume bars
   */
  const drawVolume = useCallback((params: VolumeDrawingParams) => {
    const { ctx, data, xScale, theme, chartWidth, volumeHeight, volumeScale } = params;

    if (!ctx || data.length === 0 || !volumeScale) return;

    const barWidth = Math.max(1, (chartWidth / data.length) * 0.8);
    const isDark = theme.palette.mode === 'dark';

    // Draw volume bars
    data.forEach((candle, index) => {
      const x = xScale(candle.time, index);
      const height = volumeScale(candle.volume);
      const isBullish = candle.close >= candle.open;

      // Create gradient for volume bars
      const gradient = ctx.createLinearGradient(
        x - barWidth / 2,
        params.chartHeight - volumeHeight,
        x - barWidth / 2,
        params.chartHeight
      );

      if (isDark) {
        if (isBullish) {
          gradient.addColorStop(0, 'rgba(0, 255, 136, 0.3)');
          gradient.addColorStop(1, 'rgba(0, 255, 136, 0.1)');
        } else {
          gradient.addColorStop(0, 'rgba(255, 68, 68, 0.3)');
          gradient.addColorStop(1, 'rgba(255, 68, 68, 0.1)');
        }
      } else {
        if (isBullish) {
          gradient.addColorStop(0, 'rgba(76, 175, 80, 0.3)');
          gradient.addColorStop(1, 'rgba(76, 175, 80, 0.1)');
        } else {
          gradient.addColorStop(0, 'rgba(244, 67, 54, 0.3)');
          gradient.addColorStop(1, 'rgba(244, 67, 54, 0.1)');
        }
      }

      ctx.fillStyle = gradient;
      ctx.fillRect(
        x - barWidth / 2,
        params.chartHeight - height,
        barWidth,
        height
      );
    });

    // Draw volume label
    ctx.save();
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.font = '12px Inter, system-ui, -apple-system';
    ctx.globalAlpha = 0.6;
    ctx.fillText('Volume', 10, params.chartHeight - volumeHeight + 20);
    ctx.restore();
  }, []);

  return {
    drawIndicators,
    drawVolume,
  };
};
