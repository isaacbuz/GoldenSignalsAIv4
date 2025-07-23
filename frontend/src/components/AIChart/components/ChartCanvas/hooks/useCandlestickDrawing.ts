/**
 * useCandlestickDrawing Hook
 *
 * Handles the drawing of candlestick charts on canvas.
 * Includes rendering logic for OHLC data with proper styling
 * and animations.
 *
 * Features:
 * - Gradient candlesticks
 * - Shadow effects
 * - Smooth animations
 * - Responsive sizing
 */

import { useCallback, useRef } from 'react';
import { DrawingParams, CandleStyle } from '../types';

/**
 * Draw a single candlestick
 */
const drawCandle = (
  ctx: CanvasRenderingContext2D,
  x: number,
  open: number,
  high: number,
  low: number,
  close: number,
  yScale: (price: number) => number,
  candleWidth: number,
  style: CandleStyle,
  isGapFilled: boolean = false
) => {
  const isBullish = close >= open;
  const color = isBullish ? style.bullishColor : style.bearishColor;
  const bodyTop = yScale(Math.max(open, close));
  const bodyBottom = yScale(Math.min(open, close));
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1);

  // For gap-filled candles, draw a dotted line instead
  if (isGapFilled) {
    ctx.save();
    ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)'; // Gray with transparency
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]); // Dotted line pattern
    ctx.beginPath();
    ctx.moveTo(x - candleWidth / 2, yScale(close));
    ctx.lineTo(x + candleWidth / 2, yScale(close));
    ctx.stroke();
    ctx.restore();
    return; // Skip normal candle drawing
  }

  // Draw wick (high-low line)
  ctx.strokeStyle = color;
  ctx.lineWidth = style.wickWidth;
  ctx.beginPath();
  ctx.moveTo(x, yScale(high));
  ctx.lineTo(x, yScale(low));
  ctx.stroke();

  // Draw body with gradient
  const gradient = ctx.createLinearGradient(
    x - candleWidth / 2,
    bodyTop,
    x + candleWidth / 2,
    bodyBottom
  );

  if (isBullish) {
    gradient.addColorStop(0, '#00FF88');
    gradient.addColorStop(1, '#00CC66');
  } else {
    gradient.addColorStop(0, '#FF4444');
    gradient.addColorStop(1, '#CC0000');
  }

  // Apply shadow effect if specified
  if (style.shadowBlur) {
    ctx.shadowBlur = style.shadowBlur;
    ctx.shadowColor = style.shadowColor || color;
  }

  // Draw candle body
  ctx.fillStyle = gradient;
  ctx.globalAlpha = style.bodyOpacity;
  ctx.fillRect(
    x - candleWidth / 2,
    bodyTop,
    candleWidth,
    bodyHeight
  );

  // Reset shadow and alpha
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
};

/**
 * Hook for candlestick drawing functionality
 */
export const useCandlestickDrawing = () => {
  const animationFrameRef = useRef<number>();
  const startTimeRef = useRef<number>(Date.now());

  /**
   * Draw candlesticks with animation
   */
  const drawCandlesticks = useCallback((params: DrawingParams) => {
    const { ctx, data, xScale, yScale, chartWidth, theme } = params;

    if (!ctx || data.length === 0) return;

    // Clear previous animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    // Calculate candle width
    const candleWidth = Math.max(1, (chartWidth / data.length) * 0.8);

    // Define candle style
    const candleStyle: CandleStyle = {
      bullishColor: theme.palette.mode === 'dark' ? '#00FF88' : '#4CAF50',
      bearishColor: theme.palette.mode === 'dark' ? '#FF4444' : '#F44336',
      wickWidth: Math.max(1, candleWidth * 0.1),
      bodyOpacity: theme.palette.mode === 'dark' ? 0.6 : 0.8,
      shadowBlur: theme.palette.mode === 'dark' ? 2 : 0,
      shadowColor: theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.3)' : undefined,
    };

    // Animation function
    const animate = () => {
      const elapsed = Date.now() - startTimeRef.current;
      const progress = Math.min(elapsed / 1500, 1); // 1.5s animation

      // Clear canvas area
      ctx.save();
      ctx.clearRect(0, 0, chartWidth, params.chartHeight);

      // Draw each candle with animation
      data.forEach((candle, index) => {
        const x = xScale(candle.time, index);
        const animatedProgress = Math.min(progress * data.length / index, 1);

        if (animatedProgress > 0) {
          ctx.save();
          ctx.globalAlpha = animatedProgress;

          drawCandle(
            ctx,
            x,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            yScale,
            candleWidth,
            candleStyle,
            candle.isGapFilled || false
          );

          ctx.restore();
        }
      });

      ctx.restore();

      // Continue animation if not complete
      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    // Start animation
    startTimeRef.current = Date.now();
    animate();
  }, []);

  /**
   * Draw a single candle update (for real-time updates)
   */
  const updateLastCandle = useCallback((
    params: DrawingParams,
    lastCandle: any
  ) => {
    const { ctx, data, xScale, yScale, chartWidth, theme } = params;

    if (!ctx || data.length === 0) return;

    const candleWidth = Math.max(1, (chartWidth / data.length) * 0.8);
    const lastIndex = data.length - 1;
    const x = xScale(lastCandle.time, lastIndex);

    // Define candle style
    const candleStyle: CandleStyle = {
      bullishColor: theme.palette.mode === 'dark' ? '#00FF88' : '#4CAF50',
      bearishColor: theme.palette.mode === 'dark' ? '#FF4444' : '#F44336',
      wickWidth: Math.max(1, candleWidth * 0.1),
      bodyOpacity: theme.palette.mode === 'dark' ? 0.6 : 0.8,
      shadowBlur: theme.palette.mode === 'dark' ? 4 : 0, // More glow for updates
      shadowColor: theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.5)' : undefined,
    };

    // Clear last candle area with some padding
    const clearX = x - candleWidth;
    const clearWidth = candleWidth * 2;
    ctx.clearRect(clearX, 0, clearWidth, params.chartHeight);

    // Draw updated candle
    drawCandle(
      ctx,
      x,
      lastCandle.open,
      lastCandle.high,
      lastCandle.low,
      lastCandle.close,
      yScale,
      candleWidth,
      candleStyle,
      lastCandle.isGapFilled || false
    );
  }, []);

  return {
    drawCandlesticks,
    updateLastCandle,
  };
};
