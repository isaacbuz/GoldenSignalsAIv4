/**
 * useCandlestickDrawing Hook
 *
 * Handles the drawing of candlestick charts using the new coordinate system.
 * Features gradient candlesticks with shadow effects and smooth animations.
 */

import { useCallback } from 'react';
import { CoordinateSystem } from '../utils/coordinateSystem';
import { ChartDataPoint } from '../components/ChartCanvas/types';

interface DrawCandlesticksParams {
  ctx: CanvasRenderingContext2D;
  data: ChartDataPoint[];
  coordinates: CoordinateSystem;
  theme: any;
}

interface UpdateLastCandleParams {
  ctx: CanvasRenderingContext2D;
  candle: ChartDataPoint;
  coordinates: CoordinateSystem;
  theme: any;
}

export const useCandlestickDrawing = () => {
  /**
   * Draw a single candlestick
   */
  const drawCandle = useCallback((
    ctx: CanvasRenderingContext2D,
    candle: ChartDataPoint,
    coordinates: CoordinateSystem,
    candleWidth: number,
    theme: any
  ) => {
    const x = coordinates.timeToX(candle.time);
    const openY = coordinates.priceToY(candle.open);
    const highY = coordinates.priceToY(candle.high);
    const lowY = coordinates.priceToY(candle.low);
    const closeY = coordinates.priceToY(candle.close);

    const isBullish = candle.close >= candle.open;
    const bodyTop = Math.min(openY, closeY);
    const bodyBottom = Math.max(openY, closeY);
    const bodyHeight = Math.max(bodyBottom - bodyTop, 1);

    // Colors based on theme
    const bullishColor = theme.palette.mode === 'dark' ? '#00FF88' : '#4CAF50';
    const bearishColor = theme.palette.mode === 'dark' ? '#FF4444' : '#F44336';
    const color = isBullish ? bullishColor : bearishColor;

    // Draw shadow/wick with gradient effect
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, candleWidth * 0.1); // Proportional wick width
    ctx.globalAlpha = 0.6;
    ctx.shadowBlur = 2;
    ctx.shadowColor = color;

    ctx.beginPath();
    ctx.moveTo(x, highY);
    ctx.lineTo(x, lowY);
    ctx.stroke();
    ctx.restore();

    // Draw body with gradient
    ctx.save();

    if (isBullish) {
      // Create gradient for bullish candle
      const gradient = ctx.createLinearGradient(
        x - candleWidth / 2, bodyTop,
        x + candleWidth / 2, bodyBottom
      );
      gradient.addColorStop(0, color);
      gradient.addColorStop(0.5, color + 'CC');
      gradient.addColorStop(1, color + '88');
      ctx.fillStyle = gradient;
    } else {
      // Create gradient for bearish candle
      const gradient = ctx.createLinearGradient(
        x - candleWidth / 2, bodyTop,
        x + candleWidth / 2, bodyBottom
      );
      gradient.addColorStop(0, color + '88');
      gradient.addColorStop(0.5, color + 'CC');
      gradient.addColorStop(1, color);
      ctx.fillStyle = gradient;
    }

    // Draw rounded rectangle body
    const radius = Math.min(candleWidth / 4, 3);
    ctx.beginPath();
    ctx.roundRect(
      x - candleWidth / 2,
      bodyTop,
      candleWidth,
      bodyHeight,
      radius
    );
    ctx.fill();

    // Add subtle border
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.restore();
  }, []);

  /**
   * Draw all candlesticks
   */
  const drawCandlesticks = useCallback((params: DrawCandlesticksParams) => {
    const { ctx, data, coordinates, theme } = params;

    if (data.length === 0) return;

    // Calculate candle width based on visible range, not total data
    const viewport = coordinates.getViewportBounds();
    const visibleData = data.filter(candle => {
      const x = coordinates.timeToX(candle.time);
      return x >= viewport.x - 50 && x <= viewport.x + viewport.width + 50;
    });

    // Base candle width on visible candles for consistent sizing
    const visibleCount = Math.max(visibleData.length, 1);
    const availableWidth = viewport.width / visibleCount;

    // Adjust candle body percentage based on data density
    // More candles = thinner bodies, fewer candles = thicker bodies
    const bodyPercentage = visibleCount > 100 ? 0.7 :
                           visibleCount > 50 ? 0.8 :
                           visibleCount > 20 ? 0.85 : 0.9;

    const candleWidth = Math.max(2, Math.min(availableWidth * bodyPercentage, 50));

    // Draw each candle
    data.forEach((candle) => {
      // Skip candles outside viewport
      const x = coordinates.timeToX(candle.time);
      if (x < viewport.x - candleWidth || x > viewport.x + viewport.width + candleWidth) {
        return;
      }

      drawCandle(ctx, candle, coordinates, candleWidth, theme);
    });
  }, [drawCandle]);

  /**
   * Update only the last candle (for real-time updates)
   */
  const updateLastCandle = useCallback((params: UpdateLastCandleParams) => {
    const { ctx, candle, coordinates, theme } = params;

    // Calculate candle position and clear area
    const x = coordinates.timeToX(candle.time);
    const viewport = coordinates.getViewportBounds();
    const candleWidth = 20; // Use fixed width for last candle

    // Clear the area around the last candle
    ctx.save();
    ctx.clearRect(
      x - candleWidth,
      viewport.y,
      candleWidth * 2,
      viewport.height
    );
    ctx.restore();

    // Redraw the candle
    drawCandle(ctx, candle, coordinates, candleWidth, theme);
  }, [drawCandle]);

  return {
    drawCandlesticks,
    updateLastCandle,
  };
};
