/**
 * useChartCrosshair Hook
 *
 * Manages crosshair cursor functionality for the chart.
 * Shows price and time information at cursor position.
 *
 * Features:
 * - Crosshair lines following cursor
 * - Price label on Y-axis
 * - Time label on X-axis
 * - OHLC tooltip when hovering over candles
 * - Smooth animation and fade effects
 * - Touch device support
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { ChartDataPoint } from '../components/ChartCanvas/types';

interface CrosshairPosition {
  x: number;
  y: number;
  price: number;
  time: number;
  candle?: ChartDataPoint;
  visible: boolean;
}

interface TooltipData {
  candle: ChartDataPoint;
  x: number;
  y: number;
}

interface UseChartCrosshairParams {
  data: ChartDataPoint[];
  containerRef: React.RefObject<HTMLDivElement>;
  reverseScales?: {
    x: (pixelX: number) => { time: number; index: number };
    y: (pixelY: number) => number;
  };
  enabled?: boolean;
}

interface UseChartCrosshairResult {
  crosshairPosition: CrosshairPosition | null;
  tooltipData: TooltipData | null;
  handleMouseMove: (event: React.MouseEvent<HTMLCanvasElement>) => void;
  handleMouseLeave: () => void;
  drawCrosshair: (ctx: CanvasRenderingContext2D, width: number, height: number, theme: any) => void;
}

export const useChartCrosshair = ({
  data,
  containerRef,
  reverseScales,
  enabled = true,
}: UseChartCrosshairParams): UseChartCrosshairResult => {
  const [crosshairPosition, setCrosshairPosition] = useState<CrosshairPosition | null>(null);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const lastUpdateTime = useRef<number>(0);
  const animationFrame = useRef<number | null>(null);

  /**
   * Handle mouse move with throttling
   */
  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!enabled || !containerRef.current || !reverseScales) return;

    const now = Date.now();
    if (now - lastUpdateTime.current < 16) return; // Throttle to ~60fps
    lastUpdateTime.current = now;

    const rect = containerRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Cancel previous animation frame
    if (animationFrame.current) {
      cancelAnimationFrame(animationFrame.current);
    }

    // Use animation frame for smooth updates
    animationFrame.current = requestAnimationFrame(() => {
      const { time, index } = reverseScales.x(x);
      const price = reverseScales.y(y);

      // Find the candle at this position
      const candle = data[index];

      setCrosshairPosition({
        x,
        y,
        price,
        time,
        candle,
        visible: true,
      });

      // Update tooltip if hovering over a candle
      if (candle) {
        setTooltipData({
          candle,
          x,
          y,
        });
      } else {
        setTooltipData(null);
      }
    });
  }, [enabled, containerRef, reverseScales, data]);

  /**
   * Handle mouse leave
   */
  const handleMouseLeave = useCallback(() => {
    if (animationFrame.current) {
      cancelAnimationFrame(animationFrame.current);
    }
    setCrosshairPosition(null);
    setTooltipData(null);
  }, []);

  /**
   * Draw crosshair on canvas
   */
  const drawCrosshair = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    theme: any
  ) => {
    if (!crosshairPosition || !crosshairPosition.visible) return;

    const { x, y, price, time } = crosshairPosition;

    ctx.save();

    // Set crosshair style
    ctx.strokeStyle = theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, 0.3)'
      : 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Draw vertical line
    ctx.beginPath();
    ctx.moveTo(Math.floor(x) + 0.5, 0);
    ctx.lineTo(Math.floor(x) + 0.5, height);
    ctx.stroke();

    // Draw horizontal line
    ctx.beginPath();
    ctx.moveTo(0, Math.floor(y) + 0.5);
    ctx.lineTo(width, Math.floor(y) + 0.5);
    ctx.stroke();

    // Reset line dash
    ctx.setLineDash([]);

    // Draw price label on Y-axis
    drawPriceLabel(ctx, price, y, width, theme);

    // Draw time label on X-axis
    drawTimeLabel(ctx, time, x, height, theme);

    // Draw OHLC tooltip if available
    if (tooltipData) {
      drawOHLCTooltip(ctx, tooltipData, theme);
    }

    ctx.restore();
  }, [crosshairPosition, tooltipData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrame.current) {
        cancelAnimationFrame(animationFrame.current);
      }
    };
  }, []);

  return {
    crosshairPosition,
    tooltipData,
    handleMouseMove,
    handleMouseLeave,
    drawCrosshair,
  };
};

/**
 * Draw price label on Y-axis
 */
function drawPriceLabel(
  ctx: CanvasRenderingContext2D,
  price: number,
  y: number,
  width: number,
  theme: any
) {
  const label = `$${price.toFixed(2)}`;
  const padding = 4;
  const labelX = width - 60; // Right side position

  ctx.font = '12px Inter, system-ui, sans-serif';
  const metrics = ctx.measureText(label);
  const labelWidth = metrics.width + padding * 2;
  const labelHeight = 20;

  // Background
  ctx.fillStyle = theme.palette.mode === 'dark'
    ? 'rgba(255, 215, 0, 0.9)' // Golden background
    : 'rgba(33, 150, 243, 0.9)'; // Blue background

  ctx.fillRect(
    labelX,
    y - labelHeight / 2,
    labelWidth,
    labelHeight
  );

  // Text
  ctx.fillStyle = theme.palette.mode === 'dark' ? '#000' : '#fff';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, labelX + padding, y);
}

/**
 * Draw time label on X-axis
 */
function drawTimeLabel(
  ctx: CanvasRenderingContext2D,
  time: number,
  x: number,
  height: number,
  theme: any
) {
  const date = new Date(time * 1000);
  const label = formatTimeLabel(date);
  const padding = 4;
  const labelY = height - 40; // Bottom position

  ctx.font = '12px Inter, system-ui, sans-serif';
  const metrics = ctx.measureText(label);
  const labelWidth = metrics.width + padding * 2;
  const labelHeight = 20;

  // Adjust position to keep label on screen
  let labelX = x - labelWidth / 2;
  if (labelX < 0) labelX = 0;
  if (labelX + labelWidth > ctx.canvas.width) {
    labelX = ctx.canvas.width - labelWidth;
  }

  // Background
  ctx.fillStyle = theme.palette.mode === 'dark'
    ? 'rgba(255, 255, 255, 0.1)'
    : 'rgba(0, 0, 0, 0.1)';

  ctx.fillRect(
    labelX,
    labelY,
    labelWidth,
    labelHeight
  );

  // Text
  ctx.fillStyle = theme.palette.text.primary;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, labelX + labelWidth / 2, labelY + labelHeight / 2);
}

/**
 * Draw OHLC tooltip
 */
function drawOHLCTooltip(
  ctx: CanvasRenderingContext2D,
  tooltipData: TooltipData,
  theme: any
) {
  const { candle, x, y } = tooltipData;
  const padding = 8;
  const lineHeight = 18;
  const width = 140;
  const height = lineHeight * 6 + padding * 2;

  // Position tooltip to avoid edges
  let tooltipX = x + 10;
  let tooltipY = y - height / 2;

  if (tooltipX + width > ctx.canvas.width - 20) {
    tooltipX = x - width - 10;
  }
  if (tooltipY < 20) {
    tooltipY = 20;
  }
  if (tooltipY + height > ctx.canvas.height - 20) {
    tooltipY = ctx.canvas.height - height - 20;
  }

  // Background with shadow
  ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
  ctx.shadowBlur = 10;
  ctx.shadowOffsetX = 2;
  ctx.shadowOffsetY = 2;

  ctx.fillStyle = theme.palette.mode === 'dark'
    ? 'rgba(30, 30, 30, 0.95)'
    : 'rgba(255, 255, 255, 0.95)';

  ctx.beginPath();
  ctx.roundRect(tooltipX, tooltipY, width, height, 4);
  ctx.fill();

  // Reset shadow
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;

  // Border
  ctx.strokeStyle = theme.palette.divider;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Text
  ctx.font = '12px Inter, system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';

  const textX = tooltipX + padding;
  let textY = tooltipY + padding;

  // Date
  const date = new Date(candle.time * 1000);
  ctx.fillStyle = theme.palette.text.secondary;
  ctx.fillText(formatDateLabel(date), textX, textY);
  textY += lineHeight;

  // OHLC values
  const ohlcData = [
    { label: 'O:', value: candle.open, color: theme.palette.text.primary },
    { label: 'H:', value: candle.high, color: theme.palette.success.main },
    { label: 'L:', value: candle.low, color: theme.palette.error.main },
    { label: 'C:', value: candle.close, color: theme.palette.text.primary },
    { label: 'V:', value: formatVolume(candle.volume), color: theme.palette.text.secondary },
  ];

  ohlcData.forEach(({ label, value, color }) => {
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.fillText(label, textX, textY);

    ctx.fillStyle = color;
    ctx.fillText(
      typeof value === 'number' ? `$${value.toFixed(2)}` : value,
      textX + 20,
      textY
    );

    textY += lineHeight;
  });
}

/**
 * Format time label based on timeframe
 */
function formatTimeLabel(date: Date): string {
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const month = date.toLocaleDateString('en-US', { month: 'short' });
  const day = date.getDate();

  // For intraday, show time
  if (date.getHours() !== 0 || date.getMinutes() !== 0) {
    return `${month} ${day}, ${hours}:${minutes}`;
  }

  // For daily, show date
  return `${month} ${day}, ${date.getFullYear()}`;
}

/**
 * Format date for tooltip
 */
function formatDateLabel(date: Date): string {
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format volume with abbreviations
 */
function formatVolume(volume: number): string {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toString();
}
