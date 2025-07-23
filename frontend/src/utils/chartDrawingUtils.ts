/**
 * Chart Drawing Utilities
 * Helper functions for drawing trading levels on canvas
 */

import { TradingAdvice } from '../services/llmAdvisor';

export interface DrawingOptions {
  ctx: CanvasRenderingContext2D;
  xScale: (time: number, index: number) => number;
  yScale: (price: number) => number;
  theme: any;
  currentTime: number;
  dataLength: number;
}

export const drawSignalLines = (
  advice: TradingAdvice,
  options: DrawingOptions,
  opacity: number = 1
) => {
  const { ctx, yScale, theme } = options;

  ctx.save();
  ctx.globalAlpha = opacity;

  // Entry line (green for buy, red for sell)
  if (advice.entry_price > 0) {
    const entryY = yScale(advice.entry_price);
    const entryColor = advice.action === 'BUY' ? '#00FF88' : '#FF4444';

    drawHorizontalLine(ctx, entryY, entryColor, 'Entry', advice.entry_price, options);
  }

  // Stop loss line (always red dashed)
  if (advice.stop_loss > 0) {
    const slY = yScale(advice.stop_loss);

    ctx.strokeStyle = '#FF4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);

    drawHorizontalLine(ctx, slY, '#FF4444', 'SL', advice.stop_loss, options);
  }

  // Take profit lines (blue dashed)
  advice.take_profits.forEach((tp, index) => {
    if (tp > 0) {
      const tpY = yScale(tp);

      ctx.strokeStyle = '#4A90E2';
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 5]);
      ctx.globalAlpha = opacity * (1 - index * 0.2); // Fade further TPs

      drawHorizontalLine(ctx, tpY, '#4A90E2', `TP${index + 1}`, tp, options);
    }
  });

  ctx.restore();
};

const drawHorizontalLine = (
  ctx: CanvasRenderingContext2D,
  y: number,
  color: string,
  label: string,
  price: number,
  options: DrawingOptions
) => {
  const { theme } = options;
  const canvasWidth = ctx.canvas.width;
  const padding = 70;

  // Draw line
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(padding, y);
  ctx.lineTo(canvasWidth - padding, y);
  ctx.stroke();

  // Add glow effect in dark mode
  if (theme.palette.mode === 'dark') {
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Draw label
  const labelX = canvasWidth - padding + 5;
  ctx.fillStyle = color;
  ctx.font = '12px "SF Pro Display", sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`${label}: $${price.toFixed(2)}`, labelX, y + 4);
};

export const drawEntryZone = (
  advice: TradingAdvice,
  options: DrawingOptions,
  opacity: number = 0.2
) => {
  const { ctx, yScale, theme } = options;

  if (!advice.metadata?.entry_zone) return;

  const zone = advice.metadata.entry_zone;
  const topY = yScale(zone.top);
  const bottomY = yScale(zone.bottom);
  const height = Math.abs(bottomY - topY);

  ctx.save();

  // Create gradient fill
  const gradient = ctx.createLinearGradient(0, topY, 0, bottomY);
  const baseColor = advice.action === 'BUY' ? '#00FF88' : '#FF4444';

  gradient.addColorStop(0, `${baseColor}00`);
  gradient.addColorStop(0.5, `${baseColor}${Math.floor(opacity * 255).toString(16)}`);
  gradient.addColorStop(1, `${baseColor}00`);

  ctx.fillStyle = gradient;
  ctx.fillRect(70, topY, ctx.canvas.width - 140, height);

  // Add border
  ctx.strokeStyle = baseColor;
  ctx.lineWidth = 1;
  ctx.globalAlpha = opacity * 2;
  ctx.setLineDash([5, 5]);
  ctx.strokeRect(70, topY, ctx.canvas.width - 140, height);

  ctx.restore();
};

export const animateDrawing = (
  drawFunc: (progress: number) => void,
  duration: number = 500,
  easingFunc: (t: number) => number = easeInOutCubic
) => {
  const startTime = Date.now();

  const animate = () => {
    const elapsed = Date.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easedProgress = easingFunc(progress);

    drawFunc(easedProgress);

    if (progress < 1) {
      requestAnimationFrame(animate);
    }
  };

  requestAnimationFrame(animate);
};

const easeInOutCubic = (t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};
