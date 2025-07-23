/**
 * useAgentLevelDrawing Hook
 *
 * Handles the drawing of agent-generated trading levels on the chart.
 * This includes entry points, stop losses, take profits, and support/resistance levels.
 *
 * Features:
 * - Entry/exit level visualization
 * - Stop loss and take profit zones
 * - Support/resistance lines
 * - Animated level indicators
 * - Interactive hover effects
 */

import { useCallback, useRef } from 'react';
import { AgentLevelDrawingParams, LineStyle } from '../types';
import { TradingLevels } from '../../../../../types/agent.types';

/**
 * Draw a horizontal price level line
 */
const drawPriceLevel = (
  ctx: CanvasRenderingContext2D,
  y: number,
  width: number,
  label: string,
  style: LineStyle & { labelColor?: string; labelBackground?: string }
) => {
  ctx.save();

  // Set line style
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.width;

  if (style.dash) {
    ctx.setLineDash(style.dash);
  }

  if (style.shadowBlur) {
    ctx.shadowBlur = style.shadowBlur;
    ctx.shadowColor = style.shadowColor || style.color;
  }

  // Draw line
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(width, y);
  ctx.stroke();

  // Draw label
  if (label) {
    ctx.shadowBlur = 0;
    const labelPadding = 8;
    const labelHeight = 20;
    const textMetrics = ctx.measureText(label);
    const labelWidth = textMetrics.width + labelPadding * 2;

    // Label background
    if (style.labelBackground) {
      ctx.fillStyle = style.labelBackground;
      ctx.fillRect(width - labelWidth - 10, y - labelHeight / 2, labelWidth, labelHeight);
    }

    // Label text
    ctx.fillStyle = style.labelColor || style.color;
    ctx.font = '12px Inter, system-ui, -apple-system';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, width - 10 - labelPadding, y);
  }

  ctx.restore();
};

/**
 * Draw a zone between two price levels
 */
const drawPriceZone = (
  ctx: CanvasRenderingContext2D,
  y1: number,
  y2: number,
  width: number,
  fillColor: string,
  label?: string
) => {
  ctx.save();
  ctx.fillStyle = fillColor;
  ctx.fillRect(0, Math.min(y1, y2), width, Math.abs(y2 - y1));

  // Draw zone label if provided
  if (label) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = '11px Inter, system-ui, -apple-system';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, width / 2, (y1 + y2) / 2);
  }

  ctx.restore();
};

/**
 * Draw an arrow indicator
 */
const drawArrow = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  direction: 'up' | 'down',
  color: string,
  size: number = 12
) => {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  // Shadow for visibility
  ctx.shadowBlur = 20;
  ctx.shadowColor = color;

  ctx.beginPath();
  if (direction === 'up') {
    ctx.moveTo(x, y - size);
    ctx.lineTo(x - size / 2, y);
    ctx.lineTo(x + size / 2, y);
  } else {
    ctx.moveTo(x, y + size);
    ctx.lineTo(x - size / 2, y);
    ctx.lineTo(x + size / 2, y);
  }
  ctx.closePath();
  ctx.fill();

  ctx.restore();
};

/**
 * Hook for agent level drawing functionality
 */
export const useAgentLevelDrawing = () => {
  const animationFrameRef = useRef<number>();
  const pulsePhaseRef = useRef<number>(0);

  /**
   * Draw agent-generated trading levels
   */
  const drawAgentLevels = useCallback((params: AgentLevelDrawingParams) => {
    const { ctx, levels, yScale, theme, chartWidth } = params;

    if (!ctx || !levels) return;

    // Clear previous animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const isDark = theme.palette.mode === 'dark';

    // Define level styles
    const levelStyles = {
      entry: {
        color: isDark ? '#FFD700' : '#FFA000',
        width: 2,
        dash: [5, 5],
        shadowBlur: isDark ? 10 : 0,
        shadowColor: '#FFD700',
        labelColor: '#000000',
        labelBackground: '#FFD700',
      },
      stopLoss: {
        color: isDark ? '#FF4444' : '#D32F2F',
        width: 2,
        shadowBlur: isDark ? 8 : 0,
        shadowColor: '#FF4444',
        labelColor: '#FFFFFF',
        labelBackground: '#FF4444',
      },
      takeProfit: {
        color: isDark ? '#00FF88' : '#4CAF50',
        width: 2,
        shadowBlur: isDark ? 8 : 0,
        shadowColor: '#00FF88',
        labelColor: '#000000',
        labelBackground: '#00FF88',
      },
      support: {
        color: isDark ? '#00BFFF' : '#2196F3',
        width: 1,
        dash: [10, 5],
        shadowBlur: 0,
      },
      resistance: {
        color: isDark ? '#FF69B4' : '#E91E63',
        width: 1,
        dash: [10, 5],
        shadowBlur: 0,
      },
    };

    // Animate function for pulsing effects
    const animate = () => {
      pulsePhaseRef.current += 0.05;
      const pulseOpacity = 0.3 + 0.3 * Math.sin(pulsePhaseRef.current);

      // Clear canvas
      ctx.clearRect(0, 0, chartWidth, params.yScale(0));

      // Draw zones first (behind lines)
      if (levels.entry && levels.stop_loss) {
        const entryY = yScale(levels.entry);
        const stopLossY = yScale(levels.stop_loss);
        drawPriceZone(
          ctx,
          entryY,
          stopLossY,
          chartWidth,
          isDark ? `rgba(255, 68, 68, ${pulseOpacity * 0.5})` : 'rgba(255, 68, 68, 0.1)',
          'Risk Zone'
        );
      }

      if (levels.entry && levels.take_profits && levels.take_profits.length > 0) {
        const entryY = yScale(levels.entry);
        const tp1Y = yScale(levels.take_profits[0]);
        drawPriceZone(
          ctx,
          entryY,
          tp1Y,
          chartWidth,
          isDark ? `rgba(0, 255, 136, ${pulseOpacity * 0.5})` : 'rgba(0, 255, 136, 0.1)',
          'Profit Zone'
        );
      }

      // Draw support/resistance levels
      if (levels.support_levels) {
        levels.support_levels.forEach((level, index) => {
          const y = yScale(level);
          drawPriceLevel(
            ctx,
            y,
            chartWidth,
            `S${index + 1}: $${level.toFixed(2)}`,
            levelStyles.support
          );
        });
      }

      if (levels.resistance_levels) {
        levels.resistance_levels.forEach((level, index) => {
          const y = yScale(level);
          drawPriceLevel(
            ctx,
            y,
            chartWidth,
            `R${index + 1}: $${level.toFixed(2)}`,
            levelStyles.resistance
          );
        });
      }

      // Draw entry level with pulsing effect
      if (levels.entry) {
        const y = yScale(levels.entry);
        const entryStyle = {
          ...levelStyles.entry,
          shadowBlur: isDark ? 10 + 10 * pulseOpacity : 0,
        };
        drawPriceLevel(
          ctx,
          y,
          chartWidth,
          `Entry: $${levels.entry.toFixed(2)}`,
          entryStyle
        );

        // Draw entry arrow
        drawArrow(
          ctx,
          chartWidth - 50,
          y,
          levels.signal_type === 'buy' ? 'up' : 'down',
          levelStyles.entry.color
        );
      }

      // Draw stop loss level
      if (levels.stop_loss) {
        const y = yScale(levels.stop_loss);
        drawPriceLevel(
          ctx,
          y,
          chartWidth,
          `SL: $${levels.stop_loss.toFixed(2)}`,
          levelStyles.stopLoss
        );
      }

      // Draw take profit levels
      if (levels.take_profits) {
        levels.take_profits.forEach((tp, index) => {
          const y = yScale(tp);
          drawPriceLevel(
            ctx,
            y,
            chartWidth,
            `TP${index + 1}: $${tp.toFixed(2)}`,
            levelStyles.takeProfit
          );
        });
      }

      // Continue animation
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    // Start animation
    animate();
  }, []);

  /**
   * Draw AI signal arrows
   */
  const drawSignalArrows = useCallback((
    ctx: CanvasRenderingContext2D,
    signals: any[],
    xScale: (time: number, index: number) => number,
    yScale: (price: number) => number,
    data: any[],
    theme: any
  ) => {
    if (!ctx || signals.length === 0) return;

    const isDark = theme.palette.mode === 'dark';

    signals.forEach(signal => {
      // Find the corresponding data point
      const dataIndex = data.findIndex(d => Math.abs(d.time - signal.time) < 60000); // Within 1 minute
      if (dataIndex === -1) return;

      const x = xScale(signal.time, dataIndex);
      const y = yScale(signal.price);

      // Draw confidence ring
      const confidenceRadius = 20 + signal.confidence * 10;
      ctx.save();
      ctx.strokeStyle = signal.type === 'buy' ? '#00FF88' : '#FF4444';
      ctx.lineWidth = 2;
      ctx.globalAlpha = signal.confidence;
      ctx.beginPath();
      ctx.arc(x, y, confidenceRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Draw arrow
      drawArrow(
        ctx,
        x,
        y + (signal.type === 'buy' ? 20 : -20),
        signal.type === 'buy' ? 'up' : 'down',
        signal.type === 'buy' ? '#00FF88' : '#FF4444'
      );

      // Draw source label if available
      if (signal.source) {
        ctx.save();
        ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)';
        ctx.font = '10px Inter, system-ui, -apple-system';
        ctx.textAlign = 'center';
        ctx.fillText(signal.source, x, y + (signal.type === 'buy' ? 40 : -40));
        ctx.restore();
      }
    });
  }, []);

  /**
   * Cleanup animation on unmount
   */
  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  return {
    drawAgentLevels,
    drawSignalArrows,
    cleanup,
  };
};
