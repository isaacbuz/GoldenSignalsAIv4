/**
 * useAgentLevelDrawing Hook
 *
 * Handles the drawing of agent-generated trading levels and signals
 * using the new coordinate system.
 */

import { useCallback, useRef } from 'react';
import { CoordinateSystem } from '../utils/coordinateSystem';
import { AgentSignal, TradingLevel } from '../components/ChartCanvas/types';
import { drawTextWithBackground } from '../utils/canvasSetup';

interface DrawAgentLevelsParams {
  ctx: CanvasRenderingContext2D;
  levels: TradingLevel[];
  coordinates: CoordinateSystem;
  theme: any;
}

interface DrawSignalArrowsParams {
  ctx: CanvasRenderingContext2D;
  signals: AgentSignal[];
  coordinates: CoordinateSystem;
  theme: any;
}

export const useAgentLevelDrawing = () => {
  // Animation state
  const animationFrameRef = useRef<number | null>(null);
  const pulsePhaseRef = useRef(0);

  /**
   * Draw a single trading level
   */
  const drawLevel = useCallback((
    ctx: CanvasRenderingContext2D,
    level: TradingLevel,
    coordinates: CoordinateSystem,
    theme: any
  ) => {
    const viewport = coordinates.getViewportBounds();
    const y = coordinates.priceToY(level.price);

    // Skip if outside viewport
    if (y < viewport.y || y > viewport.y + viewport.height) return;

    // Set colors based on level type
    let color: string;
    let alpha: number;

    switch (level.type) {
      case 'support':
        color = theme.palette.mode === 'dark' ? '#00FF88' : '#4CAF50';
        alpha = 0.3;
        break;
      case 'resistance':
        color = theme.palette.mode === 'dark' ? '#FF4444' : '#F44336';
        alpha = 0.3;
        break;
      case 'entry':
        color = theme.palette.mode === 'dark' ? '#FFD700' : '#FFC107';
        alpha = 0.4;
        break;
      case 'stop':
        color = theme.palette.mode === 'dark' ? '#FF6B6B' : '#E91E63';
        alpha = 0.3;
        break;
      case 'target':
        color = theme.palette.mode === 'dark' ? '#00BFFF' : '#2196F3';
        alpha = 0.3;
        break;
      default:
        color = theme.palette.text.secondary;
        alpha = 0.2;
    }

    ctx.save();

    // Draw the level line with glow effect
    ctx.strokeStyle = color;
    ctx.lineWidth = level.strength ? Math.max(1, level.strength * 3) : 2;
    ctx.setLineDash([10, 5]);
    ctx.globalAlpha = alpha;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;

    ctx.beginPath();
    ctx.moveTo(viewport.x, y);
    ctx.lineTo(viewport.x + viewport.width, y);
    ctx.stroke();

    // Draw label if provided
    if (level.label) {
      ctx.globalAlpha = 1;
      drawTextWithBackground(ctx, level.label, viewport.x + viewport.width - 10, y, {
        font: '12px Inter, system-ui, -apple-system, sans-serif',
        textColor: theme.palette.text.primary,
        backgroundColor: color + '40',
        align: 'right',
        baseline: 'middle',
        padding: 4,
        borderRadius: 4,
      });
    }

    // Draw price label
    ctx.globalAlpha = 0.8;
    drawTextWithBackground(ctx, level.price.toFixed(2), viewport.x + 10, y, {
      font: '11px Inter, system-ui, -apple-system, sans-serif',
      textColor: color,
      backgroundColor: theme.palette.background.paper + 'CC',
      align: 'left',
      baseline: 'middle',
      padding: 2,
      borderRadius: 2,
    });

    ctx.restore();
  }, []);

  /**
   * Draw signal arrow with animation
   */
  const drawSignalArrow = useCallback((
    ctx: CanvasRenderingContext2D,
    signal: AgentSignal,
    coordinates: CoordinateSystem,
    theme: any,
    pulsePhase: number
  ) => {
    const x = coordinates.timeToX(signal.time);
    const y = coordinates.priceToY(signal.price);
    const viewport = coordinates.getViewportBounds();

    // Skip if outside viewport
    if (x < viewport.x || x > viewport.x + viewport.width) return;

    const isBuy = signal.type === 'buy';
    const color = isBuy
      ? (theme.palette.mode === 'dark' ? '#00FF88' : '#4CAF50')
      : (theme.palette.mode === 'dark' ? '#FF4444' : '#F44336');

    ctx.save();

    // Draw confidence ring with pulse animation
    const pulseScale = 1 + Math.sin(pulsePhase) * 0.2;
    ctx.globalAlpha = signal.confidence * 0.3;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 20 * pulseScale;
    ctx.shadowColor = color;

    ctx.beginPath();
    ctx.arc(x, y, 20 * pulseScale, 0, Math.PI * 2);
    ctx.stroke();

    // Draw arrow
    ctx.globalAlpha = signal.confidence;
    ctx.fillStyle = color;
    ctx.shadowBlur = 12;

    const arrowSize = 12;
    const arrowOffset = isBuy ? 25 : -25;

    ctx.beginPath();
    if (isBuy) {
      // Up arrow
      ctx.moveTo(x, y + arrowOffset - arrowSize);
      ctx.lineTo(x - arrowSize * 0.7, y + arrowOffset);
      ctx.lineTo(x - arrowSize * 0.3, y + arrowOffset);
      ctx.lineTo(x - arrowSize * 0.3, y + arrowOffset + arrowSize);
      ctx.lineTo(x + arrowSize * 0.3, y + arrowOffset + arrowSize);
      ctx.lineTo(x + arrowSize * 0.3, y + arrowOffset);
      ctx.lineTo(x + arrowSize * 0.7, y + arrowOffset);
    } else {
      // Down arrow
      ctx.moveTo(x, y + arrowOffset + arrowSize);
      ctx.lineTo(x - arrowSize * 0.7, y + arrowOffset);
      ctx.lineTo(x - arrowSize * 0.3, y + arrowOffset);
      ctx.lineTo(x - arrowSize * 0.3, y + arrowOffset - arrowSize);
      ctx.lineTo(x + arrowSize * 0.3, y + arrowOffset - arrowSize);
      ctx.lineTo(x + arrowSize * 0.3, y + arrowOffset);
      ctx.lineTo(x + arrowSize * 0.7, y + arrowOffset);
    }
    ctx.closePath();
    ctx.fill();

    // Draw agent name and confidence
    if (signal.agentName) {
      ctx.globalAlpha = 1;
      const label = `${signal.agentName} (${(signal.confidence * 100).toFixed(0)}%)`;
      drawTextWithBackground(ctx, label, x, y + (isBuy ? -40 : 40), {
        font: '11px Inter, system-ui, -apple-system, sans-serif',
        textColor: theme.palette.text.primary,
        backgroundColor: theme.palette.background.paper + 'E6',
        align: 'center',
        baseline: isBuy ? 'bottom' : 'top',
      });
    }

    ctx.restore();
  }, []);

  /**
   * Draw all agent levels
   */
  const drawAgentLevels = useCallback((params: DrawAgentLevelsParams) => {
    const { ctx, levels, coordinates, theme } = params;

    if (!levels || levels.length === 0) return;

    // Sort levels by importance (entry first, then targets, then stops)
    const sortedLevels = [...levels].sort((a, b) => {
      const order = { entry: 0, target: 1, stop: 2, support: 3, resistance: 4 };
      return (order[a.type] || 5) - (order[b.type] || 5);
    });

    sortedLevels.forEach(level => {
      drawLevel(ctx, level, coordinates, theme);
    });
  }, [drawLevel]);

  /**
   * Draw all signal arrows with animation
   */
  const drawSignalArrows = useCallback((params: DrawSignalArrowsParams) => {
    const { ctx, signals, coordinates, theme } = params;

    if (!signals || signals.length === 0) return;

    // Update pulse phase
    pulsePhaseRef.current += 0.05;

    // Draw signals sorted by time (oldest first)
    const sortedSignals = [...signals].sort((a, b) => a.time - b.time);

    sortedSignals.forEach(signal => {
      drawSignalArrow(ctx, signal, coordinates, theme, pulsePhaseRef.current);
    });
  }, [drawSignalArrow]);

  /**
   * Cleanup animation frame
   */
  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  return {
    drawAgentLevels,
    drawSignalArrows,
    cleanup,
  };
};
