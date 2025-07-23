/**
 * ChartCanvas Component
 *
 * Core canvas rendering component for the AI Trading Chart.
 * Handles all drawing operations including candlesticks, indicators, and AI overlays.
 *
 * This component is responsible for:
 * - Canvas setup and management
 * - Coordinate system transformations
 * - Drawing delegation to specialized hooks
 * - Performance optimization through selective rendering
 * - Mouse/touch interaction handling
 *
 * The component uses multiple canvas layers:
 * - Main canvas: Price data, volume, indicators
 * - AI canvas: Predictions, signals, trading levels
 * - Interaction canvas: Crosshair, tooltips (future)
 */

import React, { useRef, useEffect, useCallback, memo, useImperativeHandle, forwardRef } from 'react';
import { Box, useTheme } from '@mui/material';
import { useChartContext } from '../../context/ChartContext';
import { useCandlestickDrawing } from '../../hooks/useCandlestickDrawing';
import { useIndicatorDrawing } from '../../hooks/useIndicatorDrawing';
import { useAgentLevelDrawing } from '../../hooks/useAgentLevelDrawing';
import { useSignalClustering } from '../../hooks/useSignalClustering';
import { useChartScales } from './hooks/useChartScales';
import { ChartDataPoint } from './types';
import { LayerManager, LayerManagerHandle } from '../../core/LayerManager';
import { CoordinateSystem } from '../../utils/coordinateSystem';
import { drawCrispLine, drawTextWithBackground } from '../../utils/canvasSetup';
import { useChartCrosshair } from '../../hooks/useChartCrosshair';
import { drawCurrentPriceLine, drawPriceLevels } from '../../utils/drawPriceLine';
import { Drawing } from '../../hooks/useDrawingTools';

interface ChartCanvasProps {
  /**
   * Width of the canvas in pixels
   */
  width?: number;

  /**
   * Height of the canvas in pixels or percentage string
   */
  height?: number | string;

  /**
   * Chart data points to render
   */
  data: ChartDataPoint[];

  /**
   * Selected indicators to display
   */
  indicators?: string[];

  /**
   * Trading signals to display
   */
  signals?: any[];

  /**
   * Agent-generated trading levels
   */
  agentLevels?: any;

  /**
   * Current price for real-time display
   */
  currentPrice?: number;

  /**
   * MUI theme object
   */
  theme?: any;

  /**
   * Chart timeframe
   */
  timeframe?: string;

  /**
   * Chart type: candlestick, line, or area
   */
  chartType?: 'candle' | 'line' | 'area';

  /**
   * Whether to show volume bars
   */
  showVolume?: boolean;

  /**
   * Whether to show interactive features
   */
  interactive?: boolean;

  /**
   * Custom padding around the chart
   */
  padding?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };

  /**
   * Zoom state from parent
   */
  zoomState?: {
    scale: number;
    offsetX: number;
    yRangeScale: number;
  };

  /**
   * Visible data subset (for zoomed view)
   */
  visibleData?: ChartDataPoint[];

  /**
   * Drawing tools data
   */
  drawings?: Drawing[];

  /**
   * Drawing tools functions
   */
  onDrawingMouseDown?: (e: React.MouseEvent, coordinates: CoordinateSystem) => void;
  onDrawingMouseMove?: (e: React.MouseEvent, coordinates: CoordinateSystem) => void;
  onDrawingMouseUp?: () => void;
  drawAllDrawings?: (ctx: CanvasRenderingContext2D, coordinates: CoordinateSystem) => void;

  /**
   * Callback when a candle is clicked
   */
  onCandleClick?: (candle: ChartDataPoint) => void;

  /**
   * Callback when time range changes
   */
  onTimeRangeChange?: (startIndex: number, endIndex: number) => void;
}

/**
 * Default padding values for the chart
 */
const DEFAULT_PADDING = {
  top: 20,
  right: 60,
  bottom: 60,
  left: 70,
};

export interface ChartCanvasHandle {
  /**
   * Force a full redraw of the chart
   */
  redraw(): void;

  /**
   * Update only the last candle (for real-time updates)
   */
  updateLastCandle(candle: ChartDataPoint): void;

  /**
   * Get coordinate system for external use
   */
  getCoordinateSystem(): CoordinateSystem | null;

  /**
   * Update AI prediction overlay
   */
  updateAIPrediction(prediction: any): void;
}

/**
 * ChartCanvas component with memoization for performance
 */
export const ChartCanvas = memo(forwardRef<ChartCanvasHandle, ChartCanvasProps>(({
  width,
  height,
  data,
  indicators,
  signals,
  agentLevels,
  currentPrice,
  theme: propTheme,
  timeframe: propTimeframe,
  interactive = true,
  padding = DEFAULT_PADDING,
  visibleData,
  zoomState,
  drawings,
  onDrawingMouseDown,
  onDrawingMouseMove,
  onDrawingMouseUp,
  drawAllDrawings,
}, ref) => {
  const muiTheme = useTheme();
  const theme = propTheme || muiTheme;

  // Use timeframe from props or default
  const timeframe = propTimeframe || '1h';

  // Layer manager ref
  const layerManagerRef = useRef<LayerManagerHandle>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Coordinate system ref
  const coordinateSystemRef = useRef<CoordinateSystem | null>(null);

  // Use visible data if provided (for zoom), otherwise use all data
  const chartData = visibleData || data || [];
  const selectedIndicators = indicators || [];
  const showVolume = selectedIndicators.includes('volume');
  const showGrid = true;

  // Calculate chart dimensions
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Initialize scale calculations
  const { xScale, yScale, priceRange, reverseScales } = useChartScales({
    data: chartData,
    width: chartWidth,
    height: chartHeight,
    padding,
  });

  // Initialize crosshair
  const {
    crosshairPosition,
    tooltipData,
    handleMouseMove: handleCrosshairMove,
    handleMouseLeave: handleCrosshairLeave,
    drawCrosshair,
  } = useChartCrosshair({
    data: chartData,
    containerRef,
    reverseScales,
    enabled: interactive,
  });

  // Update coordinate system when scales change
  useEffect(() => {
    if (chartData.length > 0) {
      const timeRange = {
        start: chartData[0].time,
        end: chartData[chartData.length - 1].time,
      };

      coordinateSystemRef.current = new CoordinateSystem(
        width,
        height,
        padding,
        priceRange,
        timeRange
      );

      // Set index mapping for evenly spaced candles
      coordinateSystemRef.current.setIndexMapping(chartData);
    }
  }, [chartData, width, height, padding, priceRange]);

  // Initialize drawing hooks
  const { drawCandlesticks, updateLastCandle } = useCandlestickDrawing();
  const { drawIndicators, drawVolume } = useIndicatorDrawing();
  const { drawAgentLevels, drawSignalArrows, cleanup } = useAgentLevelDrawing();
  const { drawClusteredSignals } = useSignalClustering();

  /**
   * Clear all canvas layers
   */
  const clearCanvases = useCallback(() => {
    if (layerManagerRef.current) {
      layerManagerRef.current.clearAllLayers();
    }
  }, []);

  /**
   * Draw grid lines using coordinate system
   */
  const drawGrid = useCallback(() => {
    if (!layerManagerRef.current || !coordinateSystemRef.current) return;
    layerManagerRef.current.renderOnLayer('background', (ctx) => {
      if (!showGrid) return;

      const coords = coordinateSystemRef.current!;
      const viewport = coords.getViewportBounds();

      ctx.save();
      ctx.strokeStyle = theme.palette.mode === 'dark'
        ? 'rgba(255, 215, 0, 0.05)' // Golden tint in dark mode
        : 'rgba(0, 0, 0, 0.05)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);

      // Draw price grid lines
      const priceIntervals = coords.getNicePriceIntervals(8);
      priceIntervals.forEach(price => {
        const y = coords.priceToY(price);
        drawCrispLine(ctx, viewport.x, y, viewport.x + viewport.width, y);
      });

      // Draw time grid lines
      const timeIntervals = coords.getNiceTimeIntervals(12);
      timeIntervals.forEach(time => {
        const x = coords.timeToX(time);
        drawCrispLine(ctx, x, viewport.y, x, viewport.y + viewport.height);
      });

      ctx.restore();

      // Draw axes labels
      ctx.save();
      ctx.fillStyle = theme.palette.text.secondary;
      ctx.font = '12px Inter, system-ui, -apple-system, sans-serif';

      // Price labels
      priceIntervals.forEach(price => {
        const y = coords.priceToY(price);
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(price.toFixed(2), viewport.x - 10, y);
      });

      // Time labels
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      timeIntervals.forEach(time => {
        const x = coords.timeToX(time);
        const date = new Date(time);
        const label = formatTimeLabel(date, timeframe);
        ctx.fillText(label, x, viewport.y + viewport.height + 10);
      });

      ctx.restore();
    });
  }, [showGrid, theme, timeframe]);

  /**
   * Draw watermark
   */
  const drawWatermark = useCallback((ctx: CanvasRenderingContext2D) => {
    // Save context state
    ctx.save();

    // Set watermark styling
    ctx.font = 'bold 8rem -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Create golden gradient
    const gradient = ctx.createLinearGradient(0, height * 0.3, 0, height * 0.7);
    gradient.addColorStop(0, 'rgba(255, 215, 0, 0.15)'); // Golden with low opacity
    gradient.addColorStop(0.5, 'rgba(255, 228, 0, 0.20)');
    gradient.addColorStop(1, 'rgba(255, 200, 0, 0.15)');

    ctx.fillStyle = gradient;

    // Add shadow glow effect
    ctx.shadowColor = 'rgba(255, 215, 0, 0.3)';
    ctx.shadowBlur = 30;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Draw symbol watermark
    ctx.fillText(symbol.toUpperCase(), width / 2, height / 2 - 50);

    // Draw "GoldenSignalsAI" text below
    ctx.font = 'normal 2rem -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif';
    ctx.letterSpacing = '0.2em';
    ctx.shadowBlur = 20;
    ctx.fillText('GoldenSignalsAI', width / 2, height / 2 + 50);

    // Restore context state
    ctx.restore();
  }, [symbol, width, height]);

  /**
   * Main drawing function that orchestrates all rendering
   */
  const draw = useCallback(() => {
    if (!layerManagerRef.current || chartData.length === 0 || !coordinateSystemRef.current) return;

    const layerManager = layerManagerRef.current;
    const coords = coordinateSystemRef.current;

    // Clear all layers
    clearCanvases();

    // Draw grid and axes
    drawGrid();

    // Draw main chart on main layer
    layerManager.renderOnLayer('main', (ctx) => {
      const drawingParams = {
        ctx,
        data: chartData,
        coordinates: coords,
        theme,
      };

      // Draw candlesticks
      drawCandlesticks(drawingParams);

      // Draw watermark
      drawWatermark(ctx);
    });

    // Draw volume on main layer
    if (showVolume) {
      layerManager.renderOnLayer('main', (ctx) => {
        drawVolume({
          ctx,
          data: chartData,
          coordinates: coords,
          theme,
          volumeHeight: 80,
        });
      });
    }

    // Draw indicators on indicators layer
    if (selectedIndicators.length > 0) {
      layerManager.renderOnLayer('indicators', (ctx) => {
        drawIndicators({
          ctx,
          data: chartData,
          coordinates: coords,
          theme,
          indicators: selectedIndicators,
        });
      });
    }

    // Draw agent levels on agents layer
    if (agentLevels) {
      layerManager.renderOnLayer('agents', (ctx) => {
        drawAgentLevels({
          ctx,
          levels: agentLevels,
          coordinates: coords,
          theme,
        });
      });
    }

    // Draw current price line on indicators layer
    if (currentPrice && currentPrice > 0) {
      layerManager.renderOnLayer('indicators', (ctx) => {
        drawCurrentPriceLine(ctx, currentPrice, yScale, width, theme);
      });
    }

    // Draw agent signals on agents layer with clustering
    if (signals && signals.length > 0) {
      layerManager.renderOnLayer('agents', (ctx) => {
        // Use clustering for cleaner visualization
        drawClusteredSignals({
          ctx,
          signals,
          coordinates: coords,
          theme,
          minClusterSize: 1,
          showLabels: true,
          animationPhase: Date.now() * 0.001,
        });
      });
    }

    // Draw crosshair on interaction layer
    if (interactive && crosshairPosition) {
      layerManager.renderOnLayer('interactions', (ctx) => {
        drawCrosshair(ctx, width, height, theme);
      });
    }

    // Draw drawing tools on interactions layer
    if (drawAllDrawings && coordinateSystemRef.current) {
      layerManager.renderOnLayer('interactions', (ctx) => {
        drawAllDrawings(ctx, coordinateSystemRef.current!);
      });
    }
  }, [
    chartData,
    clearCanvases,
    drawGrid,
    drawWatermark,
    showVolume,
    selectedIndicators,
    agentLevels,
    signals,
    theme,
    drawCandlesticks,
    drawVolume,
    drawIndicators,
    drawAgentLevels,
    drawSignalArrows,
    drawClusteredSignals,
    interactive,
    crosshairPosition,
    drawCrosshair,
    width,
    height,
    drawAllDrawings,
  ]);

  /**
   * Redraw when dependencies change
   */
  useEffect(() => {
    draw();
  }, [draw]);

  /**
   * Animation loop for current price line
   */
  useEffect(() => {
    if (!currentPrice || currentPrice <= 0) return;

    let animationId: number;
    const animate = () => {
      draw();
      animationId = requestAnimationFrame(animate);
    };

    // Start animation
    animationId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [currentPrice, draw]);

  /**
   * Expose methods to parent components
   */
  useImperativeHandle(ref, () => ({
    redraw: () => {
      draw();
    },

    updateLastCandle: (candle: ChartDataPoint) => {
      if (!layerManagerRef.current || !coordinateSystemRef.current) return;

      // Update only the last candle on main layer
      layerManagerRef.current.renderOnLayer('main', (ctx) => {
        const coords = coordinateSystemRef.current!;
        updateLastCandle({
          ctx,
          candle,
          coordinates: coords,
          theme,
        });
      });
    },

    getCoordinateSystem: () => {
      return coordinateSystemRef.current;
    },
  }), [draw, updateLastCandle, theme]);

  // Combined mouse event handlers
  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    handleCrosshairMove(event);
    // Handle drawing tool mouse move
    if (onDrawingMouseMove && coordinateSystemRef.current) {
      onDrawingMouseMove(event, coordinateSystemRef.current);
    }
    // Trigger redraw for crosshair and drawings
    draw();
  }, [handleCrosshairMove, draw, onDrawingMouseMove]);

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    // Handle drawing tool mouse down
    if (onDrawingMouseDown && coordinateSystemRef.current) {
      onDrawingMouseDown(event, coordinateSystemRef.current);
    }
  }, [onDrawingMouseDown]);

  const handleMouseUp = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    // Handle drawing tool mouse up
    if (onDrawingMouseUp) {
      onDrawingMouseUp();
    }
  }, [onDrawingMouseUp]);

  const handleMouseLeave = useCallback(() => {
    handleCrosshairLeave();
    // Trigger redraw to clear crosshair
    draw();
  }, [handleCrosshairLeave, draw]);

  const handleWheel = useCallback((event: React.WheelEvent<HTMLCanvasElement>) => {
    // Zoom is handled in parent component
    event.preventDefault();
  }, []);

  return (
    <Box ref={containerRef} sx={{ position: 'relative', width, height }}>
      <LayerManager
        ref={layerManagerRef}
        width={width}
        height={height}
        backgroundColor={theme.palette.background.default}
        onMouseMove={interactive ? handleMouseMove : undefined}
        onMouseDown={interactive ? handleMouseDown : undefined}
        onMouseUp={interactive ? handleMouseUp : undefined}
        onMouseLeave={interactive ? handleMouseLeave : undefined}
        onWheel={interactive ? handleWheel : undefined}
      />
    </Box>
  );
}));

ChartCanvas.displayName = 'ChartCanvas';

export type { ChartCanvasHandle };

/**
 * Format time label based on timeframe
 */
function formatTimeLabel(date: Date, timeframe: string): string {
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const month = date.toLocaleDateString('en-US', { month: 'short' });
  const day = date.getDate();
  const year = date.getFullYear();

  switch (timeframe) {
    case '1m':
    case '5m':
    case '15m':
      return `${hours}:${minutes}`;
    case '1h':
    case '4h':
      return `${month} ${day}, ${hours}:00`;
    case '1d':
    case '1w':
    case '1M':
      return `${month} ${day}, ${year}`;
    default:
      return `${month} ${day}`;
  }
}
