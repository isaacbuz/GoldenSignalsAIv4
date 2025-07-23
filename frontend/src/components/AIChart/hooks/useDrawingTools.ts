/**
 * useDrawingTools Hook
 *
 * Manages drawing tools for the chart including:
 * - Trendlines
 * - Horizontal lines (support/resistance)
 * - Fibonacci retracements
 * - Rectangle zones
 *
 * Handles mouse interactions, drawing state, and persistence.
 */

import { useState, useCallback, useRef } from 'react';
import { CoordinateSystem } from '../utils/coordinateSystem';

export type DrawingTool = 'none' | 'trendline' | 'horizontal' | 'fibonacci' | 'rectangle';

export interface Drawing {
  id: string;
  type: DrawingTool;
  points: Array<{ x: number; y: number; time: number; price: number }>;
  color: string;
  lineWidth: number;
  style?: 'solid' | 'dashed' | 'dotted';
  label?: string;
  locked?: boolean;
}

interface UseDrawingToolsParams {
  onDrawingComplete?: (drawing: Drawing) => void;
  maxDrawings?: number;
}

interface UseDrawingToolsResult {
  selectedTool: DrawingTool;
  setSelectedTool: (tool: DrawingTool) => void;
  drawings: Drawing[];
  currentDrawing: Drawing | null;
  isDrawing: boolean;
  handleMouseDown: (e: React.MouseEvent, coordinates: CoordinateSystem) => void;
  handleMouseMove: (e: React.MouseEvent, coordinates: CoordinateSystem) => void;
  handleMouseUp: () => void;
  deleteDrawing: (id: string) => void;
  clearAllDrawings: () => void;
  updateDrawing: (id: string, updates: Partial<Drawing>) => void;
  drawAllDrawings: (ctx: CanvasRenderingContext2D, coordinates: CoordinateSystem) => void;
}

export const useDrawingTools = ({
  onDrawingComplete,
  maxDrawings = 50,
}: UseDrawingToolsParams = {}): UseDrawingToolsResult => {
  const [selectedTool, setSelectedTool] = useState<DrawingTool>('none');
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [currentDrawing, setCurrentDrawing] = useState<Drawing | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const drawingIdCounter = useRef(0);

  /**
   * Generate unique drawing ID
   */
  const generateId = useCallback(() => {
    drawingIdCounter.current += 1;
    return `drawing_${Date.now()}_${drawingIdCounter.current}`;
  }, []);

  /**
   * Get default color for drawing tool
   */
  const getDefaultColor = useCallback((tool: DrawingTool): string => {
    switch (tool) {
      case 'trendline':
        return '#2196F3'; // Blue
      case 'horizontal':
        return '#4CAF50'; // Green for support/resistance
      case 'fibonacci':
        return '#FF9800'; // Orange
      case 'rectangle':
        return '#9C27B0'; // Purple
      default:
        return '#2196F3';
    }
  }, []);

  /**
   * Start drawing on mouse down
   */
  const handleMouseDown = useCallback((
    e: React.MouseEvent,
    coordinates: CoordinateSystem
  ) => {
    if (selectedTool === 'none' || !coordinates) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const time = coordinates.xToTime(x);
    const price = coordinates.yToPrice(y);

    const newDrawing: Drawing = {
      id: generateId(),
      type: selectedTool,
      points: [{ x, y, time, price }],
      color: getDefaultColor(selectedTool),
      lineWidth: 2,
      style: 'solid',
    };

    setCurrentDrawing(newDrawing);
    setIsDrawing(true);
  }, [selectedTool, generateId, getDefaultColor]);

  /**
   * Update drawing on mouse move
   */
  const handleMouseMove = useCallback((
    e: React.MouseEvent,
    coordinates: CoordinateSystem
  ) => {
    if (!isDrawing || !currentDrawing || !coordinates) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const time = coordinates.xToTime(x);
    const price = coordinates.yToPrice(y);

    const updatedPoints = [...currentDrawing.points];

    // For most tools, we update the second point
    if (currentDrawing.type === 'trendline' ||
        currentDrawing.type === 'rectangle' ||
        currentDrawing.type === 'fibonacci') {
      if (updatedPoints.length === 1) {
        updatedPoints.push({ x, y, time, price });
      } else {
        updatedPoints[1] = { x, y, time, price };
      }
    } else if (currentDrawing.type === 'horizontal') {
      // For horizontal lines, keep Y constant
      updatedPoints[0].y = y;
      updatedPoints[0].price = price;
    }

    setCurrentDrawing({
      ...currentDrawing,
      points: updatedPoints,
    });
  }, [isDrawing, currentDrawing]);

  /**
   * Complete drawing on mouse up
   */
  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !currentDrawing) return;

    // Validate drawing has required points
    const isValid = currentDrawing.type === 'horizontal' ||
                   currentDrawing.points.length >= 2;

    if (isValid) {
      // Add to drawings list
      setDrawings(prev => {
        const newDrawings = [...prev, currentDrawing];
        // Limit number of drawings
        if (newDrawings.length > maxDrawings) {
          newDrawings.shift();
        }
        return newDrawings;
      });

      // Callback
      onDrawingComplete?.(currentDrawing);
    }

    // Reset state
    setCurrentDrawing(null);
    setIsDrawing(false);
  }, [isDrawing, currentDrawing, maxDrawings, onDrawingComplete]);

  /**
   * Delete a specific drawing
   */
  const deleteDrawing = useCallback((id: string) => {
    setDrawings(prev => prev.filter(d => d.id !== id));
  }, []);

  /**
   * Clear all drawings
   */
  const clearAllDrawings = useCallback(() => {
    setDrawings([]);
    setCurrentDrawing(null);
    setIsDrawing(false);
  }, []);

  /**
   * Update drawing properties
   */
  const updateDrawing = useCallback((id: string, updates: Partial<Drawing>) => {
    setDrawings(prev => prev.map(d =>
      d.id === id ? { ...d, ...updates } : d
    ));
  }, []);

  /**
   * Draw a single drawing on canvas
   */
  const drawDrawing = useCallback((
    ctx: CanvasRenderingContext2D,
    drawing: Drawing,
    coordinates: CoordinateSystem
  ) => {
    ctx.save();
    ctx.strokeStyle = drawing.color;
    ctx.lineWidth = drawing.lineWidth;

    // Set line style
    switch (drawing.style) {
      case 'dashed':
        ctx.setLineDash([8, 4]);
        break;
      case 'dotted':
        ctx.setLineDash([2, 2]);
        break;
      default:
        ctx.setLineDash([]);
    }

    switch (drawing.type) {
      case 'trendline':
        if (drawing.points.length >= 2) {
          const x1 = coordinates.timeToX(drawing.points[0].time);
          const y1 = coordinates.priceToY(drawing.points[0].price);
          const x2 = coordinates.timeToX(drawing.points[1].time);
          const y2 = coordinates.priceToY(drawing.points[1].price);

          // Extend line to chart edges
          const slope = (y2 - y1) / (x2 - x1);
          const extendedX1 = 0;
          const extendedY1 = y1 - slope * (x1 - extendedX1);
          const extendedX2 = ctx.canvas.width;
          const extendedY2 = y1 + slope * (extendedX2 - x1);

          ctx.beginPath();
          ctx.moveTo(extendedX1, extendedY1);
          ctx.lineTo(extendedX2, extendedY2);
          ctx.stroke();
        }
        break;

      case 'horizontal':
        if (drawing.points.length >= 1) {
          const y = coordinates.priceToY(drawing.points[0].price);

          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(ctx.canvas.width, y);
          ctx.stroke();

          // Draw price label
          if (drawing.label !== false) {
            ctx.fillStyle = drawing.color;
            ctx.font = '12px Inter, system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(
              `$${drawing.points[0].price.toFixed(2)}`,
              ctx.canvas.width - 10,
              y
            );
          }
        }
        break;

      case 'fibonacci':
        if (drawing.points.length >= 2) {
          const x1 = coordinates.timeToX(drawing.points[0].time);
          const y1 = coordinates.priceToY(drawing.points[0].price);
          const x2 = coordinates.timeToX(drawing.points[1].time);
          const y2 = coordinates.priceToY(drawing.points[1].price);

          const priceDiff = drawing.points[1].price - drawing.points[0].price;
          const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];

          ctx.font = '11px Inter, system-ui, sans-serif';
          ctx.textAlign = 'left';

          levels.forEach(level => {
            const price = drawing.points[0].price + priceDiff * level;
            const y = coordinates.priceToY(price);

            // Draw line
            ctx.strokeStyle = drawing.color;
            ctx.globalAlpha = level === 0 || level === 1 ? 1 : 0.7;
            ctx.beginPath();
            ctx.moveTo(Math.min(x1, x2), y);
            ctx.lineTo(Math.max(x1, x2), y);
            ctx.stroke();

            // Draw label
            ctx.fillStyle = drawing.color;
            ctx.globalAlpha = 1;
            ctx.textBaseline = 'middle';
            ctx.fillText(
              `${(level * 100).toFixed(1)}% - $${price.toFixed(2)}`,
              Math.max(x1, x2) + 10,
              y
            );
          });
        }
        break;

      case 'rectangle':
        if (drawing.points.length >= 2) {
          const x1 = coordinates.timeToX(drawing.points[0].time);
          const y1 = coordinates.priceToY(drawing.points[0].price);
          const x2 = coordinates.timeToX(drawing.points[1].time);
          const y2 = coordinates.priceToY(drawing.points[1].price);

          const x = Math.min(x1, x2);
          const y = Math.min(y1, y2);
          const width = Math.abs(x2 - x1);
          const height = Math.abs(y2 - y1);

          // Fill with transparency
          ctx.fillStyle = drawing.color + '20';
          ctx.fillRect(x, y, width, height);

          // Draw border
          ctx.strokeRect(x, y, width, height);
        }
        break;
    }

    ctx.restore();
  }, []);

  /**
   * Draw all drawings on canvas
   */
  const drawAllDrawings = useCallback((
    ctx: CanvasRenderingContext2D,
    coordinates: CoordinateSystem
  ) => {
    // Draw completed drawings
    drawings.forEach(drawing => {
      drawDrawing(ctx, drawing, coordinates);
    });

    // Draw current drawing if in progress
    if (currentDrawing && isDrawing) {
      drawDrawing(ctx, currentDrawing, coordinates);
    }
  }, [drawings, currentDrawing, isDrawing, drawDrawing]);

  return {
    selectedTool,
    setSelectedTool,
    drawings,
    currentDrawing,
    isDrawing,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    deleteDrawing,
    clearAllDrawings,
    updateDrawing,
    drawAllDrawings,
  };
};
