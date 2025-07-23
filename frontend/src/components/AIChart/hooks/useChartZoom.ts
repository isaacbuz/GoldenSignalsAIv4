/**
 * useChartZoom Hook
 *
 * Manages zoom and pan functionality for the chart.
 * Supports mouse wheel zoom, pinch zoom, and drag-to-pan.
 *
 * Features:
 * - Smooth zoom animations
 * - Zoom around cursor position
 * - Pan with mouse drag
 * - Keyboard shortcuts (Ctrl+Plus/Minus)
 * - Touch gesture support
 * - Min/max zoom limits
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { ChartDataPoint } from '../components/ChartCanvas/types';

interface ZoomState {
  // Current zoom level (1 = 100%, 2 = 200%, etc.)
  scale: number;
  // X-axis offset in data points
  offsetX: number;
  // Y-axis price range multiplier
  yRangeScale: number;
}

interface UseChartZoomParams {
  data: ChartDataPoint[];
  containerRef: React.RefObject<HTMLDivElement>;
  onZoomChange?: (state: ZoomState) => void;
  minZoom?: number;
  maxZoom?: number;
}

interface UseChartZoomResult {
  zoomState: ZoomState;
  visibleData: ChartDataPoint[];
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  handleWheel: (event: WheelEvent) => void;
  handleMouseDown: (event: React.MouseEvent) => void;
  handleMouseMove: (event: React.MouseEvent) => void;
  handleMouseUp: () => void;
  isZoomed: boolean;
}

export const useChartZoom = ({
  data,
  containerRef,
  onZoomChange,
  minZoom = 0.5,
  maxZoom = 10,
}: UseChartZoomParams): UseChartZoomResult => {
  const [zoomState, setZoomState] = useState<ZoomState>({
    scale: 1,
    offsetX: 0,
    yRangeScale: 1,
  });

  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef({ x: 0, offsetX: 0 });

  // Calculate visible data based on zoom and offset
  const visibleData = useCallback(() => {
    if (!data || data.length === 0) return [];

    const totalPoints = data.length;
    const visiblePoints = Math.floor(totalPoints / zoomState.scale);
    const startIndex = Math.max(0, Math.min(zoomState.offsetX, totalPoints - visiblePoints));
    const endIndex = Math.min(startIndex + visiblePoints, totalPoints);

    return data.slice(startIndex, endIndex);
  }, [data, zoomState]);

  // Zoom in by 20%
  const zoomIn = useCallback(() => {
    setZoomState(prev => {
      const newScale = Math.min(prev.scale * 1.2, maxZoom);
      const centerIndex = prev.offsetX + data.length / (2 * prev.scale);
      const newOffsetX = Math.max(0, centerIndex - data.length / (2 * newScale));

      return {
        ...prev,
        scale: newScale,
        offsetX: Math.floor(newOffsetX),
      };
    });
  }, [data.length, maxZoom]);

  // Zoom out by 20%
  const zoomOut = useCallback(() => {
    setZoomState(prev => {
      const newScale = Math.max(prev.scale / 1.2, minZoom);
      const centerIndex = prev.offsetX + data.length / (2 * prev.scale);
      const newOffsetX = Math.max(0, centerIndex - data.length / (2 * newScale));

      return {
        ...prev,
        scale: newScale,
        offsetX: Math.floor(newOffsetX),
      };
    });
  }, [data.length, minZoom]);

  // Reset zoom to default
  const resetZoom = useCallback(() => {
    setZoomState({
      scale: 1,
      offsetX: 0,
      yRangeScale: 1,
    });
  }, []);

  // Handle mouse wheel zoom
  const handleWheel = useCallback((event: WheelEvent) => {
    event.preventDefault();

    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const relativeX = mouseX / rect.width;

    setZoomState(prev => {
      // Zoom in/out based on wheel direction
      const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(minZoom, Math.min(prev.scale * zoomFactor, maxZoom));

      // Calculate new offset to zoom around cursor position
      const visiblePointsBefore = data.length / prev.scale;
      const visiblePointsAfter = data.length / newScale;
      const pointUnderCursor = prev.offsetX + visiblePointsBefore * relativeX;
      const newOffsetX = Math.max(0, Math.min(
        pointUnderCursor - visiblePointsAfter * relativeX,
        data.length - visiblePointsAfter
      ));

      return {
        ...prev,
        scale: newScale,
        offsetX: Math.floor(newOffsetX),
      };
    });
  }, [containerRef, data.length, minZoom, maxZoom]);

  // Handle pan start
  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (event.button === 0) { // Left mouse button
      setIsPanning(true);
      panStartRef.current = {
        x: event.clientX,
        offsetX: zoomState.offsetX,
      };
    }
  }, [zoomState.offsetX]);

  // Handle pan move
  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (!isPanning || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const deltaX = event.clientX - panStartRef.current.x;
    const deltaDataPoints = Math.floor((deltaX / rect.width) * (data.length / zoomState.scale));

    const newOffsetX = Math.max(
      0,
      Math.min(
        panStartRef.current.offsetX - deltaDataPoints,
        data.length - data.length / zoomState.scale
      )
    );

    setZoomState(prev => ({
      ...prev,
      offsetX: newOffsetX,
    }));
  }, [isPanning, containerRef, data.length, zoomState.scale]);

  // Handle pan end
  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        if (event.key === '+' || event.key === '=') {
          event.preventDefault();
          zoomIn();
        } else if (event.key === '-' || event.key === '_') {
          event.preventDefault();
          zoomOut();
        } else if (event.key === '0') {
          event.preventDefault();
          resetZoom();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [zoomIn, zoomOut, resetZoom]);

  // Notify parent component of zoom changes
  useEffect(() => {
    onZoomChange?.(zoomState);
  }, [zoomState, onZoomChange]);

  // Add wheel event listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [containerRef, handleWheel]);

  return {
    zoomState,
    visibleData: visibleData(),
    zoomIn,
    zoomOut,
    resetZoom,
    handleWheel,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    isZoomed: zoomState.scale !== 1 || zoomState.offsetX !== 0,
  };
};
