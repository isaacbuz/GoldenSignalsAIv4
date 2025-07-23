/**
 * Layer Manager Component
 *
 * Manages multiple canvas layers for efficient rendering and compositing.
 * Each layer handles specific chart elements to optimize performance.
 */

import React, { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { Box } from '@mui/material';
import { setupCanvas, clearCanvas } from '../utils/canvasSetup';
import logger from '../../../services/logger';


export interface Layer {
  name: string;
  canvas: HTMLCanvasElement | null;
  ctx: CanvasRenderingContext2D | null;
  zIndex: number;
  interactive: boolean;
}

export interface LayerManagerProps {
  width: number;
  height: number;
  backgroundColor?: string;
  onMouseMove?: (event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseDown?: (event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp?: (event: React.MouseEvent<HTMLDivElement>) => void;
  onMouseLeave?: (event: React.MouseEvent<HTMLDivElement>) => void;
  onWheel?: (event: React.WheelEvent<HTMLDivElement>) => void;
}

export interface LayerManagerHandle {
  getLayer(name: string): Layer | undefined;
  clearLayer(name: string): void;
  clearAllLayers(): void;
  renderOnLayer(name: string, renderFn: (ctx: CanvasRenderingContext2D) => void): void;
  getCompositeImageData(): string;
  resize(width: number, height: number): void;
}

const LAYER_DEFINITIONS = [
  { name: 'background', zIndex: 0, interactive: false },  // Grid, axes, labels
  { name: 'main', zIndex: 1, interactive: false },        // Candlesticks, volume
  { name: 'indicators', zIndex: 2, interactive: false },  // Technical indicators
  { name: 'agents', zIndex: 3, interactive: false },      // Agent signals, levels
  { name: 'predictions', zIndex: 4, interactive: false }, // AI predictions
  { name: 'annotations', zIndex: 5, interactive: false }, // User annotations
  { name: 'interactions', zIndex: 6, interactive: true }, // Crosshair, tooltips
];

export const LayerManager = forwardRef<LayerManagerHandle, LayerManagerProps>(({
  width,
  height,
  backgroundColor,
  onMouseMove,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  onWheel
}, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const layersRef = useRef<Map<string, Layer>>(new Map());
  const compositeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize layers
  useEffect(() => {
    if (!containerRef.current) return;

    // Clear existing layers
    layersRef.current.clear();

    // Create canvas elements for each layer
    LAYER_DEFINITIONS.forEach(layerDef => {
      const canvas = document.createElement('canvas');
      canvas.style.position = 'absolute';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.pointerEvents = layerDef.interactive ? 'auto' : 'none';
      canvas.style.zIndex = layerDef.zIndex.toString();

      // Setup canvas with high-DPI support
      const ctx = setupCanvas(canvas, {
        width,
        height,
        backgroundColor: layerDef.name === 'background' ? backgroundColor : undefined
      });

      // Store layer reference
      layersRef.current.set(layerDef.name, {
        name: layerDef.name,
        canvas,
        ctx,
        zIndex: layerDef.zIndex,
        interactive: layerDef.interactive
      });

      // Add to container
      containerRef.current!.appendChild(canvas);
    });

    // Create composite canvas for exports
    compositeCanvasRef.current = document.createElement('canvas');

    return () => {
      // Cleanup
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
      layersRef.current.clear();
    };
  }, []); // Only run on mount

  // Handle resize
  useEffect(() => {
    layersRef.current.forEach(layer => {
      if (layer.canvas && layer.ctx) {
        const wasBackground = layer.name === 'background';
        layer.ctx = setupCanvas(layer.canvas, {
          width,
          height,
          backgroundColor: wasBackground ? backgroundColor : undefined
        });
      }
    });

    // Resize composite canvas
    if (compositeCanvasRef.current) {
      setupCanvas(compositeCanvasRef.current, { width, height });
    }
  }, [width, height, backgroundColor]);

  // Expose methods through ref
  useImperativeHandle(ref, () => ({
    getLayer(name: string): Layer | undefined {
      return layersRef.current.get(name);
    },

    clearLayer(name: string): void {
      const layer = layersRef.current.get(name);
      if (layer?.ctx) {
        clearCanvas(layer.ctx, layer.name === 'background' ? backgroundColor : undefined);
      }
    },

    clearAllLayers(): void {
      layersRef.current.forEach(layer => {
        if (layer.ctx) {
          clearCanvas(layer.ctx, layer.name === 'background' ? backgroundColor : undefined);
        }
      });
    },

    renderOnLayer(name: string, renderFn: (ctx: CanvasRenderingContext2D) => void): void {
      const layer = layersRef.current.get(name);
      if (layer?.ctx) {
        layer.ctx.save();
        try {
          renderFn(layer.ctx);
        } catch (error) {
          logger.error(`Error rendering on layer ${name}:`, error);
        } finally {
          layer.ctx.restore();
        }
      }
    },

    getCompositeImageData(): string {
      if (!compositeCanvasRef.current) return '';

      const compositeCtx = compositeCanvasRef.current.getContext('2d')!;
      clearCanvas(compositeCtx);

      // Composite all layers in order
      const sortedLayers = Array.from(layersRef.current.values())
        .sort((a, b) => a.zIndex - b.zIndex);

      sortedLayers.forEach(layer => {
        if (layer.canvas) {
          compositeCtx.drawImage(layer.canvas, 0, 0, width, height);
        }
      });

      return compositeCanvasRef.current.toDataURL('image/png');
    },

    resize(newWidth: number, newHeight: number): void {
      layersRef.current.forEach(layer => {
        if (layer.canvas && layer.ctx) {
          const wasBackground = layer.name === 'background';
          layer.ctx = setupCanvas(layer.canvas, {
            width: newWidth,
            height: newHeight,
            backgroundColor: wasBackground ? backgroundColor : undefined
          });
        }
      });
    }
  }), [backgroundColor]);

  return (
    <Box
      ref={containerRef}
      sx={{
        position: 'relative',
        width,
        height,
        overflow: 'hidden',
        cursor: 'crosshair'
      }}
      onMouseMove={onMouseMove}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseLeave}
      onWheel={onWheel}
    />
  );
});

LayerManager.displayName = 'LayerManager';

/**
 * Hook to use layer manager in child components
 */
export function useLayer(layerManager: React.RefObject<LayerManagerHandle>, layerName: string) {
  const [layer, setLayer] = React.useState<Layer | undefined>();

  React.useEffect(() => {
    if (layerManager.current) {
      setLayer(layerManager.current.getLayer(layerName));
    }
  }, [layerManager, layerName]);

  const clear = React.useCallback(() => {
    if (layerManager.current) {
      layerManager.current.clearLayer(layerName);
    }
  }, [layerManager, layerName]);

  const render = React.useCallback((renderFn: (ctx: CanvasRenderingContext2D) => void) => {
    if (layerManager.current) {
      layerManager.current.renderOnLayer(layerName, renderFn);
    }
  }, [layerManager, layerName]);

  return { layer, clear, render };
}
