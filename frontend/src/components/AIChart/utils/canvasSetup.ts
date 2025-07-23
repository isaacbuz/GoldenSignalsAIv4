/**
 * Canvas Setup Utilities
 *
 * Handles proper canvas initialization for high-DPI displays
 * and provides utilities for canvas management.
 */

export interface CanvasConfig {
  width: number;
  height: number;
  backgroundColor?: string;
  willReadFrequently?: boolean;
}

/**
 * Setup canvas for high-DPI displays
 * Ensures crisp rendering on retina screens
 */
export function setupCanvas(
  canvas: HTMLCanvasElement,
  config: CanvasConfig
): CanvasRenderingContext2D {
  const { width, height, backgroundColor, willReadFrequently = false } = config;

  // Get device pixel ratio
  const dpr = window.devicePixelRatio || 1;

  // Set display size (CSS pixels)
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  // Set actual size in memory (scaled for DPR)
  canvas.width = width * dpr;
  canvas.height = height * dpr;

  // Get context with optimization hints
  const ctx = canvas.getContext('2d', {
    alpha: !backgroundColor,
    desynchronized: true,
    willReadFrequently
  })!;

  // Scale context to match device pixel ratio
  ctx.scale(dpr, dpr);

  // Set default styles
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  // Clear with background color if specified
  if (backgroundColor) {
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
  }

  return ctx;
}

/**
 * Clear canvas efficiently
 */
export function clearCanvas(
  ctx: CanvasRenderingContext2D,
  backgroundColor?: string
): void {
  const { width, height } = ctx.canvas;
  const dpr = window.devicePixelRatio || 1;

  // Clear the entire canvas
  ctx.clearRect(0, 0, width / dpr, height / dpr);

  // Fill with background color if specified
  if (backgroundColor) {
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width / dpr, height / dpr);
  }
}

/**
 * Save canvas state with common defaults
 */
export function saveCanvasState(ctx: CanvasRenderingContext2D): void {
  ctx.save();

  // Reset common properties to defaults
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.shadowBlur = 0;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;
  ctx.filter = 'none';
  ctx.lineCap = 'butt';
  ctx.lineJoin = 'miter';
  ctx.lineDashOffset = 0;
  ctx.setLineDash([]);
}

/**
 * Create offscreen canvas for caching
 */
export function createOffscreenCanvas(
  width: number,
  height: number
): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;

  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);

  return { canvas, ctx };
}

/**
 * Measure text with caching
 */
const textMeasureCache = new Map<string, TextMetrics>();

export function measureText(
  ctx: CanvasRenderingContext2D,
  text: string,
  font: string
): TextMetrics {
  const cacheKey = `${font}:${text}`;

  if (textMeasureCache.has(cacheKey)) {
    return textMeasureCache.get(cacheKey)!;
  }

  ctx.save();
  ctx.font = font;
  const metrics = ctx.measureText(text);
  ctx.restore();

  textMeasureCache.set(cacheKey, metrics);

  // Clear cache if it gets too large
  if (textMeasureCache.size > 1000) {
    const entries = Array.from(textMeasureCache.entries());
    entries.slice(0, 500).forEach(([key]) => textMeasureCache.delete(key));
  }

  return metrics;
}

/**
 * Draw crisp lines (avoid blurry lines on canvas)
 */
export function drawCrispLine(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  lineWidth: number = 1
): void {
  // Offset by 0.5 for odd line widths to avoid anti-aliasing
  const offset = lineWidth % 2 === 1 ? 0.5 : 0;

  ctx.beginPath();

  if (x1 === x2) {
    // Vertical line
    ctx.moveTo(Math.floor(x1) + offset, y1);
    ctx.lineTo(Math.floor(x2) + offset, y2);
  } else if (y1 === y2) {
    // Horizontal line
    ctx.moveTo(x1, Math.floor(y1) + offset);
    ctx.lineTo(x2, Math.floor(y2) + offset);
  } else {
    // Diagonal line
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }

  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

/**
 * Draw text with background
 */
export function drawTextWithBackground(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  options: {
    font?: string;
    textColor?: string;
    backgroundColor?: string;
    padding?: number;
    borderRadius?: number;
    align?: 'left' | 'center' | 'right';
    baseline?: 'top' | 'middle' | 'bottom';
  } = {}
): void {
  const {
    font = '12px Inter, system-ui, -apple-system, sans-serif',
    textColor = '#FFFFFF',
    backgroundColor = 'rgba(0, 0, 0, 0.7)',
    padding = 4,
    borderRadius = 4,
    align = 'left',
    baseline = 'middle'
  } = options;

  ctx.save();

  // Set font and measure text
  ctx.font = font;
  const metrics = ctx.measureText(text);
  const textWidth = metrics.width;
  const textHeight = parseInt(font); // Approximate height

  // Calculate background position
  let bgX = x - padding;
  let bgY = y - textHeight / 2 - padding;

  // Adjust for alignment
  if (align === 'center') {
    bgX -= textWidth / 2;
  } else if (align === 'right') {
    bgX -= textWidth;
  }

  // Adjust for baseline
  if (baseline === 'top') {
    bgY += textHeight / 2;
  } else if (baseline === 'bottom') {
    bgY -= textHeight / 2;
  }

  // Draw background
  if (backgroundColor) {
    ctx.fillStyle = backgroundColor;

    if (borderRadius > 0) {
      // Rounded rectangle
      ctx.beginPath();
      ctx.roundRect(
        bgX,
        bgY,
        textWidth + padding * 2,
        textHeight + padding * 2,
        borderRadius
      );
      ctx.fill();
    } else {
      // Regular rectangle
      ctx.fillRect(
        bgX,
        bgY,
        textWidth + padding * 2,
        textHeight + padding * 2
      );
    }
  }

  // Draw text
  ctx.fillStyle = textColor;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(text, x, y);

  ctx.restore();
}

/**
 * Request animation frame with fallback
 */
export const requestFrame = (callback: FrameRequestCallback): number => {
  return window.requestAnimationFrame(callback);
};

export const cancelFrame = (id: number): void => {
  window.cancelAnimationFrame(id);
};

/**
 * Throttle canvas operations
 */
export function throttleCanvasOperation(
  operation: Function,
  delay: number
): (...args: any[]) => void {
  let lastCall = 0;
  let timeout: NodeJS.Timeout | null = null;

  return (...args: any[]) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall;

    if (timeSinceLastCall >= delay) {
      lastCall = now;
      operation(...args);
    } else {
      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(() => {
        lastCall = Date.now();
        operation(...args);
      }, delay - timeSinceLastCall);
    }
  };
}
