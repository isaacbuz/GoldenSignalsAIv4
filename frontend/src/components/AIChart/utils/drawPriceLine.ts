/**
 * Draw Price Line Utilities
 *
 * Functions for drawing horizontal price lines with labels.
 * Used for current price, stop loss, take profit levels, etc.
 */

interface PriceLineOptions {
  price: number;
  color: string;
  lineWidth?: number;
  lineStyle?: 'solid' | 'dashed' | 'dotted';
  labelBackground?: string;
  labelTextColor?: string;
  labelPosition?: 'left' | 'right';
  showLabel?: boolean;
  opacity?: number;
  animated?: boolean;
}

/**
 * Draw a horizontal price line with optional label
 */
export function drawPriceLine(
  ctx: CanvasRenderingContext2D,
  price: number,
  yScale: (price: number) => number,
  width: number,
  options: PriceLineOptions
) {
  const {
    color,
    lineWidth = 1,
    lineStyle = 'solid',
    labelBackground,
    labelTextColor,
    labelPosition = 'right',
    showLabel = true,
    opacity = 1,
    animated = false,
  } = options;

  const y = Math.floor(yScale(price)) + 0.5; // Pixel-perfect positioning

  ctx.save();
  ctx.globalAlpha = opacity;

  // Set line style
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;

  // Set dash pattern
  switch (lineStyle) {
    case 'dashed':
      ctx.setLineDash([8, 4]);
      break;
    case 'dotted':
      ctx.setLineDash([2, 2]);
      break;
    default:
      ctx.setLineDash([]);
  }

  // Draw line
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(width, y);
  ctx.stroke();

  // Draw label
  if (showLabel) {
    drawPriceLabel(ctx, price, y, width, {
      background: labelBackground || color,
      textColor: labelTextColor || '#fff',
      position: labelPosition,
      animated,
    });
  }

  ctx.restore();
}

/**
 * Draw current price line with pulsing animation
 */
export function drawCurrentPriceLine(
  ctx: CanvasRenderingContext2D,
  currentPrice: number,
  yScale: (price: number) => number,
  width: number,
  theme: any
) {
  const isDark = theme.palette.mode === 'dark';
  const pulseOpacity = 0.6 + Math.sin(Date.now() * 0.003) * 0.3;

  drawPriceLine(ctx, currentPrice, yScale, width, {
    price: currentPrice,
    color: isDark ? '#FFD700' : '#1976D2',
    lineWidth: 2,
    lineStyle: 'solid',
    labelBackground: isDark ? '#FFD700' : '#1976D2',
    labelTextColor: isDark ? '#000' : '#fff',
    showLabel: true,
    opacity: pulseOpacity,
    animated: true,
  });
}

/**
 * Draw price label
 */
function drawPriceLabel(
  ctx: CanvasRenderingContext2D,
  price: number,
  y: number,
  width: number,
  options: {
    background: string;
    textColor: string;
    position: 'left' | 'right';
    animated: boolean;
  }
) {
  const { background, textColor, position, animated } = options;
  const label = `$${price.toFixed(2)}`;
  const padding = 6;
  const borderRadius = 4;

  ctx.font = 'bold 12px Inter, system-ui, sans-serif';
  const metrics = ctx.measureText(label);
  const labelWidth = metrics.width + padding * 2;
  const labelHeight = 22;

  // Calculate position
  const x = position === 'right' ? width - labelWidth - 10 : 10;

  // Animated offset
  const animOffset = animated ? Math.sin(Date.now() * 0.002) * 2 : 0;

  ctx.save();

  // Shadow for depth
  if (animated) {
    ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 2;
  }

  // Background
  ctx.fillStyle = background;
  ctx.beginPath();
  ctx.roundRect(
    x + animOffset,
    y - labelHeight / 2,
    labelWidth,
    labelHeight,
    borderRadius
  );
  ctx.fill();

  // Reset shadow
  ctx.shadowColor = 'transparent';

  // Text
  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(
    label,
    x + labelWidth / 2 + animOffset,
    y
  );

  ctx.restore();
}

/**
 * Draw multiple price levels (e.g., support/resistance)
 */
export function drawPriceLevels(
  ctx: CanvasRenderingContext2D,
  levels: Array<{ price: number; type: 'support' | 'resistance' | 'target' | 'stop' }>,
  yScale: (price: number) => number,
  width: number,
  theme: any
) {
  const isDark = theme.palette.mode === 'dark';

  const levelStyles = {
    support: {
      color: theme.palette.success.main,
      lineStyle: 'dashed' as const,
      opacity: 0.6,
    },
    resistance: {
      color: theme.palette.error.main,
      lineStyle: 'dashed' as const,
      opacity: 0.6,
    },
    target: {
      color: theme.palette.success.main,
      lineStyle: 'dotted' as const,
      opacity: 0.4,
    },
    stop: {
      color: theme.palette.error.main,
      lineStyle: 'dotted' as const,
      opacity: 0.4,
    },
  };

  levels.forEach(({ price, type }) => {
    const style = levelStyles[type];
    drawPriceLine(ctx, price, yScale, width, {
      price,
      color: style.color,
      lineWidth: 1,
      lineStyle: style.lineStyle,
      showLabel: true,
      opacity: style.opacity,
      labelPosition: 'left',
    });
  });
}
