import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, useTheme, alpha, Typography, Chip, IconButton } from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as ShowChartIcon,
  Timeline as TimelineIcon,
  BarChart as BarChartIcon,
} from '@mui/icons-material';

interface ChartDataPoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Signal {
  time: number;
  type: 'buy' | 'sell';
  price: number;
  confidence: number;
}

interface EnhancedCustomChartProps {
  data: ChartDataPoint[];
  signals?: Signal[];
  width?: number;
  height?: number;
  showGrid?: boolean;
  showWatermark?: boolean;
  symbol?: string;
  indicators?: string[];
  theme?: 'dark' | 'light' | 'auto';
  onHover?: (data: ChartDataPoint | null) => void;
}

const pulse = keyframes`
  0%, 100% {
    opacity: 0.8;
    transform: scale(1);
  }
  50% {
    opacity: 1;
    transform: scale(1.1);
  }
`;

const glow = keyframes`
  0%, 100% {
    filter: drop-shadow(0 0 8px currentColor);
  }
  50% {
    filter: drop-shadow(0 0 16px currentColor) drop-shadow(0 0 24px currentColor);
  }
`;

const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  background: theme.palette.mode === 'dark'
    ? `linear-gradient(135deg, #0A0E1A 0%, ${alpha('#1a237e', 0.3)} 100%)`
    : `linear-gradient(135deg, #FAFAFA 0%, ${alpha('#e3f2fd', 0.5)} 100%)`,
  borderRadius: theme.spacing(2),
  overflow: 'hidden',
  boxShadow: theme.palette.mode === 'dark'
    ? `0 8px 32px ${alpha('#000', 0.4)}`
    : `0 8px 32px ${alpha('#000', 0.1)}`,
}));

const Canvas = styled('canvas')(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  cursor: 'crosshair',
}));

const GlowCanvas = styled('canvas')(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  pointerEvents: 'none',
  filter: 'blur(20px)',
  opacity: 0.4,
}));

const Watermark = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  pointerEvents: 'none',
  userSelect: 'none',
  textAlign: 'center',
  opacity: 0.03,
  '& .symbol': {
    fontSize: '12rem',
    fontWeight: 900,
    lineHeight: 1,
    letterSpacing: '-0.02em',
    background: `linear-gradient(180deg, ${theme.palette.text.primary} 0%, transparent 100%)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  '& .brand': {
    fontSize: '3rem',
    fontWeight: 800,
    letterSpacing: '0.2em',
    marginTop: '-2rem',
    background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
}));

const InfoPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  left: theme.spacing(2),
  padding: theme.spacing(2, 3),
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  borderRadius: theme.spacing(1.5),
  backdropFilter: 'blur(20px)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  boxShadow: `0 4px 24px ${alpha(theme.palette.common.black, 0.1)}`,
  minWidth: 280,
  '& .header': {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
    marginBottom: theme.spacing(1),
  },
  '& .price': {
    fontSize: '2rem',
    fontWeight: 800,
    fontFamily: 'monospace',
    marginBottom: theme.spacing(0.5),
  },
  '& .details': {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: theme.spacing(1),
    fontSize: '0.875rem',
    fontFamily: 'monospace',
    opacity: 0.8,
  },
}));

const SignalMarker = styled(Box)<{ type: 'buy' | 'sell' }>(({ theme, type }) => ({
  position: 'absolute',
  width: 32,
  height: 32,
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: type === 'buy' ? theme.palette.success.main : theme.palette.error.main,
  color: theme.palette.common.white,
  fontSize: '1.2rem',
  fontWeight: 'bold',
  animation: `${pulse} 2s ease-in-out infinite`,
  cursor: 'pointer',
  zIndex: 10,
  '&::before': {
    content: '""',
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    backgroundColor: 'inherit',
    animation: `${glow} 2s ease-in-out infinite`,
  },
}));

const IndicatorButton = styled(IconButton)(({ theme }) => ({
  position: 'absolute',
  backgroundColor: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  '&:hover': {
    backgroundColor: theme.palette.background.paper,
  },
}));

export const EnhancedCustomChart: React.FC<EnhancedCustomChartProps> = ({
  data,
  signals = [],
  width,
  height,
  showGrid = true,
  showWatermark = true,
  symbol = 'AAPL',
  indicators = ['sma', 'volume'],
  theme: chartTheme = 'auto',
  onHover,
}) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glowCanvasRef = useRef<HTMLCanvasElement>(null);
  const gridCanvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: width || 800, height: height || 500 });
  const [hoveredData, setHoveredData] = useState<ChartDataPoint | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const animationRef = useRef<number>();
  const [animationProgress, setAnimationProgress] = useState(0);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: width || rect.width,
          height: height || rect.height,
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [width, height]);

  // Animate chart on mount
  useEffect(() => {
    let startTime: number;
    const duration = 1500; // 1.5 second animation

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);

      setAnimationProgress(easeOutExpo(progress));

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data]);

  // Easing function for smooth animation
  const easeOutExpo = (t: number): number => {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
  };

  // Draw gradient background
  const drawBackground = useCallback(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, dimensions.width, dimensions.height);

    if (theme.palette.mode === 'dark') {
      gradient.addColorStop(0, alpha('#1a237e', 0.05));
      gradient.addColorStop(1, alpha('#0a0e1a', 0.02));
    } else {
      gradient.addColorStop(0, alpha('#e3f2fd', 0.1));
      gradient.addColorStop(1, alpha('#ffffff', 0.05));
    }

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    if (showGrid) {
      ctx.strokeStyle = alpha(theme.palette.divider, 0.05);
      ctx.lineWidth = 1;

      // Vertical lines with gradient fade
      const xStep = dimensions.width / 12;
      for (let x = 0; x <= dimensions.width; x += xStep) {
        const gradient = ctx.createLinearGradient(x, 0, x, dimensions.height);
        gradient.addColorStop(0, alpha(theme.palette.divider, 0));
        gradient.addColorStop(0.1, alpha(theme.palette.divider, 0.05));
        gradient.addColorStop(0.9, alpha(theme.palette.divider, 0.05));
        gradient.addColorStop(1, alpha(theme.palette.divider, 0));

        ctx.strokeStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, dimensions.height);
        ctx.stroke();
      }

      // Horizontal lines with gradient fade
      const yStep = dimensions.height / 8;
      for (let y = 0; y <= dimensions.height; y += yStep) {
        const gradient = ctx.createLinearGradient(0, y, dimensions.width, y);
        gradient.addColorStop(0, alpha(theme.palette.divider, 0));
        gradient.addColorStop(0.1, alpha(theme.palette.divider, 0.05));
        gradient.addColorStop(0.9, alpha(theme.palette.divider, 0.05));
        gradient.addColorStop(1, alpha(theme.palette.divider, 0));

        ctx.strokeStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(dimensions.width, y);
        ctx.stroke();
      }
    }
  }, [dimensions, showGrid, theme]);

  // Draw chart with enhanced visuals
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    const glowCanvas = glowCanvasRef.current;
    if (!canvas || !glowCanvas || !data.length) return;

    const ctx = canvas.getContext('2d');
    const glowCtx = glowCanvas.getContext('2d');
    if (!ctx || !glowCtx) return;

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    glowCtx.clearRect(0, 0, dimensions.width, dimensions.height);

    const padding = { top: 60, right: 80, bottom: 60, left: 80 };
    const chartWidth = dimensions.width - padding.left - padding.right;
    const chartHeight = dimensions.height - padding.top - padding.bottom;

    // Calculate price range
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    data.forEach(d => {
      minPrice = Math.min(minPrice, d.low);
      maxPrice = Math.max(maxPrice, d.high);
    });
    const priceRange = maxPrice - minPrice;
    const pricePadding = priceRange * 0.1;
    minPrice -= pricePadding;
    maxPrice += pricePadding;

    // Scale functions
    const xScale = (index: number) => padding.left + (index / (data.length - 1)) * chartWidth;
    const yScale = (price: number) => padding.top + (1 - (price - minPrice) / (maxPrice - minPrice)) * chartHeight;

    // Draw area gradient fill
    const areaGradient = ctx.createLinearGradient(0, padding.top, 0, dimensions.height - padding.bottom);
    areaGradient.addColorStop(0, alpha(theme.palette.primary.main, 0.2));
    areaGradient.addColorStop(1, alpha(theme.palette.primary.main, 0));

    ctx.beginPath();
    data.forEach((d, i) => {
      const x = xScale(i);
      const y = yScale(d.close);
      const animatedY = dimensions.height - (dimensions.height - y) * animationProgress;

      if (i === 0) {
        ctx.moveTo(x, animatedY);
      } else {
        ctx.lineTo(x, animatedY);
      }
    });

    // Complete the area
    ctx.lineTo(xScale(data.length - 1), dimensions.height - padding.bottom);
    ctx.lineTo(xScale(0), dimensions.height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = areaGradient;
    ctx.fill();

    // Draw main price line with glow
    ctx.strokeStyle = theme.palette.primary.main;
    ctx.lineWidth = 3;
    ctx.shadowColor = theme.palette.primary.main;
    ctx.shadowBlur = 20;
    ctx.beginPath();

    data.forEach((d, i) => {
      const x = xScale(i);
      const y = yScale(d.close);
      const animatedY = dimensions.height - (dimensions.height - y) * animationProgress;

      if (i === 0) {
        ctx.moveTo(x, animatedY);
      } else {
        ctx.lineTo(x, animatedY);
      }
    });

    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw the same line on glow canvas for enhanced effect
    glowCtx.strokeStyle = theme.palette.primary.main;
    glowCtx.lineWidth = 6;
    glowCtx.globalAlpha = 0.6;
    glowCtx.beginPath();

    data.forEach((d, i) => {
      const x = xScale(i);
      const y = yScale(d.close);
      const animatedY = dimensions.height - (dimensions.height - y) * animationProgress;

      if (i === 0) {
        glowCtx.moveTo(x, animatedY);
      } else {
        glowCtx.lineTo(x, animatedY);
      }
    });

    glowCtx.stroke();

    // Draw candlesticks with enhanced style
    const candleWidth = Math.max(2, (chartWidth / data.length) * 0.6);

    data.forEach((d, i) => {
      const x = xScale(i);
      const yHigh = yScale(d.high);
      const yLow = yScale(d.low);
      const yOpen = yScale(d.open);
      const yClose = yScale(d.close);

      const progress = animationProgress;
      const animatedYHigh = dimensions.height - (dimensions.height - yHigh) * progress;
      const animatedYLow = dimensions.height - (dimensions.height - yLow) * progress;
      const animatedYOpen = dimensions.height - (dimensions.height - yOpen) * progress;
      const animatedYClose = dimensions.height - (dimensions.height - yClose) * progress;

      // Determine color
      const isGreen = d.close >= d.open;
      const color = isGreen ? theme.palette.success.main : theme.palette.error.main;

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1;

      // Draw wick with gradient
      const wickGradient = ctx.createLinearGradient(x, animatedYHigh, x, animatedYLow);
      wickGradient.addColorStop(0, alpha(color, 0.3));
      wickGradient.addColorStop(0.5, color);
      wickGradient.addColorStop(1, alpha(color, 0.3));

      ctx.strokeStyle = wickGradient;
      ctx.beginPath();
      ctx.moveTo(x, animatedYHigh);
      ctx.lineTo(x, animatedYLow);
      ctx.stroke();

      // Draw body with rounded corners
      const bodyTop = Math.min(animatedYOpen, animatedYClose);
      const bodyBottom = Math.max(animatedYOpen, animatedYClose);
      const bodyHeight = Math.max(1, bodyBottom - bodyTop);
      const radius = Math.min(4, candleWidth / 4);

      ctx.fillStyle = isGreen ? alpha(color, 0.8) : color;
      ctx.strokeStyle = color;

      // Rounded rectangle for candle body
      ctx.beginPath();
      ctx.moveTo(x - candleWidth / 2 + radius, bodyTop);
      ctx.lineTo(x + candleWidth / 2 - radius, bodyTop);
      ctx.quadraticCurveTo(x + candleWidth / 2, bodyTop, x + candleWidth / 2, bodyTop + radius);
      ctx.lineTo(x + candleWidth / 2, bodyBottom - radius);
      ctx.quadraticCurveTo(x + candleWidth / 2, bodyBottom, x + candleWidth / 2 - radius, bodyBottom);
      ctx.lineTo(x - candleWidth / 2 + radius, bodyBottom);
      ctx.quadraticCurveTo(x - candleWidth / 2, bodyBottom, x - candleWidth / 2, bodyBottom - radius);
      ctx.lineTo(x - candleWidth / 2, bodyTop + radius);
      ctx.quadraticCurveTo(x - candleWidth / 2, bodyTop, x - candleWidth / 2 + radius, bodyTop);
      ctx.closePath();

      if (isGreen) {
        ctx.stroke();
      } else {
        ctx.fill();
      }
    });

    // Draw moving averages with smooth curves
    if (indicators.includes('sma')) {
      const sma20 = calculateSMA(data, 20);
      const sma50 = calculateSMA(data, 50);

      // SMA 20
      ctx.strokeStyle = alpha(theme.palette.info.main, 0.8);
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      drawSmoothLine(ctx, sma20, xScale, yScale, animationProgress, dimensions.height);

      // SMA 50
      ctx.strokeStyle = alpha(theme.palette.warning.main, 0.8);
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      drawSmoothLine(ctx, sma50, xScale, yScale, animationProgress, dimensions.height);
      ctx.setLineDash([]);
    }

    // Draw volume bars with gradient
    if (indicators.includes('volume')) {
      const maxVolume = Math.max(...data.map(d => d.volume));
      const volumeHeight = chartHeight * 0.15;

      data.forEach((d, i) => {
        const x = xScale(i);
        const volumeBarHeight = (d.volume / maxVolume) * volumeHeight * animationProgress;
        const isGreen = d.close >= d.open;

        const volumeGradient = ctx.createLinearGradient(
          x,
          dimensions.height - padding.bottom - volumeBarHeight,
          x,
          dimensions.height - padding.bottom
        );

        const color = isGreen ? theme.palette.success.main : theme.palette.error.main;
        volumeGradient.addColorStop(0, alpha(color, 0.4));
        volumeGradient.addColorStop(1, alpha(color, 0.1));

        ctx.fillStyle = volumeGradient;

        // Rounded volume bars
        const barWidth = candleWidth * 0.8;
        const radius = Math.min(2, barWidth / 4);

        ctx.beginPath();
        ctx.moveTo(x - barWidth / 2 + radius, dimensions.height - padding.bottom - volumeBarHeight);
        ctx.lineTo(x + barWidth / 2 - radius, dimensions.height - padding.bottom - volumeBarHeight);
        ctx.quadraticCurveTo(
          x + barWidth / 2,
          dimensions.height - padding.bottom - volumeBarHeight,
          x + barWidth / 2,
          dimensions.height - padding.bottom - volumeBarHeight + radius
        );
        ctx.lineTo(x + barWidth / 2, dimensions.height - padding.bottom);
        ctx.lineTo(x - barWidth / 2, dimensions.height - padding.bottom);
        ctx.lineTo(x - barWidth / 2, dimensions.height - padding.bottom - volumeBarHeight + radius);
        ctx.quadraticCurveTo(
          x - barWidth / 2,
          dimensions.height - padding.bottom - volumeBarHeight,
          x - barWidth / 2 + radius,
          dimensions.height - padding.bottom - volumeBarHeight
        );
        ctx.closePath();
        ctx.fill();
      });
    }

    // Draw axis labels with modern style
    ctx.fillStyle = alpha(theme.palette.text.secondary, 0.6);
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    // Price labels with background
    const priceSteps = 6;
    for (let i = 0; i <= priceSteps; i++) {
      const price = minPrice + (i / priceSteps) * (maxPrice - minPrice);
      const y = yScale(price);

      // Background for label
      const label = `$${price.toFixed(2)}`;
      const metrics = ctx.measureText(label);

      ctx.fillStyle = alpha(theme.palette.background.paper, 0.8);
      ctx.fillRect(
        padding.left - metrics.width - 20,
        y - 10,
        metrics.width + 10,
        20
      );

      ctx.fillStyle = theme.palette.text.secondary;
      ctx.fillText(label, padding.left - 10, y);
    }

    // Enhanced crosshair
    if (mousePos.x > padding.left && mousePos.x < dimensions.width - padding.right &&
        mousePos.y > padding.top && mousePos.y < dimensions.height - padding.bottom) {

      // Crosshair lines with glow
      ctx.strokeStyle = alpha(theme.palette.primary.main, 0.3);
      ctx.lineWidth = 1;
      ctx.shadowColor = theme.palette.primary.main;
      ctx.shadowBlur = 10;

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(mousePos.x, padding.top);
      ctx.lineTo(mousePos.x, dimensions.height - padding.bottom);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(padding.left, mousePos.y);
      ctx.lineTo(dimensions.width - padding.right, mousePos.y);
      ctx.stroke();

      ctx.shadowBlur = 0;

      // Price label at cursor
      const price = minPrice + (1 - (mousePos.y - padding.top) / chartHeight) * (maxPrice - minPrice);
      const priceLabel = `$${price.toFixed(2)}`;

      ctx.fillStyle = theme.palette.primary.main;
      ctx.fillRect(dimensions.width - padding.right + 5, mousePos.y - 12, 70, 24);
      ctx.fillStyle = theme.palette.primary.contrastText;
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(priceLabel, dimensions.width - padding.right + 10, mousePos.y);
    }
  }, [data, dimensions, theme, indicators, animationProgress, mousePos]);

  // Helper function to draw smooth curves
  const drawSmoothLine = (
    ctx: CanvasRenderingContext2D,
    data: (number | null)[],
    xScale: (i: number) => number,
    yScale: (p: number) => number,
    progress: number,
    height: number
  ) => {
    ctx.beginPath();
    let started = false;

    for (let i = 0; i < data.length; i++) {
      if (data[i] !== null) {
        const x = xScale(i);
        const y = yScale(data[i]!);
        const animatedY = height - (height - y) * progress;

        if (!started) {
          ctx.moveTo(x, animatedY);
          started = true;
        } else if (i > 0 && data[i - 1] !== null) {
          // Use quadratic curves for smoothness
          const prevX = xScale(i - 1);
          const prevY = yScale(data[i - 1]!);
          const prevAnimatedY = height - (height - prevY) * progress;

          const cpX = (prevX + x) / 2;
          const cpY = (prevAnimatedY + animatedY) / 2;

          ctx.quadraticCurveTo(prevX, prevAnimatedY, cpX, cpY);
        }
      }
    }

    ctx.stroke();
  };

  // Calculate SMA
  const calculateSMA = (data: ChartDataPoint[], period: number): (number | null)[] => {
    const result: (number | null)[] = [];

    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((acc, d) => acc + d.close, 0);
        result.push(sum / period);
      }
    }

    return result;
  };

  // Handle mouse interactions
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePos({ x, y });

    // Find closest data point
    const padding = { left: 80, right: 80 };
    const chartWidth = dimensions.width - padding.left - padding.right;
    const relativeX = x - padding.left;
    const index = Math.round((relativeX / chartWidth) * (data.length - 1));

    if (index >= 0 && index < data.length) {
      const dataPoint = data[index];
      setHoveredData(dataPoint);
      onHover?.(dataPoint);
    }
  };

  const handleMouseLeave = () => {
    setMousePos({ x: 0, y: 0 });
    setHoveredData(null);
    onHover?.(null);
  };

  // Draw effects
  useEffect(() => {
    drawBackground();
  }, [drawBackground]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  // Format values
  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatVolume = (volume: number) => {
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
    return volume.toString();
  };

  const formatChange = (current: number, previous: number) => {
    const change = current - previous;
    const changePercent = ((change / previous) * 100).toFixed(2);
    const isPositive = change >= 0;

    return {
      value: `${isPositive ? '+' : ''}${change.toFixed(2)}`,
      percent: `${isPositive ? '+' : ''}${changePercent}%`,
      isPositive,
    };
  };

  const lastData = data[data.length - 1];
  const previousData = data[data.length - 2];
  const change = lastData && previousData ? formatChange(lastData.close, previousData.close) : null;

  // Calculate signal positions
  const getSignalPosition = (signal: Signal, index: number) => {
    const padding = { top: 60, left: 80, right: 80, bottom: 60 };
    const chartWidth = dimensions.width - padding.left - padding.right;
    const chartHeight = dimensions.height - padding.top - padding.bottom;

    let minPrice = Infinity;
    let maxPrice = -Infinity;
    data.forEach(d => {
      minPrice = Math.min(minPrice, d.low);
      maxPrice = Math.max(maxPrice, d.high);
    });
    const priceRange = maxPrice - minPrice;
    const pricePadding = priceRange * 0.1;
    minPrice -= pricePadding;
    maxPrice += pricePadding;

    const x = padding.left + (index / (data.length - 1)) * chartWidth;
    const y = padding.top + (1 - (signal.price - minPrice) / (maxPrice - minPrice)) * chartHeight;

    return { x: x - 16, y: y - 16 };
  };

  return (
    <ChartContainer ref={containerRef}>
      {showWatermark && (
        <Watermark>
          <div className="symbol">{symbol}</div>
          <div className="brand">GoldenSignalsAI</div>
        </Watermark>
      )}

      <Canvas
        ref={gridCanvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ width: dimensions.width, height: dimensions.height }}
      />

      <GlowCanvas
        ref={glowCanvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ width: dimensions.width, height: dimensions.height }}
      />

      <Canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ width: dimensions.width, height: dimensions.height }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />

      {/* Trading signals */}
      {signals.map((signal, i) => {
        const position = getSignalPosition(signal, i);
        return (
          <SignalMarker
            key={i}
            type={signal.type}
            style={{ left: position.x, top: position.y }}
          >
            {signal.type === 'buy' ? '↑' : '↓'}
          </SignalMarker>
        );
      })}

      {lastData && (
        <InfoPanel>
          <div className="header">
            <Typography variant="h6" fontWeight="bold">{symbol}</Typography>
            {change && (
              <Chip
                label={change.percent}
                size="small"
                color={change.isPositive ? 'success' : 'error'}
                icon={change.isPositive ? <TrendingUpIcon /> : <TrendingDownIcon />}
              />
            )}
          </div>
          <div className="price">
            {formatPrice(lastData.close)}
            {change && (
              <Typography
                component="span"
                sx={{
                  ml: 1,
                  fontSize: '0.875rem',
                  color: change.isPositive ? 'success.main' : 'error.main',
                }}
              >
                {change.value}
              </Typography>
            )}
          </div>
          <div className="details">
            <div>Open: {formatPrice(lastData.open)}</div>
            <div>High: {formatPrice(lastData.high)}</div>
            <div>Low: {formatPrice(lastData.low)}</div>
            <div>Vol: {formatVolume(lastData.volume)}</div>
          </div>
        </InfoPanel>
      )}

      {/* Indicator toggles */}
      <Box position="absolute" bottom={16} right={16} display="flex" gap={1}>
        <IndicatorButton size="small" title="Moving Average">
          <TimelineIcon />
        </IndicatorButton>
        <IndicatorButton size="small" title="Volume">
          <BarChartIcon />
        </IndicatorButton>
        <IndicatorButton size="small" title="Indicators">
          <ShowChartIcon />
        </IndicatorButton>
      </Box>
    </ChartContainer>
  );
};

export default EnhancedCustomChart;
