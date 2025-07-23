import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, useTheme, alpha } from '@mui/material';
import { styled } from '@mui/material/styles';

interface ChartDataPoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface CustomChartProps {
  data: ChartDataPoint[];
  width?: number;
  height?: number;
  showGrid?: boolean;
  showWatermark?: boolean;
  symbol?: string;
  indicators?: string[];
  onHover?: (data: ChartDataPoint | null) => void;
}

const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  backgroundColor: theme.palette.mode === 'dark' ? '#0A0E1A' : '#FAFAFA',
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
}));

const Canvas = styled('canvas')(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  cursor: 'crosshair',
}));

const Watermark = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  pointerEvents: 'none',
  userSelect: 'none',
  textAlign: 'center',
  '& .symbol': {
    fontSize: '8rem',
    fontWeight: 900,
    color: alpha(theme.palette.text.primary, 0.04),
    lineHeight: 1,
    letterSpacing: '0.1em',
  },
  '& .brand': {
    fontSize: '2rem',
    fontWeight: 700,
    letterSpacing: '0.3em',
    marginTop: '-1rem',
    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.15)}, ${alpha(theme.palette.secondary.main, 0.15)})`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
}));

const InfoOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  left: theme.spacing(2),
  padding: theme.spacing(1, 2),
  backgroundColor: alpha(theme.palette.background.paper, 0.9),
  borderRadius: theme.spacing(0.5),
  backdropFilter: 'blur(10px)',
  border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
  fontSize: '0.875rem',
  fontFamily: 'monospace',
  '& .price': {
    fontSize: '1.25rem',
    fontWeight: 'bold',
    marginBottom: theme.spacing(0.5),
  },
  '& .change': {
    marginLeft: theme.spacing(1),
  },
  '& .positive': {
    color: theme.palette.success.main,
  },
  '& .negative': {
    color: theme.palette.error.main,
  },
}));

export const CustomChart: React.FC<CustomChartProps> = ({
  data,
  width,
  height,
  showGrid = true,
  showWatermark = true,
  symbol = 'AAPL',
  indicators = [],
  onHover,
}) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
    const duration = 1000; // 1 second animation

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);

      setAnimationProgress(easeOutQuart(progress));

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
  const easeOutQuart = (t: number): number => {
    return 1 - Math.pow(1 - t, 4);
  };

  // Draw grid
  const drawGrid = useCallback(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    if (!showGrid) return;

    ctx.strokeStyle = alpha(theme.palette.divider, 0.1);
    ctx.lineWidth = 1;

    // Vertical lines
    const xStep = dimensions.width / 10;
    for (let x = 0; x <= dimensions.width; x += xStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, dimensions.height);
      ctx.stroke();
    }

    // Horizontal lines
    const yStep = dimensions.height / 8;
    for (let y = 0; y <= dimensions.height; y += yStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(dimensions.width, y);
      ctx.stroke();
    }
  }, [dimensions, showGrid, theme]);

  // Draw chart
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    const padding = { top: 40, right: 60, bottom: 40, left: 60 };
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

    // Draw candlesticks with animation
    const candleWidth = Math.max(1, (chartWidth / data.length) * 0.8);

    data.forEach((d, i) => {
      const x = xScale(i);
      const yHigh = yScale(d.high);
      const yLow = yScale(d.low);
      const yOpen = yScale(d.open);
      const yClose = yScale(d.close);

      const progress = animationProgress;
      const animatedY = (y: number) => dimensions.height - (dimensions.height - y) * progress;

      // Determine color
      const isGreen = d.close >= d.open;
      const color = isGreen ? theme.palette.success.main : theme.palette.error.main;

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1;

      // Draw wick
      ctx.beginPath();
      ctx.moveTo(x, animatedY(yHigh));
      ctx.lineTo(x, animatedY(yLow));
      ctx.stroke();

      // Draw body
      const bodyTop = animatedY(Math.min(yOpen, yClose));
      const bodyBottom = animatedY(Math.max(yOpen, yClose));
      const bodyHeight = bodyBottom - bodyTop;

      if (isGreen) {
        // Hollow candle for green
        ctx.strokeRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
      } else {
        // Filled candle for red
        ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
      }
    });

    // Draw moving average
    if (indicators.includes('sma')) {
      const smaData = calculateSMA(data, 20);

      ctx.strokeStyle = alpha(theme.palette.primary.main, 0.8);
      ctx.lineWidth = 2;
      ctx.beginPath();

      smaData.forEach((value, i) => {
        if (value !== null) {
          const x = xScale(i);
          const y = yScale(value) * animationProgress + dimensions.height * (1 - animationProgress);

          if (i === 0 || smaData[i - 1] === null) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
      });

      ctx.stroke();
    }

    // Draw volume bars
    if (indicators.includes('volume')) {
      const maxVolume = Math.max(...data.map(d => d.volume));
      const volumeHeight = chartHeight * 0.2;

      data.forEach((d, i) => {
        const x = xScale(i);
        const volumeBarHeight = (d.volume / maxVolume) * volumeHeight * animationProgress;
        const isGreen = d.close >= d.open;

        ctx.fillStyle = alpha(
          isGreen ? theme.palette.success.main : theme.palette.error.main,
          0.3
        );

        ctx.fillRect(
          x - candleWidth / 2,
          dimensions.height - padding.bottom - volumeBarHeight,
          candleWidth,
          volumeBarHeight
        );
      });
    }

    // Draw price labels
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const priceSteps = 5;
    for (let i = 0; i <= priceSteps; i++) {
      const price = minPrice + (i / priceSteps) * (maxPrice - minPrice);
      const y = yScale(price);

      ctx.fillText(price.toFixed(2), padding.left - 10, y);
    }

    // Draw crosshair if hovering
    if (mousePos.x > 0 && mousePos.y > 0) {
      ctx.strokeStyle = alpha(theme.palette.text.primary, 0.3);
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);

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

      ctx.setLineDash([]);
    }
  }, [data, dimensions, theme, indicators, animationProgress, mousePos]);

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

  // Handle mouse move
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePos({ x, y });

    // Find closest data point
    const padding = { left: 60, right: 60 };
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
    drawGrid();
  }, [drawGrid]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  // Format price change
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

      <Canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ width: dimensions.width, height: dimensions.height }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />

      {lastData && (
        <InfoOverlay>
          <div className="price">
            ${lastData.close.toFixed(2)}
            {change && (
              <span className={`change ${change.isPositive ? 'positive' : 'negative'}`}>
                {change.value} ({change.percent})
              </span>
            )}
          </div>
          <div>O: ${lastData.open.toFixed(2)} H: ${lastData.high.toFixed(2)}</div>
          <div>L: ${lastData.low.toFixed(2)} V: {(lastData.volume / 1000000).toFixed(1)}M</div>
        </InfoOverlay>
      )}
    </ChartContainer>
  );
};

export default CustomChart;
