/**
 * Simple Chart Component
 * Fallback chart with minimal dependencies
 */

import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

const ChartContainer = styled(Paper)({
  width: '100%',
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: '#ffffff',
});

const Header = styled(Box)({
  padding: '16px 24px',
  borderBottom: '1px solid #e0e0e0',
});

const CanvasContainer = styled(Box)({
  flex: 1,
  position: 'relative',
  padding: '20px',
});

interface SimpleChartProps {
  symbol?: string;
}

export const SimpleChart: React.FC<SimpleChartProps> = ({ symbol = 'TSLA' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, [symbol]);

  useEffect(() => {
    if (data.length > 0 && canvasRef.current) {
      drawChart();
    }
  }, [data]);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `http://localhost:8000/api/v1/market-data/${symbol}/history?period=1d&interval=5m`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }

      const result = await response.json();
      setData(result.data || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
      // Use sample data as fallback
      setData(generateSampleData());
    } finally {
      setLoading(false);
    }
  };

  const generateSampleData = () => {
    const now = Date.now();
    return Array.from({ length: 50 }, (_, i) => ({
      time: now - (50 - i) * 5 * 60 * 1000,
      open: 250 + Math.random() * 20,
      high: 255 + Math.random() * 20,
      low: 245 + Math.random() * 20,
      close: 250 + Math.random() * 20,
      volume: Math.floor(Math.random() * 1000000),
    }));
  };

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (data.length === 0) return;

    // Calculate price range
    const prices = data.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Draw grid
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = (canvas.height * i) / 10;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();

      // Price labels
      const price = maxPrice - (priceRange * i) / 10;
      ctx.fillStyle = '#666';
      ctx.font = '12px -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`$${price.toFixed(2)}`, canvas.width - 10, y + 4);
    }

    // Draw candlesticks
    const candleWidth = canvas.width / data.length * 0.8;
    const spacing = canvas.width / data.length * 0.2;

    data.forEach((candle, i) => {
      const x = i * (candleWidth + spacing) + spacing / 2;
      const isGreen = candle.close >= candle.open;

      // Calculate positions
      const highY = ((maxPrice - candle.high) / priceRange) * canvas.height;
      const lowY = ((maxPrice - candle.low) / priceRange) * canvas.height;
      const openY = ((maxPrice - candle.open) / priceRange) * canvas.height;
      const closeY = ((maxPrice - candle.close) / priceRange) * canvas.height;

      // Draw wick
      ctx.strokeStyle = isGreen ? '#00c853' : '#ff1744';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();

      // Draw body
      ctx.fillStyle = isGreen ? '#00c853' : '#ff1744';
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY);
      ctx.fillRect(x, bodyTop, candleWidth, bodyHeight || 1);
    });

    // Draw title
    ctx.fillStyle = '#000';
    ctx.font = 'bold 24px -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${symbol} - $${data[data.length - 1].close.toFixed(2)}`, 20, 40);
  };

  if (loading) {
    return (
      <ChartContainer>
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
          <CircularProgress />
        </Box>
      </ChartContainer>
    );
  }

  return (
    <ChartContainer>
      <Header>
        <Typography variant="h5" component="h1">
          {symbol} Chart
        </Typography>
        {error && (
          <Typography variant="body2" color="error">
            {error} - Using sample data
          </Typography>
        )}
      </Header>

      <CanvasContainer>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            maxHeight: '600px'
          }}
        />
      </CanvasContainer>
    </ChartContainer>
  );
};

export default SimpleChart;
