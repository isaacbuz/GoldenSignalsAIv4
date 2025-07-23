/**
 * Simplified AITradingChart
 * Minimal version without complex dependencies to isolate issues
 */

import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, Button } from '@mui/material';
import { styled } from '@mui/material/styles';

const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  minHeight: '400px',
  backgroundColor: '#000000',
  display: 'flex',
  flexDirection: 'column',
}));

const Header = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(2),
  background: 'linear-gradient(180deg, rgba(0, 0, 0, 0.8) 0%, rgba(0, 0, 0, 0.6) 100%)',
  backdropFilter: 'blur(20px)',
  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  zIndex: 10,
}));

const CanvasContainer = styled(Box)({
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
});

interface AITradingChartSimpleProps {
  symbol?: string;
  height?: string | number;
}

const AITradingChartSimple: React.FC<AITradingChartSimpleProps> = ({
  symbol = 'TSLA',
  height = '100vh'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<any[]>([]);

  // Draw simple chart
  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw watermark
    ctx.save();
    ctx.font = 'bold 120px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, 'rgba(255, 215, 0, 0.1)');
    gradient.addColorStop(0.5, 'rgba(255, 228, 0, 0.15)');
    gradient.addColorStop(1, 'rgba(255, 200, 0, 0.1)');

    ctx.fillStyle = gradient;
    ctx.shadowColor = 'rgba(255, 215, 0, 0.3)';
    ctx.shadowBlur = 30;

    ctx.fillText(symbol.toUpperCase(), canvas.width / 2, canvas.height / 2 - 50);

    ctx.font = 'normal 24px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillText('GoldenSignalsAI', canvas.width / 2, canvas.height / 2 + 50);
    ctx.restore();

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    // Vertical lines
    for (let x = 0; x < canvas.width; x += 100) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw sample candlesticks if we have data
    if (data.length > 0) {
      const candleWidth = (canvas.width / data.length) * 0.8;
      const candleSpacing = (canvas.width / data.length) * 0.2;

      data.forEach((candle, index) => {
        const x = index * (candleWidth + candleSpacing) + candleSpacing / 2;
        const isGreen = candle.close > candle.open;

        // Candle body
        ctx.fillStyle = isGreen ? '#00D964' : '#FF3B30';
        const bodyHeight = Math.abs(candle.close - candle.open) * 2;
        const bodyY = Math.min(candle.open, candle.close) * 2 + 100;

        ctx.fillRect(x, bodyY, candleWidth, bodyHeight);

        // Wick
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x + candleWidth / 2, candle.high * 2 + 80);
        ctx.lineTo(x + candleWidth / 2, candle.low * 2 + 120);
        ctx.stroke();
      });
    }
  };

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        canvasRef.current.width = canvasRef.current.offsetWidth;
        canvasRef.current.height = canvasRef.current.offsetHeight;
        drawChart();
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => window.removeEventListener('resize', handleResize);
  }, [data]);

  // Fetch data
  const fetchData = async () => {
    setIsLoading(true);
    try {
      // Generate sample data for now
      const sampleData = Array.from({ length: 50 }, (_, i) => ({
        time: Date.now() - (50 - i) * 60000,
        open: 100 + Math.random() * 10,
        high: 105 + Math.random() * 10,
        low: 95 + Math.random() * 10,
        close: 100 + Math.random() * 10,
        volume: Math.floor(Math.random() * 1000000)
      }));

      setData(sampleData);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [symbol]);

  useEffect(() => {
    drawChart();
  }, [data]);

  return (
    <ChartContainer sx={{ height }}>
      <Header>
        <Typography variant="h5" sx={{ color: '#FFD700' }}>
          {symbol} - GoldenSignalsAI
        </Typography>
        <Button
          variant="contained"
          onClick={fetchData}
          disabled={isLoading}
          sx={{
            backgroundColor: '#007AFF',
            '&:hover': { backgroundColor: '#0051D5' }
          }}
        >
          {isLoading ? 'Loading...' : 'Refresh'}
        </Button>
      </Header>

      <CanvasContainer>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block'
          }}
        />
      </CanvasContainer>
    </ChartContainer>
  );
};

export default AITradingChartSimple;
