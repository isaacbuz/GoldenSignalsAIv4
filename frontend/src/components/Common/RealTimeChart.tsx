/**
 * Real-Time Chart Component
 * 
 * Advanced charting with real-time data updates for institutional trading
 * Refactored to use @mui/x-charts for consistency with the design system.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Box,
  Card,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  FormControl,
  Select,
  Chip,
  Stack,
  Tooltip,
  useTheme,
  Skeleton,
} from '@mui/material';
import {
  MoreVert,
  Fullscreen,
  FullscreenExit,
  TrendingUp,
  TrendingDown,
  Settings,
  Download,
} from '@mui/icons-material';
import { LineChart, ChartsXAxis, ChartsYAxis, LinePlot, MarkPlot, ResponsiveChartContainer } from '@mui/x-charts';
import { useWebSocket } from '../../services/websocket';
import { formatPrice } from '../../services/api';

interface ChartDataPoint {
  timestamp: number;
  price: number;
}

interface Signal {
  timestamp: number;
  type: 'BUY' | 'SELL' | 'HOLD';
  price: number;
}

interface RealTimeChartProps {
  symbol: string;
  height?: number;
  showSignals?: boolean;
  timeframe?: '1m' | '5m' | '15m' | '1h';
  maxDataPoints?: number;
}

export default function RealTimeChart({
  symbol,
  height = 400,
  showSignals = true,
  timeframe = '5m',
  maxDataPoints = 200,
}: RealTimeChartProps) {
  const theme = useTheme();
  const chartRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [isLoading, setIsLoading] = useState(true);
  const [lastPrice, setLastPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [priceChangePercent, setPriceChangePercent] = useState<number>(0);

  const { subscribeToSymbol, unsubscribeFromSymbol, isConnected } = useWebSocket();

  // Subscribe to real-time data
  useEffect(() => {
    if (isConnected && symbol) {
      subscribeToSymbol(symbol);

      const generateInitialData = () => {
        const now = Date.now();
        const initialData: ChartDataPoint[] = [];
        let basePrice = 100 + Math.random() * 400;

        for (let i = maxDataPoints; i >= 0; i--) {
          const timestamp = now - i * getTimeframeMs(selectedTimeframe);
          const volatility = 0.02;
          const change = (Math.random() - 0.5) * volatility * basePrice;
          const price = Math.max(0.01, basePrice + change);
          initialData.push({ timestamp, price });
          basePrice = price;
        }

        setData(initialData);
        setLastPrice(basePrice);
        setIsLoading(false);
      };

      generateInitialData();

      return () => {
        unsubscribeFromSymbol(symbol);
      };
    }
  }, [symbol, isConnected, selectedTimeframe, maxDataPoints, subscribeToSymbol, unsubscribeFromSymbol]);

  // Simulate real-time updates
  useEffect(() => {
    if (!isConnected || isLoading) return;

    const interval = setInterval(() => {
      setData(prevData => {
        if (prevData.length === 0) return prevData;

        const lastPoint = prevData[prevData.length - 1];
        const newPrice = Math.max(0.01, lastPoint.price + (Math.random() - 0.5) * 0.001 * lastPoint.price);
        const newPoint = { timestamp: Date.now(), price: newPrice };
        
        if (prevData.length > 1) {
          const firstPrice = prevData[0].price;
          setPriceChange(newPrice - firstPrice);
          setPriceChangePercent(((newPrice - firstPrice) / firstPrice) * 100);
        }

        setLastPrice(newPrice);
        return [...prevData, newPoint].slice(-maxDataPoints);
      });

      if (showSignals && Math.random() < 0.1) {
        setSignals(prev => [
          ...prev,
          {
            timestamp: Date.now(),
            type: Math.random() > 0.5 ? 'BUY' : 'SELL',
            price: lastPrice || 0,
          },
        ].slice(-20) as Signal[]);
      }
    }, getTimeframeMs(selectedTimeframe) / 10);

    return () => clearInterval(interval);
  }, [isConnected, isLoading, selectedTimeframe, lastPrice, maxDataPoints, showSignals]);

  const getTimeframeMs = (tf: string): number => {
    const timeframes: Record<string, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
    };
    return timeframes[tf] || timeframes['5m'];
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => setAnchorEl(event.currentTarget);
  const handleMenuClose = () => setAnchorEl(null);

  const toggleFullscreen = useCallback(() => {
    const element = chartRef.current;
    if (!element) return;
    if (!document.fullscreenElement) {
      element.requestFullscreen().then(() => setIsFullscreen(true));
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false));
    }
  }, []);
  
  const getSignalColor = (type: string) => {
    return type === 'BUY' ? theme.palette.success.main : theme.palette.error.main;
  };
  
  const priceData = data.map(d => d.price);
  const timestampData = data.map(d => new Date(d.timestamp));
  
  const yAxisDomain = [Math.min(...priceData) * 0.99, Math.max(...priceData) * 1.01];

  return (
    <Card sx={{ height: isFullscreen ? '100vh' : height, display: 'flex', flexDirection: 'column' }} ref={chartRef}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ p: 2 }}>
        <Box>
          <Typography variant="h6">{symbol} Real-Time</Typography>
          {lastPrice && (
            <Stack direction="row" alignItems="center" spacing={1}>
              <Typography variant="body1" color={priceChange >= 0 ? 'success.main' : 'error.main'}>
                {formatPrice(lastPrice)}
              </Typography>
              <Chip
                icon={priceChange >= 0 ? <TrendingUp /> : <TrendingDown />}
                label={`${priceChange.toFixed(2)} (${priceChangePercent.toFixed(2)}%)`}
                size="small"
                color={priceChange >= 0 ? 'success' : 'error'}
                variant="outlined"
              />
            </Stack>
          )}
        </Box>
        <Box>
          <FormControl size="small" variant="outlined" sx={{ mr: 1 }}>
            <Select value={selectedTimeframe} onChange={(e) => setSelectedTimeframe(e.target.value as any)}>
              <MenuItem value="1m">1m</MenuItem>
              <MenuItem value="5m">5m</MenuItem>
              <MenuItem value="15m">15m</MenuItem>
              <MenuItem value="1h">1h</MenuItem>
            </Select>
          </FormControl>
          <IconButton onClick={handleMenuClick}>
            <MoreVert />
          </IconButton>
          <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleMenuClose}>
            <MenuItem onClick={toggleFullscreen}>{isFullscreen ? <FullscreenExit /> : <Fullscreen />} Toggle Fullscreen</MenuItem>
            <MenuItem onClick={() => { console.log('Exporting...'); handleMenuClose(); }}>
              <Download /> Export Data
            </MenuItem>
            <MenuItem onClick={handleMenuClose}><Settings /> Chart Settings</MenuItem>
          </Menu>
        </Box>
      </Stack>
      <Box sx={{ flexGrow: 1, width: '100%', height: '100%' }}>
        {isLoading ? (
          <Skeleton variant="rectangular" width="100%" height="100%" />
        ) : (
          <ResponsiveChartContainer
            series={[{ type: 'line', data: priceData }]}
            xAxis={[{ 
              data: timestampData, 
              scaleType: 'time',
              valueFormatter: (date) => new Date(date).toLocaleTimeString()
            }]}
            yAxis={[{ 
              min: yAxisDomain[0], 
              max: yAxisDomain[1],
              valueFormatter: (price) => formatPrice(price)
            }]}
            margin={{ top: 20, right: 30, bottom: 40, left: 50 }}
          >
            <LinePlot />
            {showSignals && (
              <MarkPlot>
                {signals.map((signal, i) => (
                  <Tooltip key={i} title={`${signal.type} @ ${formatPrice(signal.price)}`}>
                    <g transform={`translate(${new Date(signal.timestamp).getTime()}, ${signal.price})`}>
                       <circle r="5" fill={getSignalColor(signal.type)} />
                    </g>
                  </Tooltip>
                ))}
              </MarkPlot>
            )}
            <ChartsXAxis
              label="Time"
              tickLabelStyle={{ fill: theme.palette.text.secondary }}
              labelStyle={{ fill: theme.palette.text.primary }}
            />
            <ChartsYAxis
              label="Price (USD)"
              tickLabelStyle={{ fill: theme.palette.text.secondary }}
              labelStyle={{ fill: theme.palette.text.primary }}
            />
          </ResponsiveChartContainer>
        )}
      </Box>
    </Card>
  );
} 