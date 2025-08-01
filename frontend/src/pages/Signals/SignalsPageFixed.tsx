/**
 * Fixed Signals Page
 *
 * A simplified version of the signals page that focuses on showing the AI Trading Chart
 * with real market data without TypeScript errors
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack,
  useTheme,
  alpha,
  CircularProgress,
  Alert,
  Divider,
  Button,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as ShowChartIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { WorkingChart } from '../../components/Chart/WorkingChart';

const StyledCard = styled(Card)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(10px)',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: alpha(theme.palette.primary.main, 0.3),
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[8],
  },
}));

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
}


export const SignalsPageFixed: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket connected');
      // Subscribe to symbol updates
      ws.send(JSON.stringify({
        type: 'subscribe',
        symbol: selectedSymbol
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'price_update' && data.symbol === selectedSymbol) {
          setMarketData(prev => prev ? {
            ...prev,
            price: data.price,
            change: data.change || prev.change,
            changePercent: data.changePercent || prev.changePercent,
          } : null);
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [selectedSymbol]);

  // Fetch market data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch historical data to get current price
        const priceResponse = await axios.get(
          `http://localhost:8000/api/v1/market-data/historical/${selectedSymbol}`,
          {
            params: {
              timeframe: '1h',
              bars: 1
            }
          }
        );

        // Process market data from historical data
        const data = priceResponse.data;
        if (data && data.length > 0) {
          const latestBar = data[data.length - 1];
          setMarketData({
            symbol: selectedSymbol,
            price: latestBar.close || 0,
            change: latestBar.close - latestBar.open,
            changePercent: ((latestBar.close - latestBar.open) / latestBar.open) * 100,
            volume: latestBar.volume || 0,
            high: latestBar.high || 0,
            low: latestBar.low || 0,
            open: latestBar.open || 0,
          });
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to fetch market data. Please check if the backend is running.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);


  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth={false} sx={{ mt: 2, mb: 4 }}>
      <Grid container spacing={3}>
        {/* Header */}
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="h4" sx={{ fontWeight: 600, color: theme.palette.primary.main }}>
              AI Trading Signals
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center">
              <Button
                variant="contained"
                color="primary"
                onClick={() => navigate('/ai-chart')}
                sx={{ fontWeight: 600 }}
              >
                View AI Chart
              </Button>
              <Stack direction="row" spacing={1}>
              {['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'].map(symbol => (
                <Chip
                  key={symbol}
                  label={symbol}
                  onClick={() => setSelectedSymbol(symbol)}
                  color={selectedSymbol === symbol ? 'primary' : 'default'}
                  variant={selectedSymbol === symbol ? 'filled' : 'outlined'}
                  sx={{ cursor: 'pointer' }}
                />
              ))}
              </Stack>
            </Stack>
          </Box>
        </Grid>

        {/* Market Overview Card */}
        <Grid item xs={12} md={3}>
          <StyledCard>
            <CardContent>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    {selectedSymbol}
                  </Typography>
                  <Typography variant="h3" sx={{ mt: 1, fontWeight: 600 }}>
                    ${marketData?.price.toFixed(2)}
                  </Typography>
                </Box>

                <Box display="flex" alignItems="center" gap={1}>
                  {marketData && marketData.change >= 0 ? (
                    <TrendingUpIcon sx={{ color: theme.palette.success.main }} />
                  ) : (
                    <TrendingDownIcon sx={{ color: theme.palette.error.main }} />
                  )}
                  <Typography
                    variant="h6"
                    sx={{
                      color: marketData && marketData.change >= 0
                        ? theme.palette.success.main
                        : theme.palette.error.main,
                      fontWeight: 600,
                    }}
                  >
                    {marketData?.change >= 0 ? '+' : ''}{marketData?.change.toFixed(2)}
                    ({marketData?.changePercent.toFixed(2)}%)
                  </Typography>
                </Box>

                <Divider />

                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Day Range
                  </Typography>
                  <Typography variant="body1">
                    ${marketData?.low.toFixed(2)} - ${marketData?.high.toFixed(2)}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Volume
                  </Typography>
                  <Typography variant="body1">
                    {marketData?.volume.toLocaleString()}
                  </Typography>
                </Box>

                <Chip
                  icon={<ShowChartIcon />}
                  label="Live Data"
                  color="primary"
                  size="small"
                  sx={{ mt: 1 }}
                />
              </Stack>
            </CardContent>
          </StyledCard>
        </Grid>

        {/* Chart */}
        <Grid item xs={12} md={9}>
          <WorkingChart symbol={selectedSymbol} height={550} />
        </Grid>

        {/* AI Signals */}
        <Grid item xs={12}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                AI Analysis Status
              </Typography>
              <Alert severity="info">
                AI agents are analyzing market conditions. Real-time signals will appear here once the analysis is complete.
              </Alert>
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>
    </Container>
  );
};
