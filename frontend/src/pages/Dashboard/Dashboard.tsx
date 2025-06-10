/**
 * Professional Trading Dashboard - Inspired by TD Ameritrade & E*TRADE
 * 
 * Clean, functional design focused on trading data and insights
 */

import React from 'react';
import { Box, Card, CardContent, Typography, useTheme, alpha, Avatar, Stack, List, ListItem, ListItemText, ListItemAvatar, Container, Grid, Chip, Divider, CircularProgress, Alert, Skeleton } from '@mui/material';
import { motion } from 'framer-motion';
import TradingChart from '../../components/Chart/TradingChart';
import {
  TrendingUp,
  ShowChart,
  BarChart,
  Speed,
  TrendingDown,
  Psychology,
  Assessment,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { apiClient, Signal, MarketData } from '../../services/api';

// --- WIDGET DEFINITIONS ---

// 1. Metric Card
interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon: React.ReactNode;
  color?: string;
  isLoading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, trend = 'neutral', icon, color, isLoading }) => {
  const theme = useTheme();
  const trendColor = trend === 'up' ? theme.palette.success.main : trend === 'down' ? theme.palette.error.main : theme.palette.text.secondary;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        {isLoading ? (
          <Box>
            <Skeleton variant="text" width="60%" />
            <Skeleton variant="text" width="40%" sx={{ fontSize: 'h5.fontSize', my: 1 }} />
            <Skeleton variant="text" width="50%" />
          </Box>
        ) : (
          <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
            <Box>
              <Typography variant="body2" color="text.secondary">{title}</Typography>
              <Typography variant="h5" sx={{ fontWeight: 'bold', my: 1 }}>{value}</Typography>
              {change && (
                <Stack direction="row" alignItems="center" spacing={0.5}>
                  {trend === 'up' ? <TrendingUp sx={{ color: trendColor, fontSize: '1rem' }} /> : <TrendingDown sx={{ color: trendColor, fontSize: '1rem' }} />}
                  <Typography variant="body2" sx={{ color: trendColor }}>{change}</Typography>
                </Stack>
              )}
            </Box>
            <Avatar sx={{ bgcolor: alpha(color || theme.palette.primary.main, 0.1), color: color || theme.palette.primary.main }}>{icon}</Avatar>
          </Stack>
        )}
      </CardContent>
    </Card>
  );
};

// 2. Signal Stream with Real Data
const SignalStream: React.FC = () => {
  const theme = useTheme();
  
  // Fetch latest signals from API
  const { data: latestSignals, isLoading: signalsLoading, error: signalsError } = useQuery({
    queryKey: ['latest-signals'],
    queryFn: () => apiClient.getLatestSignals(8),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
  
  if (signalsLoading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Signal Stream</Typography>
          <Stack spacing={1} sx={{ mt: 2 }}>
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} variant="rounded" height={60} />
            ))}
          </Stack>
        </CardContent>
      </Card>
    );
  }

  if (signalsError) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Signal Stream</Typography>
          <Alert severity="error">Failed to load signals</Alert>
        </CardContent>
      </Card>
    );
  }

  const signals = latestSignals || [];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Signal Stream</Typography>
        {signals.length === 0 ? (
          <Typography variant="body2" color="text.secondary" textAlign="center" py={4}>
            No signals available
          </Typography>
        ) : (
          <List dense>
            {signals.map((signal, index) => (
              <ListItem key={signal.signal_id || index} divider={index < signals.length - 1}>
                <ListItemAvatar>
                  <Avatar sx={{ 
                    bgcolor: signal.signal_type === 'BUY' 
                      ? alpha(theme.palette.success.main, 0.1) 
                      : signal.signal_type === 'SELL'
                      ? alpha(theme.palette.error.main, 0.1)
                      : alpha(theme.palette.warning.main, 0.1)
                  }}>
                    {signal.signal_type === 'BUY' ? (
                      <TrendingUp sx={{color: 'success.main'}}/>
                    ) : signal.signal_type === 'SELL' ? (
                      <TrendingDown sx={{color: 'error.main'}}/>
                    ) : (
                      <BarChart sx={{color: 'warning.main'}}/>
                    )}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography variant="body2" fontWeight="semibold">
                        {signal.symbol} - {signal.signal_type}
                      </Typography>
                      <Chip 
                        label={signal.strength || 'MODERATE'}
                        size="small"
                        variant="outlined"
                        sx={{ fontSize: '0.75rem' }}
                      />
                    </Stack>
                  }
                  secondary={`Confidence: ${Math.round(signal.confidence * 100)}% | ${signal.current_price ? `$${signal.current_price.toFixed(2)}` : 'Price N/A'}`}
                />
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

// 3. Market Pulse with Real Data
const MarketPulse: React.FC = () => {
  const { data: marketSummary, isLoading } = useQuery({
    queryKey: ['market-summary'],
    queryFn: () => apiClient.getMarketSummary(),
    refetchInterval: 60000, // Refresh every minute
  });

  // Fetch market data for key symbols
  const symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY'];
  const { data: multiMarketData, isLoading: marketDataLoading } = useQuery({
    queryKey: ['multi-market-data', symbols],
    queryFn: () => apiClient.getMultipleMarketData(symbols),
    refetchInterval: 30000,
  });

  if (isLoading || marketDataLoading) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Market Pulse</Typography>
          <Stack spacing={1} sx={{ mt: 2 }}>
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} variant="rounded" height={40} />
            ))}
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Market Pulse</Typography>
        <List dense>
          {symbols.map((symbol, index) => {
            const data = multiMarketData?.[symbol];
            const isPositive = (data?.change || 0) >= 0;
            
            return (
              <ListItem key={symbol} divider={index < symbols.length - 1}>
                <ListItemText 
                  primary={
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2" fontWeight="medium">{symbol}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        ${data?.price?.toFixed(2) || 'N/A'}
                      </Typography>
                    </Stack>
                  }
                  secondary={
                    data ? (
                      <Typography 
                        variant="body2" 
                        color={isPositive ? 'success.main' : 'error.main'}
                      >
                        {isPositive ? '+' : ''}{data.change?.toFixed(2)} ({isPositive ? '+' : ''}{data.change_percent?.toFixed(2)}%)
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">Loading...</Typography>
                    )
                  }
                />
              </ListItem>
            );
          })}
        </List>
      </CardContent>
    </Card>
  );
};

// 4. Agent Performance Widget with Real Data
const AgentPerformanceWidget: React.FC = () => {
  const { data: performanceData, isLoading, error } = useQuery({
    queryKey: ['agent-performance'],
    queryFn: () => apiClient.getAgentPerformance(),
    refetchInterval: 60000, // Refresh every minute
  });

  if (isLoading) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Agent Performance</Typography>
          <Stack spacing={2} sx={{ mt: 2 }}>
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} variant="rounded" height={24} />
            ))}
          </Stack>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Agent Performance</Typography>
          <Alert severity="error">Failed to load agent performance</Alert>
        </CardContent>
      </Card>
    );
  }

  const agents = performanceData?.agents || {};
  
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Agent Performance</Typography>
        <Stack spacing={2} sx={{ mt: 2 }}>
          {Object.values(agents).map((agent: any) => (
            <Box key={agent.agent_name}>
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Stack direction="row" spacing={1} alignItems="center">
                  <Psychology sx={{ fontSize: '1rem', color: 'primary.main' }} />
                  <Typography variant="body2" textTransform="capitalize">
                    {agent.agent_name.replace('_', ' ')}
                  </Typography>
                </Stack>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Typography variant="body2" color="text.secondary">
                    {agent.accuracy.toFixed(1)}%
                  </Typography>
                  <Chip 
                    label={`${agent.total_signals} signals`}
                    size="small" 
                    variant="outlined"
                    sx={{ fontSize: '0.75rem' }}
                  />
                </Stack>
              </Stack>
            </Box>
          ))}
          
          {performanceData?.summary && (
            <>
              <Divider sx={{ my: 1 }} />
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Overall: {performanceData?.summary?.avg_accuracy?.toFixed(1)}% accuracy • {performanceData?.summary?.total_signals} total signals
                </Typography>
              </Box>
            </>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

// 5. Watchlist Widget with Real Data
const Watchlist: React.FC = () => {
  const watchlistSymbols = ['NVDA', 'AMD', 'META'];
  const { data: watchlistData, isLoading } = useQuery({
    queryKey: ['watchlist-data', watchlistSymbols],
    queryFn: () => apiClient.getMultipleMarketData(watchlistSymbols),
    refetchInterval: 30000,
  });

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Watchlist</Typography>
        {isLoading ? (
          <Stack spacing={1} sx={{ mt: 2 }}>
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} variant="rounded" height={40} />
            ))}
          </Stack>
        ) : (
          <List dense>
            {watchlistSymbols.map((symbol, index) => {
              const data = watchlistData?.[symbol];
              const isPositive = (data?.change || 0) >= 0;
              
              return (
                <ListItem key={symbol} divider={index < watchlistSymbols.length - 1}>
                  <ListItemText 
                    primary={
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2" fontWeight="medium">{symbol}</Typography>
                        <Typography variant="body2">
                          ${data?.price?.toFixed(2) || 'Loading...'}
                        </Typography>
                      </Stack>
                    }
                    secondary={
                      data && (
                        <Typography 
                          variant="body2" 
                          color={isPositive ? 'success.main' : 'error.main'}
                          textAlign="right"
                        >
                          {isPositive ? '+' : ''}{data.change?.toFixed(2)} ({isPositive ? '+' : ''}{data.change_percent?.toFixed(2)}%)
                        </Typography>
                      )
                    }
                  />
                </ListItem>
              );
            })}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

// Market Sentiment Card with Real Data
const MarketSentimentCard: React.FC = () => {
  const { data: latestSignals, isLoading } = useQuery({
    queryKey: ['market-sentiment-signals'],
    queryFn: () => apiClient.getLatestSignals(20),
    refetchInterval: 60000,
  });

  // Calculate market sentiment from signals
  const bullishSignals = latestSignals?.filter(s => s.signal_type === 'BUY').length || 0;
  const bearishSignals = latestSignals?.filter(s => s.signal_type === 'SELL').length || 0;
  const totalSignals = latestSignals?.length || 0;
  
  const bullishPercentage = totalSignals > 0 ? (bullishSignals / totalSignals) * 100 : 50;
  const sentiment = bullishPercentage > 60 ? 'Bullish' : bullishPercentage < 40 ? 'Bearish' : 'Neutral';
  const sentimentColor = sentiment === 'Bullish' ? 'success' : sentiment === 'Bearish' ? 'error' : 'warning';
  const SentimentIcon = sentiment === 'Bullish' ? TrendingUp : sentiment === 'Bearish' ? TrendingDown : BarChart;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Market Sentiment</Typography>
        {isLoading ? (
          <Stack spacing={1} sx={{ mt: 1 }}>
            <Skeleton variant="text" width="50%" />
            <Skeleton variant="text" width="70%" />
          </Stack>
        ) : (
          <Stack direction="row" alignItems="center" spacing={1}>
            <SentimentIcon color={sentimentColor} />
            <Typography variant="body1" color={`${sentimentColor}.main`}>
              {sentiment}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              ({Math.round(bullishPercentage)}% signals bullish)
            </Typography>
          </Stack>
        )}
      </CardContent>
    </Card>
  );
};

// Signal Rationale Widget
function SignalRationale({ signal }: { signal: any }) {
  if (!signal) return null;
  
  return (
    <Card sx={{ minWidth: 300, flex: 1 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Signal Rationale</Typography>
        <Chip
          label={signal.signal_type}
          color={signal.signal_type === 'BUY' ? 'success' : signal.signal_type === 'SELL' ? 'error' : 'warning'}
          sx={{ fontWeight: 700, mb: 1 }}
        />
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Confidence: {Math.round(signal.confidence * 100)}%
        </Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          {signal.reasoning || 'Signal generated based on technical analysis and market indicators.'}
        </Typography>
        
        {signal.indicators && (
          <>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Technical Indicators:</Typography>
            <Stack spacing={1}>
              {Object.entries(signal.indicators).slice(0, 4).map(([key, value]: [string, any]) => (
                <Stack key={key} direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    {key.replace('_', ' ').toUpperCase()}:
                  </Typography>
                  <Typography variant="body2">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                  </Typography>
                </Stack>
              ))}
            </Stack>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// --- MAIN DASHBOARD COMPONENT ---

const MotionStack = motion(Stack);

export default function Dashboard() {
  const theme = useTheme();
  const [selectedSignal, setSelectedSignal] = React.useState<any>(null);

  // Fetch latest signals for signal selection
  const { data: dashboardSignals } = useQuery({
    queryKey: ['dashboard-signals'],
    queryFn: () => apiClient.getLatestSignals(5),
    refetchInterval: 30000,
  });

  // Calculate metrics from real data
  const { data: allSignals } = useQuery({
    queryKey: ['all-signals-metrics'],
    queryFn: () => apiClient.getLatestSignals(20),
    refetchInterval: 60000,
  });

  const { data: agentPerformance } = useQuery({
    queryKey: ['agent-performance-metrics'],
    queryFn: () => apiClient.getAgentPerformance(),
    refetchInterval: 60000,
  });

  // Calculate real metrics
  const safeAllSignals = allSignals || [];
  const activeSignalsCount = safeAllSignals.filter(s => s.signal_type !== 'HOLD').length;
  const avgConfidence = safeAllSignals.reduce((sum, s) => sum + s.confidence, 0) / (safeAllSignals.length || 1);
  const bullishSignalsCount = safeAllSignals.filter(s => s.signal_type === 'BUY').length;
  const totalSignalsCount = safeAllSignals.length;
  const marketSentiment = totalSignalsCount > 0 ? (bullishSignalsCount / totalSignalsCount) * 100 : 50;
  
  // Get top performing symbol
  const symbolPerformance = safeAllSignals.reduce((acc: Record<string, { signals: number; avgConfidence: number }>, signal) => {
    if (!acc[signal.symbol]) {
      acc[signal.symbol] = { signals: 0, avgConfidence: 0 };
    }
    acc[signal.symbol].signals += 1;
    acc[signal.symbol].avgConfidence += signal.confidence;
    return acc;
  }, {} as Record<string, { signals: number; avgConfidence: number }>);
  
  const topSymbol = Object.entries(symbolPerformance).reduce<{ symbol: string; confidence: number }>((top, [symbol, data]) => {
    const avgConf = data.avgConfidence / data.signals;
    return avgConf > (top.confidence || 0) ? { symbol, confidence: avgConf } : top;
  }, { symbol: 'AAPL', confidence: 0.75 });

  // Extract agent performance accuracy to avoid TypeScript issues
  const agentAccuracy = agentPerformance?.summary?.avg_accuracy;
  const agentAccuracyText = agentAccuracy ? `${agentAccuracy.toFixed(1)}% Acc.` : 'N/A';

  const metrics: MetricCardProps[] = [
    { 
      title: 'Active Signals', 
      value: activeSignalsCount.toString(), 
      change: `${totalSignalsCount} Total`, 
      trend: 'up', 
      icon: <ShowChart />, 
      color: theme.palette.primary.main 
    },
    { 
      title: 'Market Sentiment', 
      value: marketSentiment > 60 ? 'Bullish' : marketSentiment < 40 ? 'Bearish' : 'Neutral', 
      change: `${Math.round(marketSentiment)}%`, 
      trend: marketSentiment > 50 ? 'up' : 'down', 
      icon: <TrendingUp />, 
      color: theme.palette.success.main 
    },
    { 
      title: 'Top Performer', 
      value: topSymbol.symbol, 
      change: `${Math.round(topSymbol.confidence * 100)}%`, 
      trend: 'up', 
      icon: <BarChart />, 
      color: theme.palette.secondary.main 
    },
    { 
      title: 'Avg. Confidence', 
      value: `${Math.round(avgConfidence * 100)}%`, 
      change: agentAccuracyText, 
      trend: avgConfidence > 0.7 ? 'up' : 'down', 
      icon: <Speed />, 
      color: theme.palette.warning.main 
    },
  ];

  // Auto-select the first signal
  React.useEffect(() => {
    if (dashboardSignals && dashboardSignals.length > 0 && !selectedSignal) {
      setSelectedSignal(dashboardSignals[0]);
    }
  }, [dashboardSignals, selectedSignal]);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <MotionStack spacing={4} alignItems="center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
        {/* Central Chart */}
        <Box sx={{ width: '100%', maxWidth: 1200, mx: 'auto' }}>
          <TradingChart defaultSymbol="AAPL" height={500} onSelectSignal={setSelectedSignal} />
        </Box>

        {/* Signal Summary Cards */}
        <Grid container spacing={3} justifyContent="center">
          {metrics.map((metric, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <MetricCard {...metric} />
            </Grid>
          ))}
        </Grid>

        {/* Signal Stream and Agent Performance */}
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} md={8}>
            <SignalStream />
          </Grid>
          <Grid item xs={12} md={4}>
            <AgentPerformanceWidget />
          </Grid>
        </Grid>

        {/* Market Pulse and Watchlist */}
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} md={6}>
            <MarketPulse />
          </Grid>
          <Grid item xs={12} md={6}>
            <Watchlist />
          </Grid>
        </Grid>

        {/* Market Sentiment Card and Signal Rationale */}
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} md={6}>
            <MarketSentimentCard />
          </Grid>
          <Grid item xs={12} md={6}>
            <SignalRationale signal={selectedSignal} />
          </Grid>
        </Grid>
      </MotionStack>
    </Container>
  );
} 