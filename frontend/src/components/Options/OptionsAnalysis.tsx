import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  Chip,
  Grid,
  Divider,
  useTheme,
  alpha,
  Tooltip,
  IconButton,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Timeline,
  Info,
  Speed,
  Assessment,
  AccessTime,
  AttachMoney,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { fetchActiveSignals, AISignal } from '../../services/api';

interface OptionsMetric {
  label: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  tooltip?: string;
}

interface OptionsAnalysisProps {
  symbol: string;
  metrics: {
    ivPercentile: number;
    putCallRatio: number;
    volumeAnalysis: {
      trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
      value: number;
    };
    optionsFlow: {
      sentiment: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL';
      callVolume: number;
      putVolume: number;
    };
  };
}

export const OptionsAnalysis: React.FC<OptionsAnalysisProps> = ({ symbol, metrics }) => {
  const theme = useTheme();

  const { data: signals, isLoading } = useQuery<AISignal[]>({
    queryKey: ['activeSignals'],
    queryFn: fetchActiveSignals,
    // Disabled auto-refresh to prevent constant updating
    staleTime: 300000, // Keep data fresh for 5 minutes
  });

  const getFlowColor = (sentiment: string) => {
    switch (sentiment) {
      case 'STRONG_BUY':
        return theme.palette.success.main;
      case 'BUY':
        return theme.palette.success.light;
      case 'STRONG_SELL':
        return theme.palette.error.main;
      case 'SELL':
        return theme.palette.error.light;
      default:
        return theme.palette.warning.main;
    }
  };

  const optionsMetrics: OptionsMetric[] = [
    {
      label: 'IV Percentile',
      value: `${metrics.ivPercentile}%`,
      tooltip: 'Current implied volatility relative to its 52-week range',
    },
    {
      label: 'Put/Call Ratio',
      value: metrics.putCallRatio.toFixed(2),
      trend: metrics.putCallRatio > 1 ? 'down' : 'up',
      tooltip: 'Ratio of put options to call options volume',
    },
    {
      label: 'Call Volume',
      value: metrics.optionsFlow.callVolume.toLocaleString(),
      trend: 'up',
      tooltip: 'Total call options contracts traded today',
    },
    {
      label: 'Put Volume',
      value: metrics.optionsFlow.putVolume.toLocaleString(),
      trend: 'down',
      tooltip: 'Total put options contracts traded today',
    },
  ];

  if (isLoading) {
    return <LinearProgress />;
  }

  if (!signals || signals.length === 0) {
    return (
      <Card sx={{ height: '100%', bgcolor: 'background.paper' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Options Analysis
          </Typography>
          <Typography color="text.secondary">No active signals available</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%', bgcolor: 'background.paper' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Options Analysis
        </Typography>

        <Grid container spacing={3}>
          {signals.map((signal) => (
            <Grid item xs={12} key={signal.id}>
              <Card
                variant="outlined"
                sx={{
                  borderColor: signal.type === 'CALL' ? 'success.main' : 'error.main',
                }}
              >
                <CardContent>
                  <Stack spacing={2}>
                    {/* Header */}
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography variant="h6">{signal.symbol}</Typography>
                        <Chip
                          icon={signal.type === 'CALL' ? <TrendingUp /> : <TrendingDown />}
                          label={signal.type}
                          color={signal.type === 'CALL' ? 'success' : 'error'}
                          size="small"
                        />
                        <Chip
                          icon={<Speed />}
                          label={signal.urgency}
                          color={
                            signal.urgency === 'HIGH'
                              ? 'error'
                              : signal.urgency === 'MEDIUM'
                                ? 'warning'
                                : 'info'
                          }
                          size="small"
                        />
                      </Stack>
                      <Chip
                        icon={<AccessTime />}
                        label={signal.timeframe}
                        variant="outlined"
                        size="small"
                      />
                    </Stack>

                    {/* Strike and Expiry */}
                    <Stack direction="row" spacing={2}>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Strike Price
                        </Typography>
                        <Typography variant="body1" fontWeight="bold">
                          ${signal.strike.toFixed(2)}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Expiry
                        </Typography>
                        <Typography variant="body1" fontWeight="bold">
                          {new Date(signal.expiry).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </Stack>

                    {/* Entry and Targets */}
                    <Grid container spacing={2}>
                      <Grid item xs={4}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Entry
                          </Typography>
                          <Typography variant="body1" color="primary" fontWeight="bold">
                            ${signal.entryPrice.toFixed(2)}
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Target
                          </Typography>
                          <Typography variant="body1" color="success.main" fontWeight="bold">
                            ${signal.targetPrice.toFixed(2)}
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Stop Loss
                          </Typography>
                          <Typography variant="body1" color="error.main" fontWeight="bold">
                            ${signal.stopLoss.toFixed(2)}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Confidence */}
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2" color="text.secondary">
                          AI Confidence
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {(signal.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={signal.confidence * 100}
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          mt: 1,
                          bgcolor: theme.palette.grey[200],
                          '& .MuiLinearProgress-bar': {
                            bgcolor:
                              signal.confidence > 0.7
                                ? 'success.main'
                                : signal.confidence > 0.5
                                  ? 'warning.main'
                                  : 'error.main',
                          },
                        }}
                      />
                    </Box>

                    {/* Patterns */}
                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Detected Patterns
                      </Typography>
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {signal.patterns.map((pattern, index) => (
                          <Chip
                            key={index}
                            icon={<Timeline />}
                            label={pattern}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Stack>
                    </Box>

                    {/* Reasoning */}
                    <Typography variant="body2" color="text.secondary">
                      {signal.reasoning}
                    </Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};
