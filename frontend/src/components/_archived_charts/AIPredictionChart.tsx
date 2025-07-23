/**
 * AI Prediction Chart - Visualizes AI's market predictions with confidence levels
 * Showcases the AI's analytical capabilities
 */

import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Stack,
  useTheme,
  alpha,
  Tooltip as MuiTooltip,
  IconButton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Psychology,
  Info,
  Timer,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { motion } from 'framer-motion';
import { Chip } from '@mui/material';

interface AIPredictionChartProps {
  symbol: string;
  currentPrice: number;
  prediction: {
    target: number;
    confidence: number;
    timeframe: string;
    upperBand: number;
    lowerBand: number;
    supportLevels: number[];
    resistanceLevels: number[];
  };
  historicalData: Array<{
    timestamp: string;
    price: number;
  }>;
  onTimeframeChange?: (timeframe: string) => void;
  predictions?: Array<{
    date: string;
    value: number;
  }>;
  confidence?: number;
  timeframe?: string;
  isLoading?: boolean;
  error?: string | null;
}

const MotionPaper = motion.create(Paper);

export const AIPredictionChart: React.FC<AIPredictionChartProps> = ({
  symbol,
  currentPrice,
  prediction,
  historicalData = [],
  onTimeframeChange,
  predictions = [],
  confidence = 0,
  timeframe = '1d',
  isLoading = false,
  error = null
}) => {
  const theme = useTheme();
  const isBullish = prediction.target > currentPrice;
  const signalColor = isBullish ? theme.palette.success.main : theme.palette.error.main;

  const chartData = useMemo(() => {
    const formattedData = [
      ...(Array.isArray(historicalData) ? historicalData.map(point => ({
        date: new Date(point.timestamp),
        value: point.price,
        type: 'Historical'
      })) : []),
      ...(Array.isArray(predictions) ? predictions.map(point => ({
        date: new Date(point.date),
        value: point.value,
        type: 'Prediction'
      })) : [])
    ];

    return formattedData.sort((a, b) => a.date.getTime() - b.date.getTime());
  }, [historicalData, predictions]);

  const yDomain = useMemo(() => {
    if (!chartData.length) return [0, 100];

    const values = chartData.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = (max - min) * 0.1;

    return [Math.max(0, min - padding), max + padding];
  }, [chartData]);

  return (
    <MotionPaper
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      sx={{
        p: 3,
        background: `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)} 0%, ${alpha(signalColor, 0.1)} 100%)`,
        border: `1px solid ${alpha(signalColor, 0.2)}`,
        borderRadius: 3,
      }}
    >
      <Stack spacing={3}>
        {/* Header */}
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Stack direction="row" spacing={2} alignItems="center">
            <Box
              sx={{
                width: 48,
                height: 48,
                borderRadius: 2,
                backgroundColor: alpha(signalColor, 0.2),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: `2px solid ${signalColor}`,
              }}
            >
              {isBullish ? (
                <TrendingUp sx={{ fontSize: 28, color: signalColor }} />
              ) : (
                <TrendingDown sx={{ fontSize: 28, color: signalColor }} />
              )}
            </Box>
            <Box>
              <Typography variant="h5" fontWeight="bold">
                {symbol} Prediction
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <Chip
                  icon={<Psychology sx={{ fontSize: 16 }} />}
                  label={`${confidence}% Confidence`}
                  size="small"
                  sx={{
                    backgroundColor: alpha(signalColor, 0.2),
                    color: signalColor,
                    fontWeight: 'bold',
                  }}
                />
                <Typography variant="body2" color="text.secondary">
                  {timeframe}
                </Typography>
              </Stack>
            </Box>
          </Stack>
          <Stack direction="row" spacing={1}>
            <MuiTooltip title="View Analysis">
              <IconButton size="small">
                <Info />
              </IconButton>
            </MuiTooltip>
          </Stack>
        </Stack>

        {/* Chart */}
        <Box sx={{ height: 400, width: '100%' }}>
          <ResponsiveContainer>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.1)} />
              <XAxis
                dataKey="date"
                stroke={theme.palette.text.secondary}
                tick={{ fill: theme.palette.text.secondary }}
              />
              <YAxis
                stroke={theme.palette.text.secondary}
                tick={{ fill: theme.palette.text.secondary }}
                domain={yDomain}
              />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: theme.palette.background.paper,
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  borderRadius: 8,
                }}
                labelStyle={{ color: theme.palette.text.primary }}
              />

              {/* Historical Price Line */}
              <Line
                type="monotone"
                dataKey="value"
                stroke={theme.palette.primary.main}
                strokeWidth={2}
                dot={false}
              />

              {/* Prediction Line */}
              <Line
                type="monotone"
                dataKey="value"
                stroke={signalColor}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
              />

              {/* Confidence Bands */}
              <ReferenceArea
                y1={prediction.upperBand}
                y2={prediction.lowerBand}
                fill={alpha(signalColor, 0.1)}
                stroke={alpha(signalColor, 0.2)}
              />

              {/* Support Levels */}
              {prediction.supportLevels.map((level, index) => (
                <ReferenceLine
                  key={`support-${index}`}
                  y={level}
                  stroke={theme.palette.success.main}
                  strokeDasharray="3 3"
                  label={{
                    value: `Support ${index + 1}`,
                    position: 'right',
                    fill: theme.palette.success.main,
                  }}
                />
              ))}

              {/* Resistance Levels */}
              {prediction.resistanceLevels.map((level, index) => (
                <ReferenceLine
                  key={`resistance-${index}`}
                  y={level}
                  stroke={theme.palette.error.main}
                  strokeDasharray="3 3"
                  label={{
                    value: `Resistance ${index + 1}`,
                    position: 'right',
                    fill: theme.palette.error.main,
                  }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Box>

        {/* Key Levels */}
        <Stack direction="row" spacing={3} justifyContent="center">
          <Box>
            <Typography variant="body2" color="text.secondary">Current Price</Typography>
            <Typography variant="h6">${currentPrice.toFixed(2)}</Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">Target Price</Typography>
            <Typography variant="h6" color={signalColor}>
              ${prediction.target.toFixed(2)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">Expected Move</Typography>
            <Typography variant="h6" color={signalColor}>
              {((prediction.target - currentPrice) / currentPrice * 100).toFixed(1)}%
            </Typography>
          </Box>
        </Stack>
      </Stack>
    </MotionPaper>
  );
};

export default AIPredictionChart;
