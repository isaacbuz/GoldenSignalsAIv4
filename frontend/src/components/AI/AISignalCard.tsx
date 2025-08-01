/**
 * AI Signal Card
 * Presents AI-generated signals with authority and conviction
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Button,
  Chip,
  Stack,
  IconButton,
  Collapse,
  LinearProgress,
  useTheme,
  alpha,
  Tooltip
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Psychology,
  Timer,
  ExpandMore,
  BoltOutlined,
  CheckCircle,
  Warning,
  Bolt,
  Info,
  Notifications
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface AISignal {
  id: string;
  symbol: string;
  type: 'CALL' | 'PUT';
  strike: number;
  expiry: string;
  confidence: number;
  currentPrice: number;
  entryPrice: number;
  predictedMove: string;
  timeframe: string;
  reasoning: string;
  agentConsensus: {
    total: number;
    agreeing: number;
  };
  historicalAccuracy: number;
  urgency: 'CRITICAL' | 'HIGH' | 'MEDIUM';
  entryWindow: number; // minutes
  patterns: string[];
}

interface AISignalCardProps {
  signal: AISignal;
  onExecute: (signal: AISignal) => void;
  onDismiss: (id: string) => void;
}

const MotionCard = motion.create(Card);

export const AISignalCard: React.FC<AISignalCardProps> = ({
  signal,
  onExecute,
  onDismiss
}) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);
  const [timeLeft, setTimeLeft] = useState(signal.entryWindow * 60); // Convert to seconds

  const isCall = signal.type === 'CALL';
  const signalColor = isCall ? theme.palette.success.main : theme.palette.error.main;
  const Icon = isCall ? TrendingUp : TrendingDown;

  // Countdown timer for entry window
  useEffect(() => {
    if (timeLeft <= 0) return;

    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getUrgencyColor = () => {
    switch (signal.urgency) {
      case 'CRITICAL': return theme.palette.error.main;
      case 'HIGH': return theme.palette.warning.main;
      case 'MEDIUM': return theme.palette.info.main;
      default: return theme.palette.primary.main;
    }
  };

  const urgencyColor = {
    CRITICAL: theme.palette.error.main,
    HIGH: theme.palette.warning.main,
    MEDIUM: theme.palette.info.main,
  }[signal.urgency];

  return (
    <MotionCard
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      sx={{
        background: `linear-gradient(135deg, ${alpha(signalColor, 0.1)} 0%, ${alpha(theme.palette.background.paper, 0.9)} 100%)`,
        border: `1px solid ${alpha(signalColor, 0.2)}`,
        borderRadius: 2,
        position: 'relative',
        overflow: 'visible',
      }}
    >
      {/* Urgency Indicator */}
      <Box
        sx={{
          position: 'absolute',
          top: -4,
          right: 16,
          zIndex: 1,
        }}
      >
        <Chip
          icon={<Bolt sx={{ fontSize: 16 }} />}
          label={signal.urgency}
          size="small"
          sx={{
            backgroundColor: alpha(urgencyColor, 0.2),
            color: urgencyColor,
            fontWeight: 'bold',
            animation: signal.urgency === 'CRITICAL' ? 'pulse 2s infinite' : 'none',
            '@keyframes pulse': {
              '0%': { opacity: 1 },
              '50%': { opacity: 0.6 },
              '100%': { opacity: 1 },
            },
          }}
        />
      </Box>

      <CardContent>
        <Stack spacing={2}>
          {/* Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="h5" fontWeight="bold">
                {signal.symbol}
              </Typography>
              <Chip
                icon={isCall ? <TrendingUp /> : <TrendingDown />}
                label={signal.type}
                size="small"
                sx={{
                  backgroundColor: alpha(signalColor, 0.2),
                  color: signalColor,
                  fontWeight: 'bold',
                }}
              />
            </Stack>
            <Stack direction="row" spacing={1}>
              <Tooltip title="Set Alert">
                <IconButton size="small">
                  <Notifications />
                </IconButton>
              </Tooltip>
              <Tooltip title="View Analysis">
                <IconButton size="small">
                  <Info />
                </IconButton>
              </Tooltip>
            </Stack>
          </Stack>

          {/* Key Metrics */}
          <Stack direction="row" spacing={3}>
            <Box>
              <Typography variant="body2" color="text.secondary">Strike</Typography>
              <Typography variant="h6">${signal.strike}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Expiry</Typography>
              <Typography variant="h6">{signal.expiry}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Entry</Typography>
              <Typography variant="h6">${signal.entryPrice}</Typography>
            </Box>
          </Stack>

          {/* AI Confidence */}
          <Box>
            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Psychology sx={{ color: theme.palette.primary.main }} />
                <Typography variant="body2">AI Confidence</Typography>
              </Stack>
              <Typography variant="body2" fontWeight="bold">
                {signal.confidence}%
              </Typography>
            </Stack>
            <LinearProgress
              variant="determinate"
              value={signal.confidence}
              sx={{
                height: 8,
                borderRadius: 4,
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                '& .MuiLinearProgress-bar': {
                  backgroundColor: theme.palette.primary.main,
                },
              }}
            />
          </Box>

          {/* Agent Consensus */}
          <Box>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="text.secondary">
                Agent Consensus
              </Typography>
              <Typography variant="body2" fontWeight="bold">
                {signal.agentConsensus.agreeing}/{signal.agentConsensus.total} Agents
              </Typography>
            </Stack>
            <LinearProgress
              variant="determinate"
              value={(signal.agentConsensus.agreeing / signal.agentConsensus.total) * 100}
              sx={{
                height: 4,
                borderRadius: 2,
                backgroundColor: alpha(theme.palette.secondary.main, 0.1),
                '& .MuiLinearProgress-bar': {
                  backgroundColor: theme.palette.secondary.main,
                },
              }}
            />
          </Box>

          {/* Time Window */}
          <Stack direction="row" spacing={1} alignItems="center">
            <Timer sx={{ color: theme.palette.warning.main }} />
            <Typography variant="body2">
              Entry window: {signal.entryWindow} minutes
            </Typography>
          </Stack>

          {/* Patterns */}
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Detected Patterns
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {signal.patterns.map((pattern, index) => (
                <Chip
                  key={index}
                  label={pattern}
                  size="small"
                  sx={{
                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                    color: theme.palette.primary.main,
                  }}
                />
              ))}
            </Stack>
          </Box>

          {/* Action Button */}
          <Box
            sx={{
              mt: 2,
              p: 2,
              backgroundColor: alpha(signalColor, 0.1),
              borderRadius: 1,
              border: `1px dashed ${alpha(signalColor, 0.3)}`,
            }}
          >
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Expected Move
                </Typography>
                <Typography variant="h6" color={signalColor}>
                  {signal.predictedMove}
                </Typography>
              </Box>
              <Chip
                icon={<Bolt />}
                label="Execute Now"
                onClick={() => onExecute(signal)}
                sx={{
                  backgroundColor: signalColor,
                  color: 'white',
                  '&:hover': {
                    backgroundColor: alpha(signalColor, 0.8),
                  },
                }}
              />
            </Stack>
          </Box>
        </Stack>
      </CardContent>
    </MotionCard>
  );
};
