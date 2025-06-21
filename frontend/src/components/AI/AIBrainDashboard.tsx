/**
 * AI Brain Dashboard
 * The command center showing GoldenSignalsAI's intelligence at work
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Stack,
  useTheme,
  alpha,
  Tooltip,
  IconButton,
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Psychology,
  Speed,
  TrendingUp,
  Analytics,
  Refresh,
  Info,
  Bolt,
  Timer,
} from '@mui/icons-material';

interface AIBrainDashboardProps {
  activeAgents: number;
  signalsProcessing: number;
  patternsDetected: number;
  confidenceThreshold: number;
  winRate: number;
  lastUpdate?: string;
  onRefresh?: () => void;
}

const MotionPaper = motion.create(Paper);

export const AIBrainDashboard: React.FC<AIBrainDashboardProps> = ({
  activeAgents = 19,
  signalsProcessing = 0,
  patternsDetected = 0,
  confidenceThreshold = 85,
  winRate = 87,
  lastUpdate,
  onRefresh,
}) => {
  const theme = useTheme();
  const [aiStatus, setAiStatus] = useState<'HUNTING' | 'ANALYZING' | 'SIGNAL_FOUND'>('HUNTING');
  const [processingData, setProcessingData] = useState<string[]>([]);

  // Simulate AI processing
  useEffect(() => {
    const statuses: Array<'HUNTING' | 'ANALYZING' | 'SIGNAL_FOUND'> = ['HUNTING', 'ANALYZING', 'SIGNAL_FOUND'];
    const interval = setInterval(() => {
      setAiStatus(statuses[Math.floor(Math.random() * statuses.length)]);
      
      // Simulate processing data stream
      const newData = [
        'Analyzing volume patterns...',
        'Detecting smart money flow...',
        'Calculating option Greeks...',
        'Scanning for divergences...',
        'Processing sentiment data...',
        'Identifying support levels...',
        'Evaluating risk/reward...',
        'Checking historical patterns...'
      ];
      setProcessingData(prev => [...newData.slice(0, Math.floor(Math.random() * 4) + 1)]);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (aiStatus) {
      case 'HUNTING': return theme.palette.primary.main;
      case 'ANALYZING': return theme.palette.info.main;
      case 'SIGNAL_FOUND': return theme.palette.success.main;
      default: return theme.palette.primary.main;
    }
  };

  return (
    <MotionPaper
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      sx={{
        background: `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)} 0%, ${alpha(getStatusColor(), 0.1)} 100%)`,
        border: `1px solid ${alpha(getStatusColor(), 0.2)}`,
        borderRadius: 3,
        p: 3,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Neural network background animation */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          opacity: 0.1,
          background: `radial-gradient(circle at 20% 80%, ${getStatusColor()} 0%, transparent 50%),
                       radial-gradient(circle at 80% 20%, ${getStatusColor()} 0%, transparent 50%)`,
          animation: 'pulse 4s ease-in-out infinite',
          '@keyframes pulse': {
            '0%, 100%': { opacity: 0.1 },
            '50%': { opacity: 0.2 }
          }
        }}
      />

      <Grid container spacing={3} sx={{ position: 'relative', zIndex: 1 }}>
        {/* AI Status Header */}
        <Grid item xs={12}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Stack direction="row" spacing={2} alignItems="center">
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 2,
                  backgroundColor: alpha(getStatusColor(), 0.2),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: `2px solid ${getStatusColor()}`,
                }}
              >
                <Psychology
                  sx={{
                    fontSize: 28,
                    color: getStatusColor(),
                    animation: aiStatus === 'ANALYZING' ? 'spin 2s linear infinite' : 'none',
                    '@keyframes spin': {
                      '0%': { transform: 'rotate(0deg)' },
                      '100%': { transform: 'rotate(360deg)' }
                    }
                  }}
                />
              </Box>
              <Box>
                <Typography variant="h5" fontWeight="bold">
                  AI Brain Status
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Chip
                    icon={<Bolt sx={{ fontSize: 16 }} />}
                    label={aiStatus.replace('_', ' ')}
                    size="small"
                    sx={{
                      backgroundColor: alpha(getStatusColor(), 0.2),
                      color: getStatusColor(),
                      fontWeight: 'bold',
                      animation: aiStatus === 'SIGNAL_FOUND' ? 'pulse 1s ease-in-out infinite' : 'none',
                      '@keyframes pulse': {
                        '0%, 100%': { opacity: 1 },
                        '50%': { opacity: 0.6 }
                      }
                    }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {activeAgents} agents active
                  </Typography>
                </Stack>
              </Box>
            </Stack>
            <Stack direction="row" spacing={1}>
              {lastUpdate && (
                <Tooltip title="Last Update">
                  <Chip
                    icon={<Timer sx={{ fontSize: 16 }} />}
                    label={lastUpdate}
                    size="small"
                    variant="outlined"
                  />
                </Tooltip>
              )}
              <Tooltip title="Refresh">
                <IconButton onClick={onRefresh} size="small">
                  <Refresh />
                </IconButton>
              </Tooltip>
            </Stack>
          </Stack>
        </Grid>

        {/* AI Metrics */}
        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              background: alpha(theme.palette.primary.main, 0.1),
              border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
              borderRadius: 2,
            }}
          >
            <Stack spacing={1}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Speed sx={{ fontSize: 20, color: theme.palette.primary.main }} />
                <Typography variant="body2" color="text.secondary">Processing</Typography>
              </Stack>
              <Typography variant="h4" fontWeight="bold">
                {signalsProcessing}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                signals/minute
              </Typography>
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              background: alpha(theme.palette.info.main, 0.1),
              border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
              borderRadius: 2,
            }}
          >
            <Stack spacing={1}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Analytics sx={{ fontSize: 20, color: theme.palette.info.main }} />
                <Typography variant="body2" color="text.secondary">Patterns Found</Typography>
              </Stack>
              <Typography variant="h4" fontWeight="bold">
                {patternsDetected}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                today
              </Typography>
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              background: alpha(theme.palette.success.main, 0.1),
              border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
              borderRadius: 2,
            }}
          >
            <Stack spacing={1}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <TrendingUp sx={{ fontSize: 20, color: theme.palette.success.main }} />
                <Typography variant="body2" color="text.secondary">Win Rate</Typography>
              </Stack>
              <Typography variant="h4" fontWeight="bold">
                {winRate}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                last 30 days
              </Typography>
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              background: alpha(theme.palette.warning.main, 0.1),
              border: `1px solid ${alpha(theme.palette.warning.main, 0.2)}`,
              borderRadius: 2,
            }}
          >
            <Stack spacing={1}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Psychology sx={{ fontSize: 20, color: theme.palette.warning.main }} />
                <Typography variant="body2" color="text.secondary">Confidence</Typography>
              </Stack>
              <Typography variant="h4" fontWeight="bold">
                {confidenceThreshold}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                minimum threshold
              </Typography>
            </Stack>
          </Paper>
        </Grid>

        {/* Processing Data Stream */}
        <Grid item xs={12}>
          <Paper
            sx={{
              p: 2,
              background: alpha(theme.palette.background.paper, 0.5),
              border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
              borderRadius: 2,
            }}
          >
            <Stack spacing={1}>
              <Typography variant="body2" color="text.secondary">
                Current Analysis
              </Typography>
              <AnimatePresence>
                {processingData.map((data, index) => (
                  <motion.div
                    key={data}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: getStatusColor(),
                          animation: 'pulse 2s infinite',
                        }}
                      />
                      <Typography variant="body2">{data}</Typography>
                    </Stack>
                  </motion.div>
                ))}
              </AnimatePresence>
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </MotionPaper>
  );
}; 