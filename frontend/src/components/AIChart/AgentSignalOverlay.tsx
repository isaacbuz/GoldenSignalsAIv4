/**
 * AgentSignalOverlay Component
 *
 * Displays comprehensive agent analysis results in a floating panel.
 * Shows individual agent signals, consensus voting, and final trading decision.
 *
 * Features:
 * - Individual agent signals with confidence levels
 * - Workflow decision with entry/exit parameters
 * - Risk assessment visualization
 * - Consensus voting breakdown
 * - Interactive close functionality
 *
 * The overlay appears after agent analysis completes and provides
 * traders with detailed insights into the AI decision-making process.
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  IconButton,
  LinearProgress,
  Tooltip,
  Divider,
  Grid,
  Alert,
  useTheme,
  alpha,
  Stack,
} from '@mui/material';
import {
  Close as CloseIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as PsychologyIcon,
  Speed as SpeedIcon,
  VolumeUp as VolumeUpIcon,
  ShowChart as ShowChartIcon,
  Timeline as TimelineIcon,
  CandlestickChart as CandlestickChartIcon,
  Insights as InsightsIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { AgentSignal, WorkflowDecision, AgentName } from '../../types/agent.types';

interface AgentSignalOverlayProps {
  /**
   * Collection of signals from all agents
   * Key: agent name, Value: agent signal data
   */
  agentSignals: Record<string, AgentSignal>;

  /**
   * Final workflow decision including trading parameters
   */
  workflowDecision: WorkflowDecision | null;

  /**
   * Callback to close the overlay
   */
  onClose: () => void;
}

const OverlayContainer = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  top: 20,
  right: 20,
  width: 400,
  maxHeight: '80vh',
  overflowY: 'auto',
  background: alpha(theme.palette.background.paper, 0.95),
  backdropFilter: 'blur(10px)',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
  boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.2)}`,
  zIndex: 1300,
}));

const AgentRow = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5),
  borderRadius: theme.shape.borderRadius,
  marginBottom: theme.spacing(1),
  background: alpha(theme.palette.background.default, 0.5),
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  transition: 'all 0.2s ease',
  '&:hover': {
    background: alpha(theme.palette.primary.main, 0.05),
    borderColor: alpha(theme.palette.primary.main, 0.2),
  },
}));

/**
 * Icon mapping for each agent type
 * Maps agent names to their representative icons
 */
const agentIcons: Record<string, React.ReactNode> = {
  [AgentName.RSI]: <SpeedIcon />,
  [AgentName.MACD]: <ShowChartIcon />,
  [AgentName.Volume]: <VolumeUpIcon />,
  [AgentName.Momentum]: <TrendingUpIcon />,
  [AgentName.Pattern]: <CandlestickChartIcon />,
  [AgentName.Sentiment]: <PsychologyIcon />,
  [AgentName.LSTM]: <TimelineIcon />,
  [AgentName.Options]: <InsightsIcon />,
  [AgentName.MarketRegime]: <TrendingUpIcon />,
};

const getSignalColor = (signal: string, theme: any) => {
  switch (signal.toLowerCase()) {
    case 'buy':
      return theme.palette.success.main;
    case 'sell':
      return theme.palette.error.main;
    default:
      return theme.palette.text.secondary;
  }
};

const getConfidenceIcon = (confidence: number) => {
  if (confidence >= 0.8) return <CheckCircleIcon color="success" />;
  if (confidence >= 0.6) return <WarningIcon color="warning" />;
  return <ErrorIcon color="error" />;
};

export const AgentSignalOverlay: React.FC<AgentSignalOverlayProps> = ({
  agentSignals,
  workflowDecision,
  onClose,
}) => {
  const theme = useTheme();

  const formatAgentName = (name: string) => {
    return name
      .replace(/_agent$/, '')
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <OverlayContainer>
      {/* Header */}
      <Box sx={{ p: 2, pb: 1, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PsychologyIcon color="primary" />
            AI Agent Analysis
          </Typography>
          <IconButton size="small" onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Workflow Decision */}
      {workflowDecision && (
        <Box sx={{ p: 2, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
          <Alert
            severity={workflowDecision.execute ?
              (workflowDecision.action === 'BUY' ? 'success' : 'error') :
              'info'
            }
            sx={{ mb: 2 }}
          >
            <Typography variant="subtitle2" fontWeight="bold">
              Decision: {workflowDecision.action}
              {workflowDecision.execute && ' (Execute)'}
            </Typography>
            <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
              {workflowDecision.reasoning}
            </Typography>
          </Alert>

          {workflowDecision.execute && (
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">Entry Price</Typography>
                <Typography variant="body2" fontWeight="bold">
                  ${workflowDecision.entry_price?.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">Position Size</Typography>
                <Typography variant="body2" fontWeight="bold">
                  {(workflowDecision.position_size * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">Stop Loss</Typography>
                <Typography variant="body2" color="error">
                  ${workflowDecision.stop_loss?.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">Take Profit</Typography>
                <Typography variant="body2" color="success">
                  ${workflowDecision.take_profit?.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Chip
                  label={`Risk: ${workflowDecision.risk_level}`}
                  size="small"
                  color={
                    workflowDecision.risk_level === 'LOW' ? 'success' :
                    workflowDecision.risk_level === 'MEDIUM' ? 'warning' : 'error'
                  }
                  sx={{ mt: 1 }}
                />
              </Grid>
            </Grid>
          )}
        </Box>
      )}

      {/* Agent Signals */}
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2 }}>
          Individual Agent Signals
        </Typography>

        {Object.entries(agentSignals || {}).map(([agentName, signal]) => (
          <AgentRow key={agentName}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box display="flex" alignItems="center" gap={1}>
                <Box sx={{ color: theme.palette.primary.main }}>
                  {agentIcons[agentName] || <InsightsIcon />}
                </Box>
                <Box>
                  <Typography variant="body2" fontWeight="500">
                    {formatAgentName(agentName)}
                  </Typography>
                  {signal.metadata?.reason && (
                    <Typography variant="caption" color="text.secondary">
                      {signal.metadata.reason}
                    </Typography>
                  )}
                </Box>
              </Box>

              <Box display="flex" alignItems="center" gap={1}>
                <Chip
                  label={signal.signal?.toUpperCase() || 'HOLD'}
                  size="small"
                  sx={{
                    backgroundColor: alpha(getSignalColor(signal.signal || 'hold', theme), 0.1),
                    color: getSignalColor(signal.signal || 'hold', theme),
                    fontWeight: 'bold',
                  }}
                />
                <Tooltip title={`Confidence: ${(signal.confidence * 100).toFixed(0)}%`}>
                  <Box>{getConfidenceIcon(signal.confidence)}</Box>
                </Tooltip>
              </Box>
            </Box>

            {/* Confidence Bar */}
            <Box sx={{ mt: 1 }}>
              <LinearProgress
                variant="determinate"
                value={signal.confidence * 100}
                sx={{
                  height: 4,
                  borderRadius: 2,
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getSignalColor(signal.signal || 'hold', theme),
                  },
                }}
              />
            </Box>
          </AgentRow>
        ))}
      </Box>

      {/* Consensus Summary */}
      {workflowDecision?.consensus && (
        <Box sx={{ p: 2, pt: 0 }}>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
            Consensus Votes
          </Typography>
          <Stack direction="row" spacing={1}>
            <Chip
              label={`Buy: ${(workflowDecision.consensus.weighted_votes?.buy * 100 || 0).toFixed(0)}%`}
              size="small"
              color="success"
              variant="outlined"
            />
            <Chip
              label={`Sell: ${(workflowDecision.consensus.weighted_votes?.sell * 100 || 0).toFixed(0)}%`}
              size="small"
              color="error"
              variant="outlined"
            />
            <Chip
              label={`Hold: ${(workflowDecision.consensus.weighted_votes?.hold * 100 || 0).toFixed(0)}%`}
              size="small"
              variant="outlined"
            />
          </Stack>
        </Box>
      )}
    </OverlayContainer>
  );
};
