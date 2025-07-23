import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Stack,
  Avatar,
  Divider,
  Paper,
  alpha,
  useTheme,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AccessTime as AccessTimeIcon,
  GpsFixed as TargetIcon,
  Security as SecurityIcon,
  Psychology as PsychologyIcon,
  ShowChart as ShowChartIcon,
  VolumeUp as VolumeUpIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { customStyles } from '../../theme/enhancedTheme';

// Styled components using centralized theme
const SignalCard = styled(Card)(({ theme }) => ({
  height: '100%',
  ...customStyles.goldGradient,
  border: `1px solid ${alpha(theme.palette.primary.main, 0.15)}`,
  '& .MuiCardContent-root': {
    padding: theme.spacing(1.25),
  },
}));

const ConsensusCard = styled(Card)(({ theme }) => ({
  height: '100%',
  ...customStyles.successGradient,
  border: `1px solid ${alpha(theme.palette.success.main, 0.15)}`,
  '& .MuiCardContent-root': {
    padding: theme.spacing(1.25),
  },
}));

const ReasoningCard = styled(Card)(({ theme }) => ({
  height: '100%',
  ...customStyles.infoGradient,
  border: `1px solid ${alpha(theme.palette.info.main, 0.15)}`,
  '& .MuiCardContent-root': {
    padding: theme.spacing(1.25),
  },
}));

const MetricItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(1),
  borderRadius: theme.spacing(1),
  backgroundColor: alpha(theme.palette.background.default, 0.5),
}));

const ReasoningStep = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'flex-start',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(1),
  padding: theme.spacing(1),
  borderRadius: theme.spacing(1),
  backgroundColor: alpha(theme.palette.background.default, 0.3),
}));

interface SignalAnalysisPanelProps {
  signal: {
    symbol: string;
    action: 'BUY' | 'SELL' | 'HOLD';
    price: number;
    confidence: number;
    target: number;
    stopLoss: number;
    timestamp: Date;
    reasoning: string[];
    agents: {
      total: number;
      agreeing: number;
      disagreeing: number;
    };
  };
  isAnalyzing: boolean;
}

const SignalAnalysisPanel: React.FC<SignalAnalysisPanelProps> = ({
  signal,
  isAnalyzing,
}) => {
  const theme = useTheme();

  const getSignalIcon = (action: string) => {
    switch (action) {
      case 'BUY':
        return <TrendingUpIcon color="success" />;
      case 'SELL':
        return <TrendingDownIcon color="error" />;
      default:
        return <ShowChartIcon color="info" />;
    }
  };

  const getSignalColor = (action: string) => {
    switch (action) {
      case 'BUY':
        return 'success';
      case 'SELL':
        return 'error';
      default:
        return 'info';
    }
  };

  const getReasoningIcon = (reasoning: string) => {
    if (reasoning.includes('Technical')) return <ShowChartIcon />;
    if (reasoning.includes('Sentiment')) return <PsychologyIcon />;
    if (reasoning.includes('Volume')) return <VolumeUpIcon />;
    if (reasoning.includes('Momentum')) return <TimelineIcon />;
    return <ShowChartIcon />;
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ${diffMins % 60}m ago`;
    return 'Yesterday';
  };

  if (isAnalyzing) {
    return (
      <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100%">
        <LinearProgress sx={{ width: '60%', mb: 2 }} />
        <Typography variant="h6" color="primary" gutterBottom>
          Analyzing {signal.symbol}...
        </Typography>
        <Typography variant="body2" color="text.secondary">
          30 AI agents are evaluating market conditions
        </Typography>
      </Box>
    );
  }

  return (
    <Box height="100%" display="flex" flexDirection="column">
      <Typography variant="subtitle1" sx={{ ...customStyles.sectionHeader, mb: 1, fontSize: '0.85rem' }}>
        Signal Analysis & Agent Reasoning
      </Typography>

      <Grid container spacing={1.5} sx={{ flex: 1 }}>
        {/* Current Signal */}
        <Grid item xs={12} md={4}>
          <SignalCard>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="subtitle2">
                  Current Signal
                </Typography>
              </Box>

              <Stack spacing={1}>
                <Box display="flex" alignItems="center" gap={1}>
                  {getSignalIcon(signal.action)}
                  <Typography variant="h6" color={getSignalColor(signal.action)} sx={{ fontSize: '1rem' }}>
                    {signal.action}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    @ ${signal.price.toFixed(2)}
                  </Typography>
                </Box>

                <MetricItem>
                  <TargetIcon color="success" />
                  <Typography variant="body2">
                    Target: <strong>${signal.target.toFixed(2)}</strong>
                  </Typography>
                </MetricItem>

                <MetricItem>
                  <SecurityIcon color="error" />
                  <Typography variant="body2">
                    Stop Loss: <strong>${signal.stopLoss.toFixed(2)}</strong>
                  </Typography>
                </MetricItem>

                <MetricItem>
                  <AccessTimeIcon color="info" />
                  <Typography variant="body2">
                    {formatTimeAgo(signal.timestamp)}
                  </Typography>
                </MetricItem>

                <Box mt={2}>
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    Confidence Level
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={signal.confidence * 100}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="body2" fontWeight="bold" mt={1}>
                    {(signal.confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </SignalCard>
        </Grid>

        {/* Agent Consensus */}
        <Grid item xs={12} md={4}>
          <ConsensusCard>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="subtitle2" fontWeight="bold">
                  Agent Consensus
                </Typography>
              </Box>

              <Stack spacing={2}>
                <Box textAlign="center">
                  <Typography variant="h3" fontWeight="bold" color="success.main">
                    {signal.agents.total}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    AI Agents Analyzed
                  </Typography>
                </Box>

                <Divider />

                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Avatar sx={{ bgcolor: 'success.main', width: 32, height: 32 }}>
                      ✓
                    </Avatar>
                    <Box>
                      <Typography variant="h6" fontWeight="bold">
                        {signal.agents.agreeing}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Agree
                      </Typography>
                    </Box>
                  </Box>

                  <Box display="flex" alignItems="center" gap={1}>
                    <Avatar sx={{ bgcolor: 'error.main', width: 32, height: 32 }}>
                      ✗
                    </Avatar>
                    <Box>
                      <Typography variant="h6" fontWeight="bold">
                        {signal.agents.disagreeing}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Disagree
                      </Typography>
                    </Box>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    Agreement Rate
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(signal.agents.agreeing / signal.agents.total) * 100}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="body2" fontWeight="bold" mt={1}>
                    {((signal.agents.agreeing / signal.agents.total) * 100).toFixed(1)}%
                  </Typography>
                </Box>

                <Chip
                  label="Updating in real-time"
                  size="small"
                  color="info"
                  sx={{ alignSelf: 'center' }}
                />
              </Stack>
            </CardContent>
          </ConsensusCard>
        </Grid>

        {/* Reasoning Chain */}
        <Grid item xs={12} md={4}>
          <ReasoningCard>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="subtitle2" fontWeight="bold">
                  Reasoning Chain
                </Typography>
              </Box>

              <Stack spacing={1}>
                {signal.reasoning.map((reason, index) => (
                  <ReasoningStep key={index}>
                    <Avatar sx={{ bgcolor: 'info.main', width: 24, height: 24 }}>
                      {index + 1}
                    </Avatar>
                    <Box>
                      <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                        {getReasoningIcon(reason)}
                        <Typography variant="body2" fontWeight="bold">
                          {reason.split(':')[0]}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {reason.split(':')[1]?.trim()}
                      </Typography>
                    </Box>
                  </ReasoningStep>
                ))}
              </Stack>
            </CardContent>
          </ReasoningCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SignalAnalysisPanel;
