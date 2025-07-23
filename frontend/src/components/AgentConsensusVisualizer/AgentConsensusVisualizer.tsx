import React, { useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Avatar,
  Tooltip,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack,
  Divider,
  alpha,
  useTheme,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  SmartToy as SmartToyIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as PsychologyIcon,
  ShowChart as ShowChartIcon,
  VolumeUp as VolumeUpIcon,
  Timeline as TimelineIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Styled components
const ConsensusContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
}));

const CircularProgressContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginBottom: theme.spacing(0.75),
  minHeight: 60,
}));

const AgentGrid = styled(Grid)(({ theme }) => ({
  maxHeight: '150px',
  overflowY: 'auto',
  padding: theme.spacing(1),
}));

const AgentAvatar = styled(Avatar)<{ status: 'agree' | 'disagree' | 'neutral' }>(({ theme, status }) => {
  const colors = {
    agree: theme.palette.success.main,
    disagree: theme.palette.error.main,
    neutral: theme.palette.grey[400],
  };

  return {
    width: 24,
    height: 24,
    backgroundColor: colors[status],
    fontSize: '0.7rem',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    '&:hover': {
      transform: 'scale(1.1)',
      boxShadow: theme.shadows[4],
    },
  };
});

const MetricCard = styled(Card)(({ theme }) => ({
  height: '100%',
  background: alpha(theme.palette.background.default, 0.5),
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
}));

interface Agent {
  id: string;
  name: string;
  type: 'technical' | 'sentiment' | 'fundamental' | 'momentum' | 'volume';
  status: 'agree' | 'disagree' | 'neutral';
  confidence: number;
  reasoning: string;
  lastUpdate: Date;
}

interface AgentConsensusVisualizerProps {
  agents: {
    total: number;
    agreeing: number;
    disagreeing: number;
  };
  confidence: number;
}

const AgentConsensusVisualizer: React.FC<AgentConsensusVisualizerProps> = ({
  agents,
  confidence,
}) => {
  const theme = useTheme();
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // Mock detailed agent data
  const detailedAgents: Agent[] = [
    {
      id: '1',
      name: 'RSI Analyzer',
      type: 'technical',
      status: 'agree',
      confidence: 0.92,
      reasoning: 'RSI shows oversold conditions with bullish divergence',
      lastUpdate: new Date(),
    },
    {
      id: '2',
      name: 'MACD Tracker',
      type: 'technical',
      status: 'agree',
      confidence: 0.88,
      reasoning: 'MACD line crossing above signal line with increasing histogram',
      lastUpdate: new Date(),
    },
    {
      id: '3',
      name: 'Sentiment Scanner',
      type: 'sentiment',
      status: 'agree',
      confidence: 0.85,
      reasoning: 'Social media sentiment increased 45% in last 24h',
      lastUpdate: new Date(),
    },
    {
      id: '4',
      name: 'Volume Analyzer',
      type: 'volume',
      status: 'agree',
      confidence: 0.91,
      reasoning: 'Above-average volume with strong accumulation pattern',
      lastUpdate: new Date(),
    },
    {
      id: '5',
      name: 'News Sentiment',
      type: 'sentiment',
      status: 'disagree',
      confidence: 0.76,
      reasoning: 'Recent regulatory concerns may impact short-term price',
      lastUpdate: new Date(),
    },
    {
      id: '6',
      name: 'Options Flow',
      type: 'momentum',
      status: 'agree',
      confidence: 0.89,
      reasoning: 'Unusual call option activity suggests bullish positioning',
      lastUpdate: new Date(),
    },
    // Add more agents up to 30
  ];

  const getAgentIcon = (type: Agent['type']) => {
    switch (type) {
      case 'technical':
        return <ShowChartIcon />;
      case 'sentiment':
        return <PsychologyIcon />;
      case 'fundamental':
        return <TimelineIcon />;
      case 'momentum':
        return <TrendingUpIcon />;
      case 'volume':
        return <VolumeUpIcon />;
      default:
        return <SmartToyIcon />;
    }
  };

  const getTypeColor = (type: Agent['type']) => {
    switch (type) {
      case 'technical':
        return theme.palette.primary.main;
      case 'sentiment':
        return theme.palette.secondary.main;
      case 'fundamental':
        return theme.palette.info.main;
      case 'momentum':
        return theme.palette.success.main;
      case 'volume':
        return theme.palette.warning.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const agreementPercentage = (agents.agreeing / agents.total) * 100;

  return (
    <ConsensusContainer>
      {/* Circular Progress Indicator */}
      <CircularProgressContainer>
        <CircularProgress
          variant="determinate"
          value={agreementPercentage}
          size={60}
          thickness={5}
          sx={{
            color: agreementPercentage > 70 ? 'success.main' :
                   agreementPercentage > 50 ? 'warning.main' : 'error.main',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography variant="body1" fontWeight="bold">
            {agreementPercentage.toFixed(0)}%
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
            Agreement
          </Typography>
        </Box>
      </CircularProgressContainer>

      {/* Agent Statistics */}
      <Stack spacing={0.5} mb={1} sx={{ flex: '0 0 auto' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="caption" color="text.secondary">
            Total Agents
          </Typography>
          <Typography variant="body2" fontWeight="bold">
            {agents.total}
          </Typography>
        </Box>

        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                backgroundColor: 'success.main',
              }}
            />
            <Typography variant="caption" color="text.secondary">
              Agreeing
            </Typography>
          </Box>
          <Typography variant="body2" fontWeight="bold" color="success.main">
            {agents.agreeing}
          </Typography>
        </Box>

        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                backgroundColor: 'error.main',
              }}
            />
            <Typography variant="caption" color="text.secondary">
              Disagreeing
            </Typography>
          </Box>
          <Typography variant="body2" fontWeight="bold" color="error.main">
            {agents.disagreeing}
          </Typography>
        </Box>

        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="caption" color="text.secondary">
            Confidence
          </Typography>
          <Typography variant="body2" fontWeight="bold">
            {(confidence * 100).toFixed(1)}%
          </Typography>
        </Box>
      </Stack>

      <Divider sx={{ my: 1 }} />

      {/* Agent Grid Toggle */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
        <Typography variant="caption" color="text.secondary">
          Agent Details
        </Typography>
        <IconButton
          size="small"
          onClick={() => setShowDetails(!showDetails)}
          sx={{ padding: 0.5 }}
        >
          {showDetails ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>

      {/* Agent Details */}
      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Collapse in={showDetails} sx={{ flex: showDetails ? 1 : 0, overflow: 'hidden' }}>
          <Box sx={{ height: '100%', overflowY: 'auto', pr: 0.5 }}>
            <Stack spacing={0.5}>
            {detailedAgents.slice(0, 6).map((agent) => (
              <Box
                key={agent.id}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  p: 0.5,
                  borderRadius: 0.5,
                  backgroundColor: alpha(theme.palette.background.default, 0.5),
                  cursor: 'pointer',
                  minHeight: 36,
                }}
                onClick={() => setExpandedAgent(expandedAgent === agent.id ? null : agent.id)}
              >
                <Tooltip title={agent.name}>
                  <AgentAvatar status={agent.status} sx={{ flexShrink: 0 }}>
                    {getAgentIcon(agent.type)}
                  </AgentAvatar>
                </Tooltip>

                <Box flex={1} minWidth={0}>
                  <Typography variant="caption" fontWeight="bold" noWrap>
                    {agent.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.625rem' }}>
                    {agent.type} • {(agent.confidence * 100).toFixed(0)}%
                  </Typography>
                </Box>

                <Chip
                  label={agent.status}
                  size="small"
                  color={agent.status === 'agree' ? 'success' : 'error'}
                  sx={{
                    textTransform: 'capitalize',
                    height: 16,
                    fontSize: '0.625rem',
                    '& .MuiChip-label': { px: 0.5 }
                  }}
                />
              </Box>
            ))}
          </Stack>
        </Box>
      </Collapse>

        {!showDetails && (
          <>
            {/* Agent Type Distribution */}
            <Box mt={1}>
              <Typography variant="caption" color="text.secondary" mb={0.5} display="block">
                Agent Types
              </Typography>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                {['technical', 'sentiment', 'volume', 'momentum'].map((type) => (
                  <Chip
                    key={type}
                    label={type}
                    size="small"
                    sx={{
                      backgroundColor: alpha(getTypeColor(type as Agent['type']), 0.1),
                      color: getTypeColor(type as Agent['type']),
                      textTransform: 'capitalize',
                      height: 18,
                      fontSize: '0.625rem',
                      mb: 0.5,
                    }}
                  />
                ))}
              </Stack>
            </Box>

            {/* Live Status */}
            <Box display="flex" justifyContent="center" mt="auto" pt={1}>
              <Chip
                label="Live Analysis"
                size="small"
                color="success"
                sx={{ fontSize: '0.625rem', height: 18 }}
              />
            </Box>
          </>
        )}
      </Box>
    </ConsensusContainer>
  );
};

export default AgentConsensusVisualizer;
