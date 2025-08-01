/**
 * AI Agents Page
 *
 * Monitor and manage AI trading agents with real-time performance metrics
 */

import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  LinearProgress,
  useTheme,
  alpha,
  Stack,
  Avatar,
  Divider,
  Container,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  SmartToy as AgentIcon,
  TrendingUp as PerformanceIcon,
  Speed as AccuracyIcon,
  Timeline as SignalsIcon,
  Psychology as AIIcon,
  Assessment as DashboardIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../../services/api';

// Agent Card Component
interface AgentCardProps {
  agent: any;
}

function AgentCard({ agent }: AgentCardProps) {
  const theme = useTheme();

  const getAgentColor = (type: string) => {
    switch (type) {
      case 'technical': return theme.palette.primary.main;
      case 'sentiment': return theme.palette.secondary.main;
      case 'momentum': return theme.palette.success.main;
      case 'reversion': return theme.palette.warning.main;
      case 'volume': return theme.palette.info.main;
      default: return theme.palette.grey[500];
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return theme.palette.success.main;
      case 'inactive': return theme.palette.error.main;
      case 'paused': return theme.palette.warning.main;
      default: return theme.palette.grey[500];
    }
  };

  const agentColor = getAgentColor(agent.type);

  return (
    <Card
      sx={{
        height: '100%',
        border: `1px solid ${alpha(agentColor, 0.3)}`,
        backgroundColor: alpha(agentColor, 0.05),
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 8px 24px ${alpha(agentColor, 0.2)}`,
        },
      }}
    >
      <CardContent sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar
              sx={{
                backgroundColor: agentColor,
                color: 'white',
                width: 48,
                height: 48,
              }}
            >
              <AIIcon />
            </Avatar>
            <Box>
              <Typography variant="h6" fontWeight={700}>
                {agent.name}
              </Typography>
              <Chip
                label={agent.status.toUpperCase()}
                size="small"
                sx={{
                  backgroundColor: getStatusColor(agent.status),
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                }}
              />
            </Box>
          </Box>
        </Box>

        {/* Description */}
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          {agent.description}
        </Typography>

        {/* Performance Metrics */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" fontWeight={700} color={agentColor}>
                {agent.accuracy.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Accuracy
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" fontWeight={700} color="text.primary">
                {agent.totalSignals}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Signals
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Progress Bars */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Success Rate
            </Typography>
            <Typography variant="body2" fontWeight={600}>
              {agent.correctSignals}/{agent.totalSignals}
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={agent.accuracy}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: alpha(agentColor, 0.2),
              '& .MuiLinearProgress-bar': {
                backgroundColor: agentColor,
              },
            }}
          />
        </Box>

        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Avg Confidence
            </Typography>
            <Typography variant="body2" fontWeight={600}>
              {agent.avgConfidence.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={agent.avgConfidence}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: alpha(theme.palette.info.main, 0.2),
              '& .MuiLinearProgress-bar': {
                backgroundColor: theme.palette.info.main,
              },
            }}
          />
        </Box>

        <Divider sx={{ mb: 2 }} />

        {/* Features */}
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          Key Features
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
          {agent.features.slice(0, 3).map((feature: string, index: number) => (
            <Chip
              key={index}
              label={feature}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.75rem' }}
            />
          ))}
          {agent.features.length > 3 && (
            <Chip
              label={`+${agent.features.length - 3} more`}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.75rem' }}
            />
          )}
        </Stack>

        {/* Footer */}
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            Last signal: {agent.lastSignal}
          </Typography>
          <Button
            variant="outlined"
            size="small"
            sx={{ borderColor: agentColor, color: agentColor }}
          >
            View Details
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
}

// Main Page Component
export default function AgentsPage() {
  const navigate = useNavigate();
  const { data: agents, isLoading, error } = useQuery({
    queryKey: ['agents'],
    queryFn: () => apiClient.getAgents(),
  });

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h4" fontWeight={700} gutterBottom>
          AI Trading Agents
        </Typography>
        <Grid container spacing={4}>
          {[...Array(6)].map((_, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Skeleton variant="rounded" height={350} />
            </Grid>
          ))}
        </Grid>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          Failed to load agent data. Please try again later.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2} sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h4" fontWeight={700}>
              AI Trading Agents
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Monitor and manage the fleet of autonomous agents driving the trading strategy.
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<DashboardIcon />}
            onClick={() => navigate('/agents/performance')}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
              },
            }}
          >
            Performance Dashboard
          </Button>
        </Box>
      </Stack>

      <Grid container spacing={4}>
        {agents && agents.length > 0 ? (
          agents.map((agent: any) => (
            <Grid item xs={12} md={6} lg={4} key={agent.name}>
              <AgentCard agent={agent} />
            </Grid>
          ))
        ) : (
          <Grid item xs={12}>
            <Typography variant="body1" color="text.secondary" align="center">
              No agents found.
            </Typography>
          </Grid>
        )}
      </Grid>
    </Container>
  );
}
