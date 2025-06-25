import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  LinearProgress,
  useTheme,
} from '@mui/material';
import {
  Dashboard,
  Refresh,
  Timeline,
  BubbleChart,
  GridView,
  AutoAwesome,
  TrendingUp,
  Speed,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';
import SignalCard from '../../components/Signals/SignalCard';
import AgentConsensusFlow from '../../components/Agents/AgentConsensusFlow';
import { AdvancedSignalChart } from '../../components/Charts/AdvancedSignalChart';
import { useWebSocket } from '../../hooks/useWebSocket';

const DashboardCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: '100%',
  position: 'relative',
  overflow: 'hidden',
}));

const MetricBox = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  background: 'rgba(255, 215, 0, 0.05)',
  border: '1px solid rgba(255, 215, 0, 0.2)',
}));

interface DashboardMetric {
  label: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'stable';
}

const HybridDashboard: React.FC = () => {
  const theme = useTheme();
  const [view, setView] = useState<'overview' | 'detailed' | 'flow'>('overview');
  const [timeframe, setTimeframe] = useState('1h');
  const [signals, setSignals] = useState([]);
  const [metrics, setMetrics] = useState<DashboardMetric[]>([]);
  const [loading, setLoading] = useState(false);

  // WebSocket subscription
  const { data: wsData } = useWebSocket('signals.all', {
    onMessage: (data) => {
      if (data.type === 'signal') {
        setSignals(prev => [data, ...prev].slice(0, 50));
      }
    }
  });

  useEffect(() => {
    // Load initial data
    loadDashboardData();
  }, [timeframe]);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Fetch dashboard metrics
      const metricsData: DashboardMetric[] = [
        { label: 'Active Signals', value: 127, change: 12, trend: 'up' },
        { label: 'Accuracy', value: '94.2%', change: 2.3, trend: 'up' },
        { label: 'Agent Consensus', value: '87%', change: -1.2, trend: 'down' },
        { label: 'Processing Speed', value: '42ms', trend: 'stable' },
      ];
      setMetrics(metricsData);
    } finally {
      setLoading(false);
    }
  };

  const renderOverviewView = () => (
    <Grid container spacing={3}>
      {/* Key Metrics */}
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {metrics.map((metric, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <MetricBox>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      {metric.label}
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#FFD700' }}>
                      {metric.value}
                    </Typography>
                  </Box>
                  {metric.trend && (
                    <TrendingUp
                      sx={{
                        color: metric.trend === 'up' ? '#4CAF50' : '#F44336',
                        transform: metric.trend === 'down' ? 'rotate(180deg)' : 'none',
                      }}
                    />
                  )}
                </Box>
                {metric.change && (
                  <Typography
                    variant="caption"
                    sx={{
                      color: metric.change > 0 ? '#4CAF50' : '#F44336',
                    }}
                  >
                    {metric.change > 0 ? '+' : ''}{metric.change}%
                  </Typography>
                )}
              </MetricBox>
            </Grid>
          ))}
        </Grid>
      </Grid>

      {/* Signal Chart */}
      <Grid item xs={12} lg={8}>
        <DashboardCard>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Signal Performance
            </Typography>
            <AdvancedSignalChart
              data={[]}
              type="timeline"
              height={400}
            />
          </CardContent>
        </DashboardCard>
      </Grid>

      {/* Recent Signals */}
      <Grid item xs={12} lg={4}>
        <DashboardCard>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Recent Signals
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
              {signals.slice(0, 5).map((signal, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <SignalCard signal={signal} compact />
                </Box>
              ))}
            </Box>
          </CardContent>
        </DashboardCard>
      </Grid>

      {/* Agent Status */}
      <Grid item xs={12}>
        <DashboardCard>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Agent Consensus Status
            </Typography>
            <AgentConsensusFlow height={300} />
          </CardContent>
        </DashboardCard>
      </Grid>
    </Grid>
  );

  const renderDetailedView = () => (
    <Grid container spacing={3}>
      {/* Implement detailed view with more granular data */}
      <Grid item xs={12}>
        <Typography variant="h6">Detailed Analytics View</Typography>
        {/* Add detailed components */}
      </Grid>
    </Grid>
  );

  const renderFlowView = () => (
    <Grid container spacing={3}>
      {/* Signal flow visualization */}
      <Grid item xs={12}>
        <Typography variant="h6">Signal Flow Visualization</Typography>
        {/* Add flow visualization */}
      </Grid>
    </Grid>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, ...utilityClasses.textGradient }}>
            Hybrid Intelligence Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Unified view of all signal intelligence systems
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <ToggleButtonGroup
            value={view}
            exclusive
            onChange={(_, newView) => newView && setView(newView)}
            size="small"
          >
            <ToggleButton value="overview">
              <Dashboard sx={{ mr: 1 }} /> Overview
            </ToggleButton>
            <ToggleButton value="detailed">
              <GridView sx={{ mr: 1 }} /> Detailed
            </ToggleButton>
            <ToggleButton value="flow">
              <Timeline sx={{ mr: 1 }} /> Flow
            </ToggleButton>
          </ToggleButtonGroup>

          <IconButton onClick={loadDashboardData} disabled={loading}>
            <Refresh sx={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          </IconButton>
        </Box>
      </Box>

      {/* Progress bar for loading */}
      {loading && (
        <LinearProgress
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 1,
          }}
        />
      )}

      {/* Content based on view */}
      {view === 'overview' && renderOverviewView()}
      {view === 'detailed' && renderDetailedView()}
      {view === 'flow' && renderFlowView()}
    </Box>
  );
};

export default HybridDashboard;
