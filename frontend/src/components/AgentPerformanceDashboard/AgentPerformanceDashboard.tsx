import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  useTheme,
  alpha,
  Stack,
  Alert,
  CircularProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { motion } from 'framer-motion';
import logger from '../../services/logger';


// Styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  backgroundColor: theme.palette.background.default,
  minHeight: '100vh',
}));

const MetricCard = styled(Card)(({ theme }) => ({
  height: '100%',
  transition: 'all 0.3s ease',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[4],
    borderColor: theme.palette.primary.main,
  },
}));

const AgentCard = styled(Card)(({ theme }) => ({
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 4,
    background: (props: any) => {
      const accuracy = props.accuracy || 0;
      if (accuracy >= 0.7) return theme.palette.success.main;
      if (accuracy >= 0.5) return theme.palette.warning.main;
      return theme.palette.error.main;
    },
  },
}));

// Types
interface AgentData {
  id: string;
  name: string;
  type: string;
  accuracy: number;
  winRate: number;
  totalSignals: number;
  profitLoss: number;
  sharpeRatio: number;
  consensusWeight: number;
  status: 'active' | 'inactive' | 'error';
  lastSignalTime: string;
  performance30d: number[];
}

interface AgentMetrics {
  totalAgents: number;
  activeAgents: number;
  averageAccuracy: number;
  totalProfit: number;
  bestPerformer: string;
  worstPerformer: string;
}

// Mock data generator
const generateMockAgentData = (): AgentData[] => {
  const agentTypes = ['RSI', 'MACD', 'Sentiment', 'Volume', 'Momentum', 'ML', 'Options', 'News'];
  return agentTypes.map((type, index) => ({
    id: `agent_${index}`,
    name: `${type}_Agent`,
    type: type.toLowerCase(),
    accuracy: 0.5 + Math.random() * 0.4,
    winRate: 0.4 + Math.random() * 0.4,
    totalSignals: Math.floor(100 + Math.random() * 400),
    profitLoss: -5000 + Math.random() * 15000,
    sharpeRatio: -0.5 + Math.random() * 2.5,
    consensusWeight: 0.5 + Math.random() * 1.5,
    status: Math.random() > 0.1 ? 'active' : Math.random() > 0.5 ? 'inactive' : 'error',
    lastSignalTime: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    performance30d: Array.from({ length: 30 }, () => -100 + Math.random() * 200),
  }));
};

const AgentPerformanceDashboard: React.FC = () => {
  const theme = useTheme();
  const [agents, setAgents] = useState<AgentData[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('30d');
  const [selectedAgent, setSelectedAgent] = useState<AgentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Fetch agent data
  useEffect(() => {
    fetchAgentData();
  }, [selectedTimeframe]);

  const fetchAgentData = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      const mockData = generateMockAgentData();
      setAgents(mockData);
    } catch (error) {
      logger.error('Failed to fetch agent data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate metrics
  const metrics = useMemo<AgentMetrics>(() => {
    if (agents.length === 0) {
      return {
        totalAgents: 0,
        activeAgents: 0,
        averageAccuracy: 0,
        totalProfit: 0,
        bestPerformer: 'N/A',
        worstPerformer: 'N/A',
      };
    }

    const activeAgents = agents.filter(a => a.status === 'active');
    const sortedByProfit = [...agents].sort((a, b) => b.profitLoss - a.profitLoss);

    return {
      totalAgents: agents.length,
      activeAgents: activeAgents.length,
      averageAccuracy: agents.reduce((sum, a) => sum + a.accuracy, 0) / agents.length,
      totalProfit: agents.reduce((sum, a) => sum + a.profitLoss, 0),
      bestPerformer: sortedByProfit[0]?.name || 'N/A',
      worstPerformer: sortedByProfit[sortedByProfit.length - 1]?.name || 'N/A',
    };
  }, [agents]);

  // Chart data
  const performanceChartData = {
    labels: agents.map(a => a.name.replace('_Agent', '')),
    datasets: [
      {
        label: 'Accuracy',
        data: agents.map(a => a.accuracy * 100),
        backgroundColor: alpha(theme.palette.primary.main, 0.6),
        borderColor: theme.palette.primary.main,
        borderWidth: 2,
      },
      {
        label: 'Win Rate',
        data: agents.map(a => a.winRate * 100),
        backgroundColor: alpha(theme.palette.success.main, 0.6),
        borderColor: theme.palette.success.main,
        borderWidth: 2,
      },
    ],
  };

  const weightDistributionData = {
    labels: agents.map(a => a.name.replace('_Agent', '')),
    datasets: [
      {
        data: agents.map(a => a.consensusWeight),
        backgroundColor: agents.map((_, i) =>
          `hsl(${(i * 360) / agents.length}, 70%, 60%)`
        ),
        borderWidth: 1,
        borderColor: theme.palette.background.paper,
      },
    ],
  };

  const handleAgentClick = (agent: AgentData) => {
    setSelectedAgent(agent);
    setDetailsOpen(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <DashboardContainer>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Agent Performance Dashboard
        </Typography>
        <Stack direction="row" spacing={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={selectedTimeframe}
              label="Timeframe"
              onChange={(e) => setSelectedTimeframe(e.target.value)}
            >
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
              <MenuItem value="30d">30 Days</MenuItem>
              <MenuItem value="90d">90 Days</MenuItem>
            </Select>
          </FormControl>
          <IconButton onClick={fetchAgentData} color="primary">
            <RefreshIcon />
          </IconButton>
        </Stack>
      </Box>

      {/* Metrics Overview */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <PsychologyIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Total Agents
                </Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">
                {metrics.totalAgents}
              </Typography>
              <Chip
                label={`${metrics.activeAgents} Active`}
                size="small"
                color="success"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <SpeedIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Avg Accuracy
                </Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">
                {(metrics.averageAccuracy * 100).toFixed(1)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.averageAccuracy * 100}
                sx={{ mt: 1, height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                {metrics.totalProfit >= 0 ? (
                  <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                ) : (
                  <TrendingDownIcon color="error" sx={{ mr: 1 }} />
                )}
                <Typography variant="subtitle2" color="text.secondary">
                  Total P&L
                </Typography>
              </Box>
              <Typography
                variant="h4"
                fontWeight="bold"
                color={metrics.totalProfit >= 0 ? 'success.main' : 'error.main'}
              >
                {formatCurrency(metrics.totalProfit)}
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <AnalyticsIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  Best Performer
                </Typography>
              </Box>
              <Typography variant="h6" fontWeight="bold">
                {metrics.bestPerformer.replace('_Agent', '')}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                vs {metrics.worstPerformer.replace('_Agent', '')} (worst)
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Agent Performance Comparison
              </Typography>
              <Box height={300}>
                <Bar
                  data={performanceChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 100,
                      },
                    },
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Consensus Weight Distribution
              </Typography>
              <Box height={300}>
                <Doughnut
                  data={weightDistributionData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                      },
                    },
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Agent Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Agent Details
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Agent</TableCell>
                  <TableCell align="center">Status</TableCell>
                  <TableCell align="right">Accuracy</TableCell>
                  <TableCell align="right">Win Rate</TableCell>
                  <TableCell align="right">Signals</TableCell>
                  <TableCell align="right">P&L</TableCell>
                  <TableCell align="right">Sharpe</TableCell>
                  <TableCell align="right">Weight</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {agents.map((agent) => (
                  <TableRow
                    key={agent.id}
                    hover
                    sx={{ cursor: 'pointer' }}
                    onClick={() => handleAgentClick(agent)}
                  >
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {agent.name}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={agent.status}
                        size="small"
                        color={getStatusColor(agent.status) as any}
                      />
                    </TableCell>
                    <TableCell align="right">
                      {(agent.accuracy * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="right">
                      {(agent.winRate * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="right">{agent.totalSignals}</TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: agent.profitLoss >= 0 ? 'success.main' : 'error.main',
                        fontWeight: 'medium'
                      }}
                    >
                      {formatCurrency(agent.profitLoss)}
                    </TableCell>
                    <TableCell align="right">
                      {agent.sharpeRatio.toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      {agent.consensusWeight.toFixed(2)}
                    </TableCell>
                    <TableCell align="center">
                      <IconButton size="small" onClick={(e) => {
                        e.stopPropagation();
                        handleAgentClick(agent);
                      }}>
                        <InfoIcon />
                      </IconButton>
                      <IconButton size="small">
                        <SettingsIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Agent Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedAgent && (
          <>
            <DialogTitle>
              {selectedAgent.name} - Detailed Performance
            </DialogTitle>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12}>
                  <Alert severity={selectedAgent.accuracy >= 0.7 ? 'success' : 'warning'}>
                    This agent has {selectedAgent.accuracy >= 0.7 ? 'excellent' : 'moderate'} performance
                    with {(selectedAgent.accuracy * 100).toFixed(1)}% accuracy over {selectedAgent.totalSignals} signals.
                  </Alert>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    30-Day Performance
                  </Typography>
                  <Box height={200}>
                    <Line
                      data={{
                        labels: Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`),
                        datasets: [{
                          label: 'Daily P&L',
                          data: selectedAgent.performance30d,
                          borderColor: theme.palette.primary.main,
                          backgroundColor: alpha(theme.palette.primary.main, 0.1),
                          fill: true,
                        }],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            display: false,
                          },
                        },
                      }}
                    />
                  </Box>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDetailsOpen(false)}>Close</Button>
              <Button variant="contained" startIcon={<SettingsIcon />}>
                Configure Agent
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </DashboardContainer>
  );
};

export default AgentPerformanceDashboard;
