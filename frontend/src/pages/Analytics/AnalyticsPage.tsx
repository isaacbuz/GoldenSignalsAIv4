/**
 * Analytics Page - Institutional Grade Analytics Dashboard
 * 
 * Comprehensive analytics and reporting dashboard with advanced metrics
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  Grid,
  Tabs,
  Tab,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Alert,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Analytics as AnalyticsIcon,
  Assessment,
  Timeline,
  PieChart,
  BarChart,
  ShowChart,
  Download,
  Refresh,
  FilterList,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, AreaChart, Area, BarChart as RechartsBarChart, Bar, PieChart as RechartsPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import { apiClient } from '../../services/api';
import { useAppStore } from '../../store';
import { formatPrice, formatPercentage } from '../../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

// Mock data for demonstration
const performanceData = [
  { date: '2024-01', returns: 8.5, benchmark: 5.2, signals: 145 },
  { date: '2024-02', returns: 12.3, benchmark: 7.1, signals: 167 },
  { date: '2024-03', returns: -2.1, benchmark: -1.8, signals: 134 },
  { date: '2024-04', returns: 15.7, benchmark: 9.4, signals: 189 },
  { date: '2024-05', returns: 9.2, benchmark: 6.8, signals: 156 },
  { date: '2024-06', returns: 18.4, benchmark: 11.2, signals: 203 },
];

const agentPerformanceData = [
  { name: 'GammaExposureAgent', accuracy: 72.5, signals: 89, pnl: 15420, sharpe: 2.1 },
  { name: 'SkewAgent', accuracy: 68.3, signals: 76, pnl: 12890, sharpe: 1.8 },
  { name: 'IVRankAgent', accuracy: 71.2, signals: 82, pnl: 14230, sharpe: 1.9 },
  { name: 'RegimeAgent', accuracy: 75.8, signals: 94, pnl: 18650, sharpe: 2.3 },
  { name: 'ETFArbAgent', accuracy: 69.4, signals: 67, pnl: 11340, sharpe: 1.7 },
  { name: 'MetaConsensusAgent', accuracy: 78.2, signals: 156, pnl: 24580, sharpe: 2.6 },
];

const riskMetrics = [
  { metric: 'Sharpe Ratio', value: 2.14, benchmark: 1.45, status: 'excellent' },
  { metric: 'Max Drawdown', value: -8.2, benchmark: -12.5, status: 'good' },
  { metric: 'Volatility', value: 14.8, benchmark: 18.3, status: 'good' },
  { metric: 'Beta', value: 0.85, benchmark: 1.0, status: 'neutral' },
  { metric: 'Alpha', value: 5.2, benchmark: 0.0, status: 'excellent' },
  { metric: 'Information Ratio', value: 1.68, benchmark: 0.8, status: 'excellent' },
];

const sectorAllocation = [
  { name: 'Technology', value: 28.5, color: '#00e676' },
  { name: 'Healthcare', value: 18.2, color: '#ff1744' },
  { name: 'Financial', value: 15.8, color: '#ffab00' },
  { name: 'Consumer', value: 12.4, color: '#00b0ff' },
  { name: 'Energy', value: 10.3, color: '#9c27b0' },
  { name: 'Industrial', value: 8.9, color: '#ff5722' },
  { name: 'Other', value: 5.9, color: '#607d8b' },
];

export default function AnalyticsPage() {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('6M');
  const [selectedMetric, setSelectedMetric] = useState('returns');
  const { agentPerformance } = useAppStore();

  // Fetch analytics data
  const { data: analyticsData, isLoading, error, refetch } = useQuery({
    queryKey: ['analytics', timeRange],
    queryFn: () => apiClient.getAnalytics({ timeRange }),
    refetchInterval: 30000,
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'success';
      case 'good': return 'info';
      case 'warning': return 'warning';
      case 'poor': return 'error';
      default: return 'default';
    }
  };

  const MetricCard = ({ title, value, change, icon, color = 'primary' }: any) => (
    <Card sx={{ p: 3, height: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" color="text.secondary">
          {title}
        </Typography>
        <Box sx={{ color: `${color}.main` }}>
          {icon}
        </Box>
      </Box>
      <Typography variant="h4" sx={{ mb: 1, fontWeight: 700 }}>
        {value}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {change > 0 ? (
          <TrendingUp sx={{ color: 'success.main', mr: 0.5 }} />
        ) : (
          <TrendingDown sx={{ color: 'error.main', mr: 0.5 }} />
        )}
        <Typography
          variant="body2"
          sx={{
            color: change > 0 ? 'success.main' : 'error.main',
            fontWeight: 600,
          }}
        >
          {change > 0 ? '+' : ''}{change}%
        </Typography>
      </Box>
    </Card>
  );

  if (error) {
    return (
      <Box sx={{ flexGrow: 1 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          Failed to load analytics data. Please try again.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Analytics Dashboard
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Comprehensive performance analytics and risk metrics
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="1M">1 Month</MenuItem>
              <MenuItem value="3M">3 Months</MenuItem>
              <MenuItem value="6M">6 Months</MenuItem>
              <MenuItem value="1Y">1 Year</MenuItem>
              <MenuItem value="YTD">YTD</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Download />}
          >
            Export
          </Button>
        </Stack>
      </Box>

      {isLoading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Return"
            value="24.8%"
            change={5.2}
            icon={<TrendingUp />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Sharpe Ratio"
            value="2.14"
            change={8.1}
            icon={<Assessment />}
            color="info"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Signals"
            value="47"
            change={-3.2}
            icon={<ShowChart />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Win Rate"
            value="72.5%"
            change={2.8}
            icon={<PieChart />}
            color="primary"
          />
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Performance" icon={<Timeline />} />
          <Tab label="Risk Analysis" icon={<Assessment />} />
          <Tab label="Agent Performance" icon={<AnalyticsIcon />} />
          <Tab label="Portfolio Analysis" icon={<PieChart />} />
        </Tabs>

        {/* Performance Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Performance vs Benchmark
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="returns"
                      stackId="1"
                      stroke="#00e676"
                      fill="#00e676"
                      fillOpacity={0.3}
                      name="Strategy Returns (%)"
                    />
                    <Area
                      type="monotone"
                      dataKey="benchmark"
                      stackId="2"
                      stroke="#ff1744"
                      fill="#ff1744"
                      fillOpacity={0.3}
                      name="Benchmark (%)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </Grid>
            <Grid item xs={12} lg={4}>
              <Card sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Monthly Statistics
                </Typography>
                <Stack spacing={2}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Best Month
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      +18.4% (June 2024)
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Worst Month
                    </Typography>
                    <Typography variant="h6" color="error.main">
                      -2.1% (March 2024)
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Average Monthly Return
                    </Typography>
                    <Typography variant="h6">
                      +10.3%
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Win Rate
                    </Typography>
                    <Typography variant="h6">
                      83.3% (5/6 months)
                    </Typography>
                  </Box>
                </Stack>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Risk Analysis Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Risk Metrics Comparison
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Metric</TableCell>
                        <TableCell align="right">Strategy</TableCell>
                        <TableCell align="right">Benchmark</TableCell>
                        <TableCell align="center">Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {riskMetrics.map((metric) => (
                        <TableRow key={metric.metric}>
                          <TableCell component="th" scope="row">
                            {metric.metric}
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              sx={{
                                fontWeight: 600,
                                color: metric.value > metric.benchmark ? 'success.main' : 'text.primary'
                              }}
                            >
                              {typeof metric.value === 'number' && metric.value < 0 
                                ? `${metric.value}%` 
                                : typeof metric.value === 'number' && metric.metric.includes('Ratio')
                                ? metric.value.toFixed(2)
                                : `${metric.value}%`
                              }
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            {typeof metric.benchmark === 'number' && metric.benchmark < 0 
                              ? `${metric.benchmark}%` 
                              : typeof metric.benchmark === 'number' && metric.metric.includes('Ratio')
                              ? metric.benchmark.toFixed(2)
                              : `${metric.benchmark}%`
                            }
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={metric.status}
                              color={getStatusColor(metric.status) as any}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Risk Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Pie
                      data={[
                        { name: 'Low Risk', value: 45, fill: '#00e676' },
                        { name: 'Medium Risk', value: 35, fill: '#ffab00' },
                        { name: 'High Risk', value: 20, fill: '#ff1744' },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {[
                        { name: 'Low Risk', value: 45, fill: '#00e676' },
                        { name: 'Medium Risk', value: 35, fill: '#ffab00' },
                        { name: 'High Risk', value: 20, fill: '#ff1744' },
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Agent Performance Tab */}
        <TabPanel value={tabValue} index={2}>
          <Card sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              AI Agent Performance Analysis
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Agent Name</TableCell>
                    <TableCell align="right">Accuracy</TableCell>
                    <TableCell align="right">Total Signals</TableCell>
                    <TableCell align="right">P&L</TableCell>
                    <TableCell align="right">Sharpe Ratio</TableCell>
                    <TableCell align="center">Performance</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {agentPerformanceData.map((agent) => (
                    <TableRow key={agent.name}>
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {agent.name}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          sx={{
                            color: agent.accuracy > 70 ? 'success.main' : 
                                   agent.accuracy > 60 ? 'warning.main' : 'error.main',
                            fontWeight: 600,
                          }}
                        >
                          {agent.accuracy}%
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{agent.signals}</TableCell>
                      <TableCell align="right">
                        <Typography
                          sx={{
                            color: agent.pnl > 0 ? 'success.main' : 'error.main',
                            fontWeight: 600,
                          }}
                        >
                          {formatPrice(agent.pnl)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{agent.sharpe}</TableCell>
                      <TableCell align="center">
                        <Chip
                          label={
                            agent.accuracy > 75 ? 'Excellent' :
                            agent.accuracy > 70 ? 'Good' :
                            agent.accuracy > 60 ? 'Average' : 'Poor'
                          }
                          color={
                            agent.accuracy > 75 ? 'success' :
                            agent.accuracy > 70 ? 'info' :
                            agent.accuracy > 60 ? 'warning' : 'error'
                          }
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Card>
        </TabPanel>

        {/* Portfolio Analysis Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Sector Allocation
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Pie
                      data={sectorAllocation}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      dataKey="value"
                      label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(1)}%`}
                    >
                      {sectorAllocation.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Signal Distribution by Type
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsBarChart
                    data={[
                      { type: 'BUY', count: 156, accuracy: 74 },
                      { type: 'SELL', count: 89, accuracy: 68 },
                      { type: 'HOLD', count: 234, accuracy: 82 },
                    ]}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="count" fill="#00e676" name="Signal Count" />
                  </RechartsBarChart>
                </ResponsiveContainer>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
} 