import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  TrendingUp,
  ShowChart,
  Assessment,
  CalendarMonth,
  Psychology,
  Speed,
  Timeline,
  DonutLarge,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

const StyledCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: '100%',
}));

const MetricBox = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  ...utilityClasses.glassmorphism,
  textAlign: 'center',
}));

const SignalAnalytics: React.FC = () => {
  const [timeframe, setTimeframe] = useState('7d');
  const [viewMode, setViewMode] = useState('overview');

  // Mock data
  const accuracyData = [
    { date: 'Mon', accuracy: 92.5, signals: 145 },
    { date: 'Tue', accuracy: 94.2, signals: 167 },
    { date: 'Wed', accuracy: 93.8, signals: 152 },
    { date: 'Thu', accuracy: 95.1, signals: 189 },
    { date: 'Fri', accuracy: 94.7, signals: 201 },
    { date: 'Sat', accuracy: 93.2, signals: 134 },
    { date: 'Sun', accuracy: 94.8, signals: 128 },
  ];

  const signalDistribution = [
    { name: 'BUY', value: 45, color: '#4CAF50' },
    { name: 'SELL', value: 30, color: '#F44336' },
    { name: 'HOLD', value: 25, color: '#FFA500' },
  ];

  const agentPerformance = [
    { agent: 'Sentiment', accuracy: 92, signals: 234, profit: 12.5 },
    { agent: 'Technical', accuracy: 94, signals: 289, profit: 15.7 },
    { agent: 'Flow', accuracy: 96, signals: 312, profit: 18.9 },
    { agent: 'Risk', accuracy: 91, signals: 198, profit: 8.3 },
    { agent: 'Regime', accuracy: 89, signals: 156, profit: 10.2 },
    { agent: 'Liquidity', accuracy: 93, signals: 201, profit: 14.1 },
  ];

  const radarData = [
    { metric: 'Accuracy', value: 94.2 },
    { metric: 'Speed', value: 92.5 },
    { metric: 'Coverage', value: 88.7 },
    { metric: 'Consistency', value: 91.3 },
    { metric: 'Risk Mgmt', value: 95.1 },
    { metric: 'Profit Factor', value: 89.8 },
  ];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
          Signal Analytics
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Comprehensive analysis of AI signal performance and accuracy
        </Typography>
      </Box>

      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Timeframe</InputLabel>
          <Select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} label="Timeframe">
            <MenuItem value="24h">24 Hours</MenuItem>
            <MenuItem value="7d">7 Days</MenuItem>
            <MenuItem value="30d">30 Days</MenuItem>
            <MenuItem value="90d">90 Days</MenuItem>
          </Select>
        </FormControl>

        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, newMode) => newMode && setViewMode(newMode)}
          size="small"
        >
          <ToggleButton value="overview">Overview</ToggleButton>
          <ToggleButton value="agents">Agents</ToggleButton>
          <ToggleButton value="patterns">Patterns</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox>
            <Speed sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>94.2%</Typography>
            <Typography variant="body2" color="text.secondary">Overall Accuracy</Typography>
            <Chip label="+2.1%" size="small" sx={{ mt: 1, color: '#4CAF50' }} />
          </MetricBox>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox>
            <ShowChart sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>1,247</Typography>
            <Typography variant="body2" color="text.secondary">Total Signals</Typography>
            <Chip label="+156" size="small" sx={{ mt: 1, color: '#4CAF50' }} />
          </MetricBox>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox>
            <Psychology sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>87.5%</Typography>
            <Typography variant="body2" color="text.secondary">Agent Consensus</Typography>
            <Chip label="Strong" size="small" sx={{ mt: 1 }} />
          </MetricBox>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricBox>
            <TrendingUp sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>1.84</Typography>
            <Typography variant="body2" color="text.secondary">Profit Factor</Typography>
            <Chip label="Excellent" size="small" sx={{ mt: 1, color: '#4CAF50' }} />
          </MetricBox>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Accuracy Trend */}
        <Grid item xs={12} md={8}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Accuracy Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="date" stroke="rgba(255,255,255,0.7)" />
                  <YAxis stroke="rgba(255,255,255,0.7)" domain={[85, 100]} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(10, 14, 39, 0.9)',
                      border: '1px solid rgba(255, 215, 0, 0.3)',
                      borderRadius: 8
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#FFD700"
                    strokeWidth={3}
                    dot={{ fill: '#FFD700', r: 6 }}
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="signals"
                    stroke="#4CAF50"
                    strokeWidth={2}
                    yAxisId="right"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </StyledCard>
        </Grid>

        {/* Signal Distribution */}
        <Grid item xs={12} md={4}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Signal Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={signalDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {signalDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
                {signalDistribution.map((item) => (
                  <Box key={item.name} sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        backgroundColor: item.color,
                        mr: 1,
                      }}
                    />
                    <Typography variant="caption">
                      {item.name}: {item.value}%
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </StyledCard>
        </Grid>

        {/* Agent Performance Table */}
        <Grid item xs={12}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Agent Performance
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Agent</TableCell>
                      <TableCell align="right">Accuracy</TableCell>
                      <TableCell align="right">Signals</TableCell>
                      <TableCell align="right">Profit %</TableCell>
                      <TableCell align="right">Performance</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {agentPerformance.map((agent) => (
                      <TableRow key={agent.agent}>
                        <TableCell>{agent.agent}</TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${agent.accuracy}%`}
                            size="small"
                            sx={{
                              backgroundColor:
                                agent.accuracy >= 95
                                  ? 'rgba(76, 175, 80, 0.1)'
                                  : agent.accuracy >= 90
                                  ? 'rgba(255, 215, 0, 0.1)'
                                  : 'rgba(255, 165, 0, 0.1)',
                              color:
                                agent.accuracy >= 95
                                  ? '#4CAF50'
                                  : agent.accuracy >= 90
                                  ? '#FFD700'
                                  : '#FFA500',
                            }}
                          />
                        </TableCell>
                        <TableCell align="right">{agent.signals}</TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            sx={{ color: agent.profit > 0 ? '#4CAF50' : '#F44336' }}
                          >
                            +{agent.profit}%
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <LinearProgress
                            variant="determinate"
                            value={agent.accuracy}
                            sx={{
                              height: 6,
                              borderRadius: 3,
                              backgroundColor: 'rgba(255, 255, 255, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor:
                                  agent.accuracy >= 95
                                    ? '#4CAF50'
                                    : agent.accuracy >= 90
                                    ? '#FFD700'
                                    : '#FFA500',
                              },
                            }}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </StyledCard>
        </Grid>

        {/* Performance Radar */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Performance Metrics
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255, 255, 255, 0.1)" />
                  <PolarAngleAxis dataKey="metric" stroke="rgba(255, 255, 255, 0.7)" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="rgba(255, 255, 255, 0.7)" />
                  <Radar
                    name="Performance"
                    dataKey="value"
                    stroke="#FFD700"
                    fill="#FFD700"
                    fillOpacity={0.3}
                  />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </StyledCard>
        </Grid>

        {/* Time Analysis */}
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Best Performance Times
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                    By Hour of Day
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip label="9-10 AM: 96.2%" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)', color: '#4CAF50' }} />
                    <Chip label="2-3 PM: 95.8%" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)', color: '#4CAF50' }} />
                    <Chip label="3-4 PM: 94.9%" sx={{ backgroundColor: 'rgba(255, 215, 0, 0.1)', color: '#FFD700' }} />
                  </Box>
                </Box>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                    By Day of Week
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip label="Tuesday: 95.1%" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)', color: '#4CAF50' }} />
                    <Chip label="Thursday: 94.8%" sx={{ backgroundColor: 'rgba(255, 215, 0, 0.1)', color: '#FFD700' }} />
                    <Chip label="Friday: 94.5%" sx={{ backgroundColor: 'rgba(255, 215, 0, 0.1)', color: '#FFD700' }} />
                  </Box>
                </Box>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                    By Market Condition
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip label="Trending: 96.7%" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)', color: '#4CAF50' }} />
                    <Chip label="Volatile: 91.2%" sx={{ backgroundColor: 'rgba(255, 165, 0, 0.1)', color: '#FFA500' }} />
                    <Chip label="Range: 93.8%" sx={{ backgroundColor: 'rgba(255, 215, 0, 0.1)', color: '#FFD700' }} />
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SignalAnalytics;
