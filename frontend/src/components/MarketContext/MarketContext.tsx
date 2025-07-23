import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Chip,
  Grid,
  LinearProgress,
  IconButton,
  Button,
  Divider,
  Avatar,
  Stack,
  Tooltip,
  Tab,
  Tabs,
  useTheme,
  alpha,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Newspaper as NewspaperIcon,
  Psychology as PsychologyIcon,
  ShowChart as ShowChartIcon,
  LocalFireDepartment as FireIcon,
  AccessTime as TimeIcon,
  OpenInNew as OpenInNewIcon,
  Refresh as RefreshIcon,
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import logger from '../../services/logger';


// Styled components
const ContextContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
  height: '100%',
}));

const NewsCard = styled(Card)(({ theme }) => ({
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[4],
    borderColor: theme.palette.primary.main,
  },
}));

const MetricCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: alpha(theme.palette.primary.main, 0.02),
  },
}));

const OptionsFlowCard = styled(Card)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(10px)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
}));

const SentimentIndicator = styled(Box)(({ theme }) => ({
  width: '100%',
  height: 8,
  borderRadius: 4,
  background: `linear-gradient(to right,
    ${theme.palette.error.main} 0%,
    ${theme.palette.warning.main} 35%,
    ${theme.palette.grey[500]} 50%,
    ${theme.palette.info.main} 65%,
    ${theme.palette.success.main} 100%)`,
  position: 'relative',
  marginTop: theme.spacing(1),
  marginBottom: theme.spacing(1),
}));

const SentimentMarker = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  transform: 'translate(-50%, -50%)',
  width: 16,
  height: 16,
  borderRadius: '50%',
  backgroundColor: theme.palette.background.paper,
  border: `2px solid ${theme.palette.primary.main}`,
  boxShadow: theme.shadows[2],
}));

// Types
interface NewsItem {
  id: string;
  title: string;
  source: string;
  time: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  impact: 'high' | 'medium' | 'low';
  url: string;
  summary?: string;
}

interface OptionsFlow {
  strike: number;
  expiry: string;
  type: 'call' | 'put';
  volume: number;
  openInterest: number;
  premium: number;
  flow: 'bullish' | 'bearish';
}

interface MarketMetric {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

interface MarketContextProps {
  symbol: string;
  onNewsClick?: (news: NewsItem) => void;
}

// Mock data generators
const generateMockNews = (): NewsItem[] => [
  {
    id: '1',
    title: 'Apple Reports Record Q4 Earnings, Beats Analyst Expectations',
    source: 'Reuters',
    time: '2 min ago',
    sentiment: 'positive',
    impact: 'high',
    url: '#',
    summary: 'Apple Inc. reported quarterly revenue of $123.9 billion, up 8% year over year, driven by strong iPhone sales.',
  },
  {
    id: '2',
    title: 'Analyst Upgrades AAPL to Strong Buy with $200 Price Target',
    source: 'Bloomberg',
    time: '15 min ago',
    sentiment: 'positive',
    impact: 'medium',
    url: '#',
    summary: 'Morgan Stanley raises Apple price target citing AI initiatives and Services growth momentum.',
  },
  {
    id: '3',
    title: 'Tech Sector Rotation Signals Detected in Options Market',
    source: 'MarketWatch',
    time: '1 hour ago',
    sentiment: 'neutral',
    impact: 'medium',
    url: '#',
    summary: 'Unusual options activity suggests institutional investors rotating into tech stocks.',
  },
  {
    id: '4',
    title: 'Apple Vision Pro Demand Exceeds Initial Projections',
    source: 'The Verge',
    time: '3 hours ago',
    sentiment: 'positive',
    impact: 'low',
    url: '#',
    summary: 'Supply chain sources indicate Apple increasing Vision Pro production targets for 2024.',
  },
];

const generateMockOptionsFlow = (): OptionsFlow[] => [
  {
    strike: 180,
    expiry: '12/15',
    type: 'call',
    volume: 25432,
    openInterest: 8123,
    premium: 2450000,
    flow: 'bullish',
  },
  {
    strike: 170,
    expiry: '12/15',
    type: 'put',
    volume: 3211,
    openInterest: 12456,
    premium: 320000,
    flow: 'bearish',
  },
  {
    strike: 185,
    expiry: '01/19',
    type: 'call',
    volume: 18765,
    openInterest: 5432,
    premium: 1870000,
    flow: 'bullish',
  },
  {
    strike: 175,
    expiry: '12/15',
    type: 'call',
    volume: 15234,
    openInterest: 9876,
    premium: 1523400,
    flow: 'bullish',
  },
];

const generateMarketMetrics = (): MarketMetric[] => [
  { symbol: 'SPY', name: 'S&P 500', price: 456.78, change: 8.23, changePercent: 1.84 },
  { symbol: 'QQQ', name: 'Nasdaq 100', price: 387.45, change: 7.12, changePercent: 1.87 },
  { symbol: 'DIA', name: 'Dow Jones', price: 378.90, change: 2.45, changePercent: 0.65 },
  { symbol: 'VIX', name: 'Volatility', price: 14.23, change: -0.87, changePercent: -5.76 },
];

const MarketContext: React.FC<MarketContextProps> = ({ symbol, onNewsClick }) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [optionsFlow, setOptionsFlow] = useState<OptionsFlow[]>([]);
  const [marketMetrics, setMarketMetrics] = useState<MarketMetric[]>([]);
  const [sentiment, setSentiment] = useState(75); // 0-100 scale
  const [loading, setLoading] = useState(true);
  const [aiSummary, setAiSummary] = useState('');

  useEffect(() => {
    fetchData();
  }, [symbol]);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1000));

      setNews(generateMockNews());
      setOptionsFlow(generateMockOptionsFlow());
      setMarketMetrics(generateMarketMetrics());
      setSentiment(75);
      setAiSummary(
        `Strong bullish momentum detected for ${symbol}. Recent earnings beat and positive analyst sentiment driving institutional buying. Technical breakout confirmed with high volume. Options flow heavily skewed bullish with 87% call volume.`
      );
    } catch (error) {
      logger.error('Error fetching market context:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return theme.palette.success.main;
      case 'negative':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return theme.palette.error.main;
      case 'medium':
        return theme.palette.warning.main;
      default:
        return theme.palette.info.main;
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

  const putCallRatio = optionsFlow.reduce((acc, flow) => {
    if (flow.type === 'put') acc.puts += flow.volume;
    else acc.calls += flow.volume;
    return acc;
  }, { puts: 0, calls: 0 });

  const optionsChart = {
    labels: ['Calls', 'Puts'],
    datasets: [{
      data: [putCallRatio.calls, putCallRatio.puts],
      backgroundColor: [
        alpha(theme.palette.success.main, 0.8),
        alpha(theme.palette.error.main, 0.8),
      ],
      borderWidth: 0,
    }],
  };

  const correlatedAssetsData = {
    labels: ['MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN'],
    datasets: [{
      label: 'Correlation',
      data: [0.82, 0.79, 0.75, 0.68, 0.72],
      backgroundColor: alpha(theme.palette.primary.main, 0.6),
      borderColor: theme.palette.primary.main,
      borderWidth: 2,
    }],
  };

  if (loading) {
    return (
      <ContextContainer>
        <Skeleton variant="rectangular" height={200} />
        <Skeleton variant="rectangular" height={300} />
        <Skeleton variant="rectangular" height={200} />
      </ContextContainer>
    );
  }

  return (
    <ContextContainer>
      {/* AI Summary */}
      <Card elevation={0} sx={{
        background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)} 0%, ${alpha(theme.palette.primary.main, 0.05)} 100%)`,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
      }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center" gap={1}>
              <PsychologyIcon color="primary" />
              <Typography variant="h6" fontWeight="bold">
                AI Market Analysis
              </Typography>
            </Box>
            <IconButton size="small" onClick={fetchData}>
              <RefreshIcon />
            </IconButton>
          </Box>

          <Typography variant="body2" color="text.secondary" paragraph>
            {aiSummary}
          </Typography>

          <Box>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle2">Overall Sentiment</Typography>
              <Chip
                label={`${sentiment}% Bullish`}
                color="success"
                size="small"
                icon={<TrendingUpIcon />}
              />
            </Box>
            <SentimentIndicator>
              <SentimentMarker sx={{ left: `${sentiment}%` }} />
            </SentimentIndicator>
          </Box>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} variant="fullWidth">
        <Tab label="News" icon={<NewspaperIcon />} iconPosition="start" />
        <Tab label="Options" icon={<BarChartIcon />} iconPosition="start" />
        <Tab label="Market" icon={<ShowChartIcon />} iconPosition="start" />
      </Tabs>

      {/* Tab Content */}
      <Box flex={1} overflow="auto">
        <AnimatePresence mode="wait">
          {activeTab === 0 && (
            <motion.div
              key="news"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <Stack spacing={2}>
                {news.map((item) => (
                  <NewsCard
                    key={item.id}
                    elevation={0}
                    onClick={() => onNewsClick?.(item)}
                  >
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                        <Box flex={1}>
                          <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                            {item.title}
                          </Typography>
                          <Box display="flex" alignItems="center" gap={1} mb={1}>
                            <Chip
                              label={item.source}
                              size="small"
                              variant="outlined"
                            />
                            <Typography variant="caption" color="text.secondary">
                              <TimeIcon sx={{ fontSize: 12, mr: 0.5, verticalAlign: 'middle' }} />
                              {item.time}
                            </Typography>
                          </Box>
                        </Box>
                        <Box display="flex" gap={0.5}>
                          <Chip
                            label={item.sentiment}
                            size="small"
                            sx={{
                              backgroundColor: alpha(getSentimentColor(item.sentiment), 0.1),
                              color: getSentimentColor(item.sentiment),
                              fontWeight: 600,
                            }}
                          />
                          <Chip
                            label={item.impact}
                            size="small"
                            icon={<FireIcon />}
                            sx={{
                              backgroundColor: alpha(getImpactColor(item.impact), 0.1),
                              color: getImpactColor(item.impact),
                              fontWeight: 600,
                            }}
                          />
                        </Box>
                      </Box>
                      {item.summary && (
                        <Typography variant="body2" color="text.secondary">
                          {item.summary}
                        </Typography>
                      )}
                    </CardContent>
                  </NewsCard>
                ))}
              </Stack>
            </motion.div>
          )}

          {activeTab === 1 && (
            <motion.div
              key="options"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <Stack spacing={2}>
                {/* Options Summary */}
                <OptionsFlowCard elevation={0}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Options Flow Summary
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Box height={150}>
                          <Doughnut
                            data={optionsChart}
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
                      </Grid>
                      <Grid item xs={6}>
                        <Stack spacing={1}>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              Put/Call Ratio
                            </Typography>
                            <Typography variant="h6" fontWeight="bold">
                              {(putCallRatio.puts / putCallRatio.calls).toFixed(2)}
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              Total Volume
                            </Typography>
                            <Typography variant="h6" fontWeight="bold">
                              {((putCallRatio.calls + putCallRatio.puts) / 1000).toFixed(1)}K
                            </Typography>
                          </Box>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              Smart Money Flow
                            </Typography>
                            <Chip
                              label="BULLISH"
                              color="success"
                              size="small"
                              icon={<TrendingUpIcon />}
                            />
                          </Box>
                        </Stack>
                      </Grid>
                    </Grid>
                  </CardContent>
                </OptionsFlowCard>

                {/* Options Table */}
                <Paper elevation={0} sx={{ p: 2 }}>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                    Unusual Options Activity
                  </Typography>
                  <Box sx={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left', padding: '8px' }}>Strike</th>
                          <th style={{ textAlign: 'left', padding: '8px' }}>Type</th>
                          <th style={{ textAlign: 'right', padding: '8px' }}>Volume</th>
                          <th style={{ textAlign: 'right', padding: '8px' }}>Premium</th>
                          <th style={{ textAlign: 'center', padding: '8px' }}>Flow</th>
                        </tr>
                      </thead>
                      <tbody>
                        {optionsFlow.map((flow, index) => (
                          <tr key={index}>
                            <td style={{ padding: '8px' }}>${flow.strike}</td>
                            <td style={{ padding: '8px' }}>
                              <Chip
                                label={flow.type.toUpperCase()}
                                size="small"
                                color={flow.type === 'call' ? 'success' : 'error'}
                              />
                            </td>
                            <td style={{ textAlign: 'right', padding: '8px' }}>
                              {(flow.volume / 1000).toFixed(1)}K
                            </td>
                            <td style={{ textAlign: 'right', padding: '8px' }}>
                              {formatCurrency(flow.premium)}
                            </td>
                            <td style={{ textAlign: 'center', padding: '8px' }}>
                              {flow.flow === 'bullish' ? '🔥' : '❄️'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Box>
                </Paper>
              </Stack>
            </motion.div>
          )}

          {activeTab === 2 && (
            <motion.div
              key="market"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <Stack spacing={2}>
                {/* Market Indices */}
                <Box>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                    Market Performance
                  </Typography>
                  <Grid container spacing={2}>
                    {marketMetrics.map((metric) => (
                      <Grid item xs={6} key={metric.symbol}>
                        <MetricCard elevation={0}>
                          <Typography variant="caption" color="text.secondary">
                            {metric.name}
                          </Typography>
                          <Typography variant="h6" fontWeight="bold">
                            ${metric.price.toFixed(2)}
                          </Typography>
                          <Chip
                            label={`${metric.change >= 0 ? '+' : ''}${metric.changePercent.toFixed(2)}%`}
                            size="small"
                            color={metric.change >= 0 ? 'success' : 'error'}
                            icon={metric.change >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                          />
                        </MetricCard>
                      </Grid>
                    ))}
                  </Grid>
                </Box>

                {/* Correlated Assets */}
                <Paper elevation={0} sx={{ p: 2 }}>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                    Correlated Assets
                  </Typography>
                  <Box height={200}>
                    <Bar
                      data={correlatedAssetsData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            beginAtZero: true,
                            max: 1,
                          },
                        },
                        plugins: {
                          legend: {
                            display: false,
                          },
                        },
                      }}
                    />
                  </Box>
                </Paper>

                {/* Sector Performance */}
                <Paper elevation={0} sx={{ p: 2 }}>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>
                    Sector Performance
                  </Typography>
                  <Stack spacing={1}>
                    {[
                      { name: 'Technology', change: 2.1, color: 'success' },
                      { name: 'Healthcare', change: 0.8, color: 'success' },
                      { name: 'Financials', change: -0.3, color: 'error' },
                      { name: 'Energy', change: 1.2, color: 'success' },
                      { name: 'Consumer', change: 0.5, color: 'success' },
                    ].map((sector) => (
                      <Box key={sector.name} display="flex" alignItems="center" justifyContent="space-between">
                        <Typography variant="body2">{sector.name}</Typography>
                        <Chip
                          label={`${sector.change >= 0 ? '+' : ''}${sector.change}%`}
                          size="small"
                          color={sector.color as any}
                        />
                      </Box>
                    ))}
                  </Stack>
                </Paper>
              </Stack>
            </motion.div>
          )}
        </AnimatePresence>
      </Box>
    </ContextContainer>
  );
};

export default MarketContext;
