import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
  Alert,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  ShowChart,
  Warning,
  AutoAwesome,
  Refresh,
  Timeline,
  BubbleChart,
  Insights,
  Speed,
  WaterfallChart,
  CandlestickChart,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';
import * as d3 from 'd3';

const StyledCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: '100%',
}));

const HeatmapContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: 400,
  '& .tooltip': {
    position: 'absolute',
    padding: theme.spacing(1),
    background: 'rgba(0, 0, 0, 0.9)',
    color: '#fff',
    borderRadius: 4,
    pointerEvents: 'none',
    fontSize: '0.875rem',
    zIndex: 10,
  },
}));

interface SentimentData {
  sector: string;
  sentiment: number;
  volume: number;
  change: number;
}

interface FlowData {
  symbol: string;
  callVolume: number;
  putVolume: number;
  ratio: number;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  premium: number;
}

interface Pattern {
  id: string;
  name: string;
  symbol: string;
  confidence: number;
  type: 'bullish' | 'bearish';
  description: string;
}

interface Anomaly {
  id: string;
  type: string;
  severity: 'high' | 'medium' | 'low';
  description: string;
  timestamp: string;
  affectedSymbols: string[];
}

const MarketIntelligence: React.FC = () => {
  const [view, setView] = useState<'sentiment' | 'flow' | 'patterns' | 'anomalies'>('sentiment');
  const [timeframe, setTimeframe] = useState('1d');
  const [loading, setLoading] = useState(false);
  
  // Mock data
  const [sentimentData] = useState<SentimentData[]>([
    { sector: 'Technology', sentiment: 0.75, volume: 1234567, change: 2.3 },
    { sector: 'Healthcare', sentiment: 0.45, volume: 987654, change: -0.8 },
    { sector: 'Finance', sentiment: 0.62, volume: 1567890, change: 1.2 },
    { sector: 'Energy', sentiment: -0.23, volume: 876543, change: -3.4 },
    { sector: 'Consumer', sentiment: 0.34, volume: 1098765, change: 0.5 },
    { sector: 'Industrial', sentiment: 0.18, volume: 765432, change: -1.1 },
    { sector: 'Materials', sentiment: -0.42, volume: 654321, change: -2.7 },
    { sector: 'Real Estate', sentiment: 0.56, volume: 543210, change: 1.8 },
  ]);

  const [flowData] = useState<FlowData[]>([
    { symbol: 'AAPL', callVolume: 125000, putVolume: 45000, ratio: 2.78, sentiment: 'bullish', premium: 8.5 },
    { symbol: 'SPY', callVolume: 450000, putVolume: 380000, ratio: 1.18, sentiment: 'neutral', premium: 15.2 },
    { symbol: 'TSLA', callVolume: 89000, putVolume: 156000, ratio: 0.57, sentiment: 'bearish', premium: 12.3 },
    { symbol: 'NVDA', callVolume: 234000, putVolume: 67000, ratio: 3.49, sentiment: 'bullish', premium: 18.7 },
    { symbol: 'AMZN', callVolume: 98000, putVolume: 87000, ratio: 1.13, sentiment: 'neutral', premium: 9.8 },
  ]);

  const [patterns] = useState<Pattern[]>([
    { id: '1', name: 'Golden Cross', symbol: 'MSFT', confidence: 92, type: 'bullish', description: '50-day MA crossed above 200-day MA' },
    { id: '2', name: 'Head and Shoulders', symbol: 'META', confidence: 87, type: 'bearish', description: 'Classic reversal pattern forming' },
    { id: '3', name: 'Cup and Handle', symbol: 'GOOGL', confidence: 94, type: 'bullish', description: 'Continuation pattern confirmed' },
    { id: '4', name: 'Double Bottom', symbol: 'NFLX', confidence: 89, type: 'bullish', description: 'Support level tested twice' },
  ]);

  const [anomalies] = useState<Anomaly[]>([
    { id: '1', type: 'Unusual Options Activity', severity: 'high', description: 'Massive call buying in NVDA', timestamp: '5 min ago', affectedSymbols: ['NVDA'] },
    { id: '2', type: 'Sentiment Spike', severity: 'medium', description: 'Reddit mentions increased 500%', timestamp: '1 hour ago', affectedSymbols: ['GME', 'AMC'] },
    { id: '3', type: 'Volume Anomaly', severity: 'high', description: 'Trading volume 10x average', timestamp: '2 hours ago', affectedSymbols: ['TSLA'] },
  ]);

  // Create sentiment heatmap
  useEffect(() => {
    if (view === 'sentiment') {
      createSentimentHeatmap();
    } else if (view === 'flow') {
      createFlowTreemap();
    }
  }, [view, sentimentData]);

  const createSentimentHeatmap = () => {
    const container = d3.select('#sentiment-heatmap');
    container.selectAll('*').remove();

    const width = container.node()?.getBoundingClientRect().width || 800;
    const height = 400;
    const margin = { top: 40, right: 40, bottom: 40, left: 100 };

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Color scale
    const colorScale = d3.scaleLinear<string>()
      .domain([-1, 0, 1])
      .range(['#F44336', '#FFA500', '#4CAF50']);

    // Create rectangles for each sector
    const rectHeight = innerHeight / sentimentData.length;

    g.selectAll('rect')
      .data(sentimentData)
      .enter()
      .append('rect')
      .attr('x', 0)
      .attr('y', (d, i) => i * rectHeight)
      .attr('width', innerWidth)
      .attr('height', rectHeight - 2)
      .attr('fill', d => colorScale(d.sentiment))
      .attr('opacity', 0.8)
      .on('mouseenter', function(event, d) {
        // Show tooltip
        const tooltip = container.append('div')
          .attr('class', 'tooltip')
          .style('left', `${event.offsetX + 10}px`)
          .style('top', `${event.offsetY - 10}px`)
          .html(`
            <strong>${d.sector}</strong><br/>
            Sentiment: ${(d.sentiment * 100).toFixed(1)}%<br/>
            Volume: ${d.volume.toLocaleString()}<br/>
            Change: ${d.change > 0 ? '+' : ''}${d.change.toFixed(2)}%
          `);
      })
      .on('mouseleave', function() {
        container.select('.tooltip').remove();
      });

    // Add sector labels
    g.selectAll('text')
      .data(sentimentData)
      .enter()
      .append('text')
      .attr('x', -10)
      .attr('y', (d, i) => i * rectHeight + rectHeight / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .text(d => d.sector);
  };

  const createFlowTreemap = () => {
    const container = d3.select('#flow-treemap');
    container.selectAll('*').remove();

    const width = container.node()?.getBoundingClientRect().width || 800;
    const height = 400;

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Prepare hierarchical data
    const hierarchyData = {
      name: 'Options Flow',
      children: flowData.map(d => ({
        name: d.symbol,
        value: d.callVolume + d.putVolume,
        ...d
      }))
    };

    const root = d3.hierarchy(hierarchyData)
      .sum(d => d.value)
      .sort((a, b) => b.value - a.value);

    d3.treemap()
      .size([width, height])
      .padding(2)
      (root);

    const leaf = svg.selectAll('g')
      .data(root.leaves())
      .enter().append('g')
      .attr('transform', d => `translate(${d.x0},${d.y0})`);

    leaf.append('rect')
      .attr('width', d => d.x1 - d.x0)
      .attr('height', d => d.y1 - d.y0)
      .attr('fill', d => {
        const sentiment = d.data.sentiment;
        return sentiment === 'bullish' ? '#4CAF50' : 
               sentiment === 'bearish' ? '#F44336' : '#FFA500';
      })
      .attr('opacity', 0.8);

    leaf.append('text')
      .attr('x', 4)
      .attr('y', 20)
      .text(d => d.data.name)
      .attr('font-size', '14px')
      .attr('fill', '#fff')
      .attr('font-weight', 'bold');

    leaf.append('text')
      .attr('x', 4)
      .attr('y', 40)
      .text(d => `C/P: ${d.data.ratio.toFixed(2)}`)
      .attr('font-size', '12px')
      .attr('fill', '#fff');
  };

  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 1000);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
          Market Intelligence
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Real-time sentiment analysis, options flow, and pattern detection
        </Typography>
      </Box>

      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <ToggleButtonGroup
          value={view}
          exclusive
          onChange={(_, newView) => newView && setView(newView)}
          size="small"
        >
          <ToggleButton value="sentiment">
            <Psychology sx={{ mr: 1 }} /> Sentiment
          </ToggleButton>
          <ToggleButton value="flow">
            <WaterfallChart sx={{ mr: 1 }} /> Flow
          </ToggleButton>
          <ToggleButton value="patterns">
            <CandlestickChart sx={{ mr: 1 }} /> Patterns
          </ToggleButton>
          <ToggleButton value="anomalies">
            <Warning sx={{ mr: 1 }} /> Anomalies
          </ToggleButton>
        </ToggleButtonGroup>

        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Timeframe</InputLabel>
          <Select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} label="Timeframe">
            <MenuItem value="1h">1 Hour</MenuItem>
            <MenuItem value="1d">1 Day</MenuItem>
            <MenuItem value="1w">1 Week</MenuItem>
          </Select>
        </FormControl>

        <Box sx={{ flexGrow: 1 }} />

        <Tooltip title="Refresh data">
          <IconButton onClick={handleRefresh} disabled={loading}>
            <Refresh sx={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Alert for anomalies */}
      {anomalies.filter(a => a.severity === 'high').length > 0 && (
        <Alert severity="warning" icon={<Warning />} sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>High severity anomalies detected!</strong> Check the anomalies tab for details.
          </Typography>
        </Alert>
      )}

      {/* Main Content */}
      {view === 'sentiment' && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Sector Sentiment Heatmap
                </Typography>
                <HeatmapContainer id="sentiment-heatmap" />
              </CardContent>
            </StyledCard>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Top Bullish Sectors
                </Typography>
                {sentimentData
                  .filter(s => s.sentiment > 0)
                  .sort((a, b) => b.sentiment - a.sentiment)
                  .slice(0, 3)
                  .map(sector => (
                    <Box key={sector.sector} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">{sector.sector}</Typography>
                        <Chip 
                          label={`+${(sector.sentiment * 100).toFixed(1)}%`}
                          size="small"
                          sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)', color: '#4CAF50' }}
                        />
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={sector.sentiment * 100}
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: '#4CAF50',
                          },
                        }}
                      />
                    </Box>
                  ))}
              </CardContent>
            </StyledCard>
          </Grid>

          <Grid item xs={12} md={6}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Social Sentiment Trends
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Reddit Mentions</Typography>
                    <Chip label="+234%" size="small" sx={{ color: '#4CAF50' }} />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Twitter Sentiment</Typography>
                    <Chip label="78% Positive" size="small" />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">News Sentiment</Typography>
                    <Chip label="Neutral" size="small" />
                  </Box>
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      )}

      {view === 'flow' && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Options Flow Treemap
                </Typography>
                <HeatmapContainer id="flow-treemap" />
              </CardContent>
            </StyledCard>
          </Grid>

          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Unusual Options Activity
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {flowData
                    .filter(f => f.ratio > 2 || f.ratio < 0.5)
                    .map(flow => (
                      <Paper
                        key={flow.symbol}
                        sx={{
                          p: 2,
                          backgroundColor: 'rgba(255, 255, 255, 0.02)',
                          border: '1px solid rgba(255, 215, 0, 0.2)',
                        }}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="h6">{flow.symbol}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              Call/Put Ratio: {flow.ratio.toFixed(2)}
                            </Typography>
                          </Box>
                          <Box sx={{ textAlign: 'right' }}>
                            <Chip
                              label={flow.sentiment}
                              size="small"
                              sx={{
                                backgroundColor: flow.sentiment === 'bullish' ? 'rgba(76, 175, 80, 0.1)' :
                                               flow.sentiment === 'bearish' ? 'rgba(244, 67, 54, 0.1)' :
                                               'rgba(255, 165, 0, 0.1)',
                                color: flow.sentiment === 'bullish' ? '#4CAF50' :
                                      flow.sentiment === 'bearish' ? '#F44336' : '#FFA500',
                              }}
                            />
                            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                              Premium: ${flow.premium.toFixed(2)}M
                            </Typography>
                          </Box>
                        </Box>
                      </Paper>
                    ))}
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      )}

      {view === 'patterns' && (
        <Grid container spacing={3}>
          {patterns.map(pattern => (
            <Grid item xs={12} md={6} key={pattern.id}>
              <StyledCard>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Box>
                      <Typography variant="h6">{pattern.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {pattern.symbol}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Chip
                        icon={pattern.type === 'bullish' ? <TrendingUp /> : <TrendingDown />}
                        label={pattern.type}
                        size="small"
                        sx={{
                          backgroundColor: pattern.type === 'bullish' ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)',
                          color: pattern.type === 'bullish' ? '#4CAF50' : '#F44336',
                        }}
                      />
                      <Typography variant="h6" sx={{ mt: 1, color: '#FFD700' }}>
                        {pattern.confidence}%
                      </Typography>
                    </Box>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {pattern.description}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={pattern.confidence}
                    sx={{
                      mt: 2,
                      height: 4,
                      borderRadius: 2,
                      backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: pattern.type === 'bullish' ? '#4CAF50' : '#F44336',
                      },
                    }}
                  />
                </CardContent>
              </StyledCard>
            </Grid>
          ))}
        </Grid>
      )}

      {view === 'anomalies' && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Detected Anomalies
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {anomalies.map(anomaly => (
                    <Alert
                      key={anomaly.id}
                      severity={anomaly.severity === 'high' ? 'error' : anomaly.severity === 'medium' ? 'warning' : 'info'}
                      icon={<Warning />}
                    >
                      <Box>
                        <Typography variant="subtitle2">{anomaly.type}</Typography>
                        <Typography variant="body2">{anomaly.description}</Typography>
                        <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            {anomaly.timestamp}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Symbols: {anomaly.affectedSymbols.join(', ')}
                          </Typography>
                        </Box>
                      </Box>
                    </Alert>
                  ))}
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default MarketIntelligence;
