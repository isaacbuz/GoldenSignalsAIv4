import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  InputAdornment,
  IconButton,
  Chip,
  Button,
  Menu,
  MenuItem,
  Slider,
  FormControl,
  InputLabel,
  Select,
  Switch,
  FormControlLabel,
  Divider,
  Tooltip,
  Badge,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Search,
  FilterList,
  Download,
  MoreVert,
  TrendingUp,
  TrendingDown,
  Remove,
  AutoAwesome,
  Speed,
  Psychology,
  Timeline,
  Refresh,
  NotificationsActive,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';

const StyledCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  marginBottom: theme.spacing(2),
  transition: 'all 0.3s ease',
  cursor: 'pointer',
  '&:hover': {
    transform: 'translateX(8px)',
    borderColor: 'rgba(255, 215, 0, 0.3)',
  },
}));

const SignalBadge = styled(Chip)(({ signalType }: { signalType: string }) => ({
  fontWeight: 700,
  ...(signalType === 'BUY' && {
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    color: '#4CAF50',
    border: '1px solid rgba(76, 175, 80, 0.3)',
  }),
  ...(signalType === 'SELL' && {
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    color: '#F44336',
    border: '1px solid rgba(244, 67, 54, 0.3)',
  }),
  ...(signalType === 'HOLD' && {
    backgroundColor: 'rgba(255, 165, 0, 0.1)',
    color: '#FFA500',
    border: '1px solid rgba(255, 165, 0, 0.3)',
  }),
}));

interface Signal {
  id: string;
  timestamp: Date;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  agents: string[];
  reasoning: string;
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
  historicalAccuracy: number;
}

const SignalStream: React.FC = () => {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [filteredSignals, setFilteredSignals] = useState<Signal[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterAnchorEl, setFilterAnchorEl] = useState<null | HTMLElement>(null);
  const [signalTypeFilter, setSignalTypeFilter] = useState<string[]>(['BUY', 'SELL', 'HOLD']);
  const [confidenceRange, setConfidenceRange] = useState<number[]>([0, 100]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);

  // Mock real-time signal generation
  useEffect(() => {
    const generateMockSignal = (): Signal => {
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'BTC', 'ETH'];
      const types: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
      const agents = ['sentiment', 'technical', 'flow', 'risk', 'regime', 'liquidity'];
      
      return {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: new Date(),
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        type: types[Math.floor(Math.random() * types.length)],
        confidence: 70 + Math.random() * 30,
        agents: agents.slice(0, Math.floor(Math.random() * 3) + 2),
        reasoning: 'AI consensus based on multiple indicators showing strong signals',
        impact: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)] as 'HIGH' | 'MEDIUM' | 'LOW',
        historicalAccuracy: 85 + Math.random() * 10,
      };
    };

    // Initial signals
    const initialSignals = Array.from({ length: 10 }, generateMockSignal);
    setSignals(initialSignals);
    setFilteredSignals(initialSignals);

    // Real-time updates
    if (autoRefresh) {
      const interval = setInterval(() => {
        const newSignal = generateMockSignal();
        setSignals(prev => [newSignal, ...prev].slice(0, 50));
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Filter signals
  useEffect(() => {
    let filtered = signals;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(signal =>
        signal.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Type filter
    filtered = filtered.filter(signal => signalTypeFilter.includes(signal.type));

    // Confidence filter
    filtered = filtered.filter(signal =>
      signal.confidence >= confidenceRange[0] && signal.confidence <= confidenceRange[1]
    );

    // Agent filter
    if (selectedAgents.length > 0) {
      filtered = filtered.filter(signal =>
        signal.agents.some(agent => selectedAgents.includes(agent))
      );
    }

    setFilteredSignals(filtered);
  }, [signals, searchTerm, signalTypeFilter, confidenceRange, selectedAgents]);

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
          Signal Stream
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Real-time AI-generated trading signals with confidence scores and reasoning
        </Typography>
      </Box>

      {/* Live indicator */}
      <Alert 
        severity="info" 
        icon={<AutoAwesome />}
        sx={{ mb: 3 }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip 
            label="LIVE" 
            size="small" 
            sx={{ 
              backgroundColor: 'rgba(76, 175, 80, 0.1)',
              color: '#4CAF50',
              animation: 'pulse 2s infinite'
            }}
          />
          <Typography variant="body2">
            Streaming real-time signals from {signals.filter(s => s.agents.length > 0).length} active AI agents
          </Typography>
        </Box>
      </Alert>

      {/* Controls */}
      <Card sx={{ mb: 3, ...utilityClasses.glassmorphism }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
            <TextField
              placeholder="Search symbols..."
              variant="outlined"
              size="small"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              sx={{ minWidth: 200 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />

            <Button
              variant="outlined"
              startIcon={<FilterList />}
              onClick={(e) => setFilterAnchorEl(e.currentTarget)}
              sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
            >
              Filters
              {(signalTypeFilter.length < 3 || selectedAgents.length > 0) && (
                <Badge badgeContent="•" color="primary" sx={{ ml: 1 }} />
              )}
            </Button>

            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  color="primary"
                />
              }
              label="Auto-refresh"
            />

            <Box sx={{ flexGrow: 1 }} />

            <Button
              variant="outlined"
              startIcon={<Download />}
              sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
            >
              Export
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Signal Count */}
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Showing {filteredSignals.length} of {signals.length} signals
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip label={`BUY: ${filteredSignals.filter(s => s.type === 'BUY').length}`} size="small" />
          <Chip label={`SELL: ${filteredSignals.filter(s => s.type === 'SELL').length}`} size="small" />
          <Chip label={`HOLD: ${filteredSignals.filter(s => s.type === 'HOLD').length}`} size="small" />
        </Box>
      </Box>

      {/* Signals List */}
      <Box>
        {filteredSignals.map((signal, index) => (
          <StyledCard key={signal.id}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box sx={{ flex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {signal.symbol}
                    </Typography>
                    <SignalBadge label={signal.type} signalType={signal.type} />
                    <Chip
                      label={`${signal.confidence.toFixed(1)}%`}
                      size="small"
                      icon={<Speed />}
                      sx={{ fontWeight: 600 }}
                    />
                    {signal.impact === 'HIGH' && (
                      <Chip
                        label="HIGH IMPACT"
                        size="small"
                        sx={{
                          backgroundColor: 'rgba(255, 215, 0, 0.1)',
                          color: '#FFD700',
                          border: '1px solid rgba(255, 215, 0, 0.3)',
                        }}
                      />
                    )}
                    {index === 0 && (
                      <Chip
                        label="NEW"
                        size="small"
                        sx={{
                          backgroundColor: 'rgba(33, 150, 243, 0.1)',
                          color: '#2196F3',
                          animation: 'fadeIn 0.5s ease',
                        }}
                      />
                    )}
                  </Box>

                  <Typography variant="body2" sx={{ mb: 1 }}>
                    {signal.reasoning}
                  </Typography>

                  <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      <Psychology sx={{ fontSize: 14, mr: 0.5, verticalAlign: 'middle' }} />
                      Agents: {signal.agents.join(', ')}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      <Timeline sx={{ fontSize: 14, mr: 0.5, verticalAlign: 'middle' }} />
                      Historical accuracy: {signal.historicalAccuracy.toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </Typography>
                  </Box>
                </Box>

                <IconButton size="small">
                  <MoreVert />
                </IconButton>
              </Box>

              {/* Confidence bar */}
              <Box sx={{ mt: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={signal.confidence}
                  sx={{
                    height: 4,
                    borderRadius: 2,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: signal.confidence > 90 ? '#4CAF50' : 
                                      signal.confidence > 80 ? '#FFD700' : '#FFA500',
                    },
                  }}
                />
              </Box>
            </CardContent>
          </StyledCard>
        ))}
      </Box>

      {/* Filter Menu */}
      <Menu
        anchorEl={filterAnchorEl}
        open={Boolean(filterAnchorEl)}
        onClose={() => setFilterAnchorEl(null)}
        PaperProps={{
          sx: { ...utilityClasses.glassmorphism, minWidth: 300 }
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 2 }}>Signal Type</Typography>
          <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
            {['BUY', 'SELL', 'HOLD'].map((type) => (
              <Chip
                key={type}
                label={type}
                onClick={() => {
                  if (signalTypeFilter.includes(type)) {
                    setSignalTypeFilter(signalTypeFilter.filter(t => t !== type));
                  } else {
                    setSignalTypeFilter([...signalTypeFilter, type]);
                  }
                }}
                sx={{
                  cursor: 'pointer',
                  opacity: signalTypeFilter.includes(type) ? 1 : 0.5,
                }}
              />
            ))}
          </Box>

          <Typography variant="subtitle2" sx={{ mb: 2 }}>Confidence Range</Typography>
          <Box sx={{ px: 1, mb: 3 }}>
            <Slider
              value={confidenceRange}
              onChange={(_, newValue) => setConfidenceRange(newValue as number[])}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${value}%`}
              min={0}
              max={100}
            />
          </Box>
        </Box>
      </Menu>
    </Box>
  );
};

export default SignalStream;
