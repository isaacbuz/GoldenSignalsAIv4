import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Stack,
  Avatar,
  LinearProgress,
  Divider,
  alpha,
  useTheme,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  LocalFireDepartment as FireIcon,
  Bolt as BoltIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  Schedule as ScheduleIcon,
  Info as InfoIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { customStyles } from '../../theme/enhancedTheme';

// Styled components
const SuggestionCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(0.75),
  ...customStyles.card,
  padding: 0,
  '& .MuiCardContent-root': {
    padding: theme.spacing(1.25),
  },
  '& .MuiCardActions-root': {
    padding: theme.spacing(0.75, 1.25),
  },
  '&:hover': {
    borderColor: alpha(theme.palette.primary.main, 0.5),
    transform: 'translateY(-1px)',
    transition: 'all 0.2s ease',
  },
}));

const OpportunityBadge = styled(Box)<{ type: 'hot' | 'momentum' | 'breakout' }>(({ theme, type }) => {
  const colors = {
    hot: { bg: theme.palette.error.main, icon: '🔥' },
    momentum: { bg: theme.palette.warning.main, icon: '⚡' },
    breakout: { bg: theme.palette.success.main, icon: '📈' },
  };

  return {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(0.25),
    padding: theme.spacing(0.25, 0.75),
    backgroundColor: alpha(colors[type].bg, 0.1),
    borderRadius: theme.spacing(2),
    fontSize: '0.65rem',
    fontWeight: 'bold',
    color: colors[type].bg,
    border: `1px solid ${alpha(colors[type].bg, 0.3)}`,
  };
});

const ConfidenceBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginTop: theme.spacing(1),
}));

interface SignalOpportunity {
  symbol: string;
  name: string;
  type: 'hot' | 'momentum' | 'breakout';
  action: 'BUY' | 'SELL';
  price: number;
  targetPrice: number;
  confidence: number;
  timeframe: string;
  reasoning: string;
  urgency: 'High' | 'Medium' | 'Low';
  volume: number;
  change: number;
  changePercent: number;
}

interface SignalSuggestionsProps {
  onSignalSelect: (symbol: string, timeframe: string) => void;
}

const SignalSuggestions: React.FC<SignalSuggestionsProps> = ({ onSignalSelect }) => {
  const theme = useTheme();

  const suggestions: SignalOpportunity[] = [
    {
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      type: 'hot',
      action: 'BUY',
      price: 425.80,
      targetPrice: 465.00,
      confidence: 0.92,
      timeframe: '1d',
      reasoning: 'AI boom continues, strong earnings expected',
      urgency: 'High',
      volume: 45000000,
      change: 8.45,
      changePercent: 2.03,
    },
    {
      symbol: 'TSLA',
      name: 'Tesla Inc.',
      type: 'momentum',
      action: 'BUY',
      price: 245.67,
      targetPrice: 275.00,
      confidence: 0.87,
      timeframe: '4h',
      reasoning: 'Bullish momentum building after recent dip',
      urgency: 'Medium',
      volume: 87000000,
      change: 12.33,
      changePercent: 5.28,
    },
    {
      symbol: 'AMD',
      name: 'Advanced Micro Devices',
      type: 'breakout',
      action: 'BUY',
      price: 102.15,
      targetPrice: 115.00,
      confidence: 0.78,
      timeframe: '1h',
      reasoning: 'Breaking resistance, volume surge detected',
      urgency: 'High',
      volume: 23000000,
      change: 1.87,
      changePercent: 1.86,
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      type: 'momentum',
      action: 'SELL',
      price: 280.50,
      targetPrice: 265.00,
      confidence: 0.82,
      timeframe: '1d',
      reasoning: 'Overbought conditions, profit taking expected',
      urgency: 'Low',
      volume: 31000000,
      change: -2.25,
      changePercent: -0.80,
    },
  ];

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'hot':
        return 'Hot Signal';
      case 'momentum':
        return 'Momentum Play';
      case 'breakout':
        return 'Breakout Alert';
      default:
        return type;
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'High':
        return 'error';
      case 'Medium':
        return 'warning';
      case 'Low':
        return 'info';
      default:
        return 'default';
    }
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    }
    if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const getPotentialReturn = (currentPrice: number, targetPrice: number) => {
    const returnPercent = ((targetPrice - currentPrice) / currentPrice) * 100;
    return returnPercent.toFixed(1);
  };

  return (
    <Box height="100%" display="flex" flexDirection="column">
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <Typography variant="body2" color="text.secondary">
          AI-generated opportunities
        </Typography>
        <Chip
          label="Live"
          size="small"
          color="success"
          sx={{ height: 18, fontSize: '0.65rem' }}
        />
      </Box>

      <Box flex={1} sx={{ overflowY: 'auto' }}>
        {suggestions.map((suggestion, index) => (
          <SuggestionCard key={index} variant="outlined">
            <CardContent sx={{ p: 1, pb: 0.5 }}>
              {/* Header */}
              <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={0.5}>
                <Box>
                  <Typography variant="subtitle2" fontWeight="bold">
                    {suggestion.symbol}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {suggestion.name}
                  </Typography>
                </Box>
                <Box display="flex" flexDirection="column" alignItems="flex-end" gap={0.5}>
                  <OpportunityBadge type={suggestion.type}>
                    {suggestion.type === 'hot' && '🔥'}
                    {suggestion.type === 'momentum' && '⚡'}
                    {suggestion.type === 'breakout' && '📈'}
                    {getTypeLabel(suggestion.type)}
                  </OpportunityBadge>
                  <Chip
                    label={suggestion.urgency}
                    size="small"
                    color={getUrgencyColor(suggestion.urgency)}
                    sx={{ height: 16, fontSize: '0.6rem' }}
                  />
                </Box>
              </Box>

              {/* Signal Details */}
              <Stack spacing={0.5}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Avatar
                    sx={{
                      width: 20,
                      height: 20,
                      bgcolor: suggestion.action === 'BUY' ? 'success.main' : 'error.main',
                      fontSize: '0.7rem',
                    }}
                  >
                    {suggestion.action === 'BUY' ? '↑' : '↓'}
                  </Avatar>
                  <Typography variant="body2" fontWeight="bold">
                    {suggestion.action} @ ${suggestion.price.toFixed(2)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    → ${suggestion.targetPrice.toFixed(2)}
                  </Typography>
                </Box>

                <Box display="flex" alignItems="center" gap={2}>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <TimelineIcon fontSize="small" color="action" />
                    <Typography variant="caption">
                      {suggestion.timeframe}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <Typography variant="caption">
                      Vol: {formatVolume(suggestion.volume)}
                    </Typography>
                  </Box>
                </Box>

                <Typography variant="caption" color="text.secondary">
                  {suggestion.reasoning}
                </Typography>

                <ConfidenceBar>
                  <Typography variant="caption" color="text.secondary">
                    Confidence
                  </Typography>
                  <Box flex={1}>
                    <LinearProgress
                      variant="determinate"
                      value={suggestion.confidence * 100}
                      sx={{ height: 4, borderRadius: 2 }}
                    />
                  </Box>
                  <Typography variant="caption" fontWeight="bold">
                    {(suggestion.confidence * 100).toFixed(0)}%
                  </Typography>
                </ConfidenceBar>

                <Box display="flex" justifyContent="space-between" alignItems="center" mt={0.5}>
                  <Typography variant="caption" color="text.secondary">
                    Potential: {getPotentialReturn(suggestion.price, suggestion.targetPrice)}%
                  </Typography>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    {suggestion.change >= 0 ? (
                      <TrendingUpIcon fontSize="small" color="success" />
                    ) : (
                      <TrendingDownIcon fontSize="small" color="error" />
                    )}
                    <Typography
                      variant="caption"
                      color={suggestion.change >= 0 ? 'success.main' : 'error.main'}
                    >
                      {suggestion.change >= 0 ? '+' : ''}{suggestion.change.toFixed(2)} ({suggestion.changePercent.toFixed(2)}%)
                    </Typography>
                  </Box>
                </Box>
              </Stack>
            </CardContent>

            <CardActions sx={{ p: 1, pt: 0 }}>
              <Button
                size="small"
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={() => onSignalSelect(suggestion.symbol, suggestion.timeframe)}
                fullWidth
              >
                Analyze
              </Button>
              <Tooltip title="More details">
                <IconButton size="small">
                  <InfoIcon />
                </IconButton>
              </Tooltip>
            </CardActions>
          </SuggestionCard>
        ))}
      </Box>

      <Divider sx={{ my: 1 }} />

      <Box textAlign="center">
        <Typography variant="caption" color="text.secondary">
          Updated every 5 minutes
        </Typography>
      </Box>
    </Box>
  );
};

export default SignalSuggestions;
