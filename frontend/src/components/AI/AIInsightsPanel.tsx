import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Stack,
  LinearProgress,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  ShowChart,
  Assessment,
  Warning,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { fetchAIInsights, AIInsight } from '../../services/api';

interface AIInsightsPanelProps {
  symbol: string;
}

const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({ symbol }) => {
  const theme = useTheme();

  const { data: insights, isLoading } = useQuery<AIInsight>({
    queryKey: ['aiInsights', symbol],
    queryFn: () => fetchAIInsights(symbol),
    refetchInterval: 30000,
  });

  if (isLoading) {
    return <LinearProgress />;
  }

  if (!insights) {
    return null;
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH':
        return theme.palette.success.main;
      case 'BEARISH':
        return theme.palette.error.main;
      default:
        return theme.palette.warning.main;
    }
  };

  return (
    <Card sx={{ height: '100%', bgcolor: 'background.paper' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          AI Analysis for {symbol}
        </Typography>

        {/* Sentiment and Confidence */}
        <Box sx={{ mb: 3 }}>
          <Stack direction="row" spacing={2} alignItems="center">
            <Chip
              icon={insights.analysis.sentiment === 'BULLISH' ? <TrendingUp /> : <TrendingDown />}
              label={insights.analysis.sentiment}
              color={insights.analysis.sentiment === 'BULLISH' ? 'success' : 'error'}
              sx={{ fontWeight: 'bold' }}
            />
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Confidence
              </Typography>
              <LinearProgress
                variant="determinate"
                value={insights.analysis.confidence * 100}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: theme.palette.grey[200],
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getSentimentColor(insights.analysis.sentiment),
                  },
                }}
              />
            </Box>
          </Stack>
        </Box>

        {/* Key Levels */}
        <Typography variant="subtitle1" gutterBottom>
          Key Price Levels
        </Typography>
        <List dense>
          {insights.levels.map((level, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                {level.type === 'SUPPORT' ? (
                  <Timeline color="success" />
                ) : (
                  <Timeline color="error" />
                )}
              </ListItemIcon>
              <ListItemText
                primary={`${level.type} at $${level.price.toFixed(2)}`}
                secondary={`Confidence: ${(level.confidence * 100).toFixed(1)}%`}
              />
            </ListItem>
          ))}
        </List>

        {/* Signals */}
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
          Active Signals
        </Typography>
        <List dense>
          {insights.signals.map((signal, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                <ShowChart color={signal.type === 'ENTRY' ? 'success' : 'warning'} />
              </ListItemIcon>
              <ListItemText
                primary={signal.label}
                secondary={`Price: $${signal.price.toFixed(2)} | Confidence: ${(
                  signal.confidence * 100
                ).toFixed(1)}%`}
              />
            </ListItem>
          ))}
        </List>

        {/* Pattern Analysis */}
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
          Detected Patterns
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {insights.analysis.patterns.map((pattern, index) => (
            <Chip
              key={index}
              icon={<Assessment />}
              label={pattern}
              size="small"
              color="primary"
              variant="outlined"
            />
          ))}
        </Stack>

        {/* Analysis Summary */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Analysis Summary
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {insights.analysis.summary}
          </Typography>
        </Box>

        {/* Risk Warning */}
        {insights.analysis.confidence < 0.7 && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'warning.light', borderRadius: 1 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Warning color="warning" />
              <Typography variant="body2" color="warning.dark">
                Low confidence signal - trade with caution
              </Typography>
            </Stack>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AIInsightsPanel; 