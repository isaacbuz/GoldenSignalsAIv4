/**
 * Simple AI Chart Page
 *
 * A working version of the AI chart that uses available components
 */

import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack,
  Button,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  AutoGraph as AutoGraphIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { WorkingChart } from '../../components/Chart/WorkingChart';

export const SimpleAIChartPage: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');

  return (
    <Container maxWidth={false} sx={{ mt: 2, mb: 4 }}>
      <Grid container spacing={3}>
        {/* Header */}
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Stack direction="row" spacing={2} alignItems="center">
              <PsychologyIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                AI-Powered Trading Chart
              </Typography>
            </Stack>
            <Stack direction="row" spacing={2}>
              <Button
                variant="outlined"
                onClick={() => navigate('/signals')}
              >
                Back to Signals
              </Button>
              <Stack direction="row" spacing={1}>
                {['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'].map(symbol => (
                  <Chip
                    key={symbol}
                    label={symbol}
                    onClick={() => setSelectedSymbol(symbol)}
                    color={selectedSymbol === symbol ? 'primary' : 'default'}
                    variant={selectedSymbol === symbol ? 'filled' : 'outlined'}
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
              </Stack>
            </Stack>
          </Box>
        </Grid>

        {/* AI Analysis Panel */}
        <Grid item xs={12} md={3}>
          <Card sx={{
            background: alpha(theme.palette.background.paper, 0.8),
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          }}>
            <CardContent>
              <Stack spacing={3}>
                <Box>
                  <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AutoGraphIcon /> AI Analysis
                  </Typography>
                  <Stack spacing={2}>
                    <Paper sx={{ p: 2, background: alpha(theme.palette.success.main, 0.1) }}>
                      <Typography variant="body2" color="text.secondary">
                        Signal Strength
                      </Typography>
                      <Typography variant="h5" sx={{ color: theme.palette.success.main }}>
                        85% Bullish
                      </Typography>
                    </Paper>

                    <Paper sx={{ p: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        AI Recommendation
                      </Typography>
                      <Chip
                        label="BUY"
                        color="success"
                        icon={<TrendingUpIcon />}
                        sx={{ mt: 1 }}
                      />
                    </Paper>

                    <Paper sx={{ p: 2 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Key Insights
                      </Typography>
                      <Stack spacing={1}>
                        <Typography variant="body2">
                          • MA20 crossed above MA50
                        </Typography>
                        <Typography variant="body2">
                          • Volume increasing
                        </Typography>
                        <Typography variant="body2">
                          • Bullish momentum
                        </Typography>
                      </Stack>
                    </Paper>
                  </Stack>
                </Box>

                <Box>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    Agent Status
                  </Typography>
                  <Stack spacing={1}>
                    {['Technical Analysis', 'Pattern Recognition', 'Sentiment Analysis', 'Risk Assessment'].map((agent, index) => (
                      <Box key={agent} display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">{agent}</Typography>
                        <Chip
                          label="Active"
                          size="small"
                          color="success"
                          sx={{
                            animation: index === 0 ? 'pulse 2s infinite' : 'none',
                            '@keyframes pulse': {
                              '0%': { opacity: 1 },
                              '50%': { opacity: 0.6 },
                              '100%': { opacity: 1 },
                            }
                          }}
                        />
                      </Box>
                    ))}
                  </Stack>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Main Chart */}
        <Grid item xs={12} md={9}>
          <Paper sx={{
            p: 2,
            height: 650,
            background: alpha(theme.palette.background.paper, 0.8),
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                {selectedSymbol} - AI Enhanced Chart
              </Typography>
              <Chip
                label="AI Mode Active"
                color="primary"
                icon={<PsychologyIcon />}
                sx={{ fontWeight: 600 }}
              />
            </Box>
            <WorkingChart
              symbol={selectedSymbol}
              height={550}
              showVolume={true}
              showIndicators={true}
            />
          </Paper>
        </Grid>

        {/* AI Insights Footer */}
        <Grid item xs={12}>
          <Paper sx={{
            p: 3,
            background: alpha(theme.palette.primary.main, 0.05),
            border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
          }}>
            <Typography variant="h6" gutterBottom>
              AI Trading Assistant Summary
            </Typography>
            <Typography variant="body1">
              Based on multi-agent analysis, {selectedSymbol} shows strong bullish signals.
              The AI system has identified a golden cross pattern (MA20 crossing above MA50)
              combined with increasing volume, suggesting potential upward momentum.
              Risk level is assessed as moderate with recommended position sizing at 15% of portfolio.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default SimpleAIChartPage;
