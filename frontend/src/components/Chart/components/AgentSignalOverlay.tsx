/**
 * Agent signal overlay component
 * Displays AI agent analysis results on the chart
 */

import React, { useEffect, useRef } from 'react';
import { Box, Typography, Chip, Paper, LinearProgress } from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import {
  TrendingUp as BuyIcon,
  TrendingDown as SellIcon,
  Remove as HoldIcon,
  Psychology as AIIcon,
} from '@mui/icons-material';
import { IChartApi } from 'lightweight-charts';
import { AgentSignal, ConsensusDecision } from '../hooks/useAgentAnalysis';

const OverlayContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 20,
  right: 20,
  maxWidth: 300,
  zIndex: 100,
}));

const SignalCard = styled(Paper)(({ theme }) => ({
  backgroundColor: alpha('#1e222d', 0.95),
  backdropFilter: 'blur(10px)',
  border: '1px solid #2a2e39',
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const ConsensusCard = styled(SignalCard)(({ theme }) => ({
  borderColor: '#2196f3',
  borderWidth: 2,
}));

const SignalChip = styled(Chip)<{ signal: string }>(({ theme, signal }) => ({
  fontWeight: 600,
  ...(signal === 'BUY' && {
    backgroundColor: alpha('#26a69a', 0.2),
    color: '#26a69a',
    '& .MuiChip-icon': { color: '#26a69a' },
  }),
  ...(signal === 'SELL' && {
    backgroundColor: alpha('#ef5350', 0.2),
    color: '#ef5350',
    '& .MuiChip-icon': { color: '#ef5350' },
  }),
  ...(signal === 'HOLD' && {
    backgroundColor: alpha('#ffc107', 0.2),
    color: '#ffc107',
    '& .MuiChip-icon': { color: '#ffc107' },
  }),
}));

const ConfidenceBar = styled(LinearProgress)(({ theme }) => ({
  height: 6,
  borderRadius: 3,
  backgroundColor: alpha('#ffffff', 0.1),
  '& .MuiLinearProgress-bar': {
    borderRadius: 3,
  },
}));

interface AgentSignalOverlayProps {
  signals: AgentSignal[] | null;
  consensus: ConsensusDecision | null;
  isAnalyzing: boolean;
  chart: IChartApi;
}

export const AgentSignalOverlay: React.FC<AgentSignalOverlayProps> = ({
  signals,
  consensus,
  isAnalyzing,
  chart,
}) => {
  const priceLineRef = useRef<any>(null);

  // Draw trading levels on chart
  useEffect(() => {
    if (!consensus || !chart) return;

    const series = chart.timeScale();

    // Clear previous lines
    if (priceLineRef.current) {
      series.removePriceLine(priceLineRef.current);
    }

    // Add entry price line
    if (consensus.entry_price) {
      const mainSeries = chart.getSeries()[0]; // Assuming first series is candlestick
      if (mainSeries) {
        priceLineRef.current = mainSeries.createPriceLine({
          price: consensus.entry_price,
          color: consensus.signal === 'BUY' ? '#26a69a' : '#ef5350',
          lineWidth: 2,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: `Entry: $${consensus.entry_price.toFixed(2)}`,
        });
      }
    }

    return () => {
      if (priceLineRef.current && chart) {
        const mainSeries = chart.getSeries()[0];
        if (mainSeries) {
          mainSeries.removePriceLine(priceLineRef.current);
        }
      }
    };
  }, [consensus, chart]);

  if (!signals && !isAnalyzing) return null;

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <BuyIcon fontSize="small" />;
      case 'SELL': return <SellIcon fontSize="small" />;
      default: return <HoldIcon fontSize="small" />;
    }
  };

  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return '#26a69a';
    if (risk < 0.7) return '#ffc107';
    return '#ef5350';
  };

  return (
    <OverlayContainer>
      {isAnalyzing && (
        <SignalCard>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <AIIcon sx={{ color: '#2196f3' }} />
            <Typography variant="subtitle2" sx={{ color: '#d1d4dc', fontWeight: 600 }}>
              Analyzing Market...
            </Typography>
          </Box>
          <LinearProgress sx={{ mt: 1 }} />
        </SignalCard>
      )}

      {consensus && (
        <ConsensusCard>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="subtitle2" sx={{ color: '#d1d4dc', fontWeight: 600 }}>
              AI Consensus
            </Typography>
            <SignalChip
              label={consensus.signal}
              icon={getSignalIcon(consensus.signal)}
              signal={consensus.signal}
              size="small"
            />
          </Box>

          <Box mb={2}>
            <Box display="flex" justifyContent="space-between" mb={0.5}>
              <Typography variant="caption" sx={{ color: '#787b86' }}>
                Confidence
              </Typography>
              <Typography variant="caption" sx={{ color: '#d1d4dc', fontWeight: 600 }}>
                {Math.round(consensus.confidence * 100)}%
              </Typography>
            </Box>
            <ConfidenceBar
              variant="determinate"
              value={consensus.confidence * 100}
              sx={{
                '& .MuiLinearProgress-bar': {
                  backgroundColor: consensus.confidence > 0.7 ? '#26a69a' :
                                  consensus.confidence > 0.4 ? '#ffc107' : '#ef5350',
                },
              }}
            />
          </Box>

          {consensus.entry_price && (
            <Box display="flex" flexDirection="column" gap={0.5}>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="caption" sx={{ color: '#787b86' }}>
                  Entry Price
                </Typography>
                <Typography variant="caption" sx={{ color: '#d1d4dc', fontWeight: 600 }}>
                  ${consensus.entry_price.toFixed(2)}
                </Typography>
              </Box>
              {consensus.stop_loss && (
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="caption" sx={{ color: '#787b86' }}>
                    Stop Loss
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#ef5350', fontWeight: 600 }}>
                    ${consensus.stop_loss.toFixed(2)}
                  </Typography>
                </Box>
              )}
              {consensus.take_profit && (
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="caption" sx={{ color: '#787b86' }}>
                    Take Profit
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#26a69a', fontWeight: 600 }}>
                    ${consensus.take_profit.toFixed(2)}
                  </Typography>
                </Box>
              )}
            </Box>
          )}

          <Box display="flex" justifyContent="space-between" mt={2}>
            <Box display="flex" alignItems="center" gap={0.5}>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: getRiskColor(consensus.risk_score),
                }}
              />
              <Typography variant="caption" sx={{ color: '#787b86' }}>
                Risk: {consensus.risk_score < 0.3 ? 'Low' : consensus.risk_score < 0.7 ? 'Medium' : 'High'}
              </Typography>
            </Box>
            <Typography variant="caption" sx={{ color: '#787b86' }}>
              {consensus.supporting_agents} for / {consensus.opposing_agents} against
            </Typography>
          </Box>
        </ConsensusCard>
      )}

      {signals && signals.length > 0 && (
        <SignalCard>
          <Typography variant="subtitle2" sx={{ color: '#d1d4dc', fontWeight: 600, mb: 2 }}>
            Agent Signals
          </Typography>
          <Box display="flex" flexDirection="column" gap={1}>
            {signals.slice(0, 5).map((signal, index) => (
              <Box key={index} display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="caption" sx={{ color: '#787b86' }}>
                  {signal.agent}
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="caption" sx={{ color: '#787b86' }}>
                    {Math.round(signal.confidence * 100)}%
                  </Typography>
                  <SignalChip
                    label={signal.signal}
                    signal={signal.signal}
                    size="small"
                    sx={{ height: 20, '& .MuiChip-label': { px: 1, fontSize: '0.7rem' } }}
                  />
                </Box>
              </Box>
            ))}
          </Box>
        </SignalCard>
      )}
    </OverlayContainer>
  );
};
