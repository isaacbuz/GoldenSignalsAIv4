/**
 * AI Dashboard - The Command Center for GoldenSignalsAI
 * Where AI intelligence meets trading opportunity
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Stack,
  Chip,
  useTheme,
  alpha,
  Button,
  IconButton
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Psychology, 
  TrendingUp, 
  Notifications,
  BoltOutlined,
  AutoGraph,
  SignalCellularAlt
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { AIBrainDashboard } from '../../components/AI/AIBrainDashboard';
import { AISignalCard } from '../../components/AI/AISignalCard';
import { AIPredictionChart } from '../../components/AI/AIPredictionChart';
import { useAlerts } from '../../contexts/AlertContext';
import ErrorBoundary from '../../components/ErrorBoundary';

// Mock AI Signal type (replace with your actual type)
interface AISignal {
  id: string;
  symbol: string;
  type: 'CALL' | 'PUT';
  strike: number;
  expiry: string;
  confidence: number;
  currentPrice: number;
  entryPrice: number;
  predictedMove: string;
  timeframe: string;
  reasoning: string;
  agentConsensus: {
    total: number;
    agreeing: number;
  };
  historicalAccuracy: number;
  urgency: 'CRITICAL' | 'HIGH' | 'MEDIUM';
  entryWindow: number;
  patterns: string[];
}

export const AIDashboard: React.FC = () => {
  const theme = useTheme();
  const { addAlert } = useAlerts();
  const [activeSignal, setActiveSignal] = useState<AISignal | null>(null);

  // Fetch AI signals
  const { data: aiSignals, isLoading } = useQuery({
    queryKey: ['ai-signals'],
    queryFn: async () => {
      // For now, return mock data - replace with actual API call
      return [
        {
          id: 'AI-1247',
          symbol: 'NVDA',
          type: 'CALL' as const,
          strike: 720,
          expiry: '3/22',
          confidence: 94,
          currentPrice: 700.50,
          entryPrice: 12.50,
          predictedMove: '+32%',
          timeframe: '48 hours',
          reasoning: 'Detected bullish divergence pattern with 94% historical accuracy. Volume accumulation suggests institutional buying. 14 of 19 AI agents confirm breakout imminent.',
          agentConsensus: { total: 19, agreeing: 14 },
          historicalAccuracy: 87,
          urgency: 'CRITICAL' as const,
          entryWindow: 30,
          patterns: ['Bullish Divergence', 'Volume Accumulation', 'Support Break', 'Options Flow']
        },
        {
          id: 'AI-1248',
          symbol: 'TSLA',
          type: 'PUT' as const,
          strike: 165,
          expiry: '3/29',
          confidence: 88,
          currentPrice: 170.25,
          entryPrice: 8.75,
          predictedMove: '-18%',
          timeframe: '5 days',
          reasoning: 'AI detected weakness in momentum indicators. Bearish pattern formation with high probability of breakdown. Smart money positioning for downside.',
          agentConsensus: { total: 19, agreeing: 12 },
          historicalAccuracy: 82,
          urgency: 'HIGH' as const,
          entryWindow: 45,
          patterns: ['Head & Shoulders', 'RSI Divergence', 'Volume Decline', 'Resistance Rejection']
        }
      ] as AISignal[];
    },
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  // Get AI metrics
  const { data: aiMetrics } = useQuery({
    queryKey: ['ai-metrics'],
    queryFn: async () => ({
      signalsProcessing: 47,
      patternsDetected: 234,
      winRate: 87,
      activeAgents: 19
    }),
    refetchInterval: 5000
  });

  // Set active signal (highest confidence)
  useEffect(() => {
    if (aiSignals && aiSignals.length > 0) {
      const highestConfidence = aiSignals.reduce((prev, current) => 
        (current.confidence > prev.confidence) ? current : prev
      );
      setActiveSignal(highestConfidence);
    }
  }, [aiSignals]);

  const handleExecuteTrade = (signal: AISignal) => {
    // Add to alerts
    addAlert({
      id: signal.id,
      type: signal.type,
      symbol: signal.symbol,
      confidence: signal.confidence,
      priority: signal.urgency,
      timestamp: new Date(),
      message: `Execute ${signal.type} option trade`,
      strike: signal.strike,
      expiry: signal.expiry
    });
    
    // In real app, this would open broker integration
    console.log('Executing trade:', signal);
  };

  return (
    <Box sx={{ 
      minHeight: '100vh', 
      backgroundColor: '#000000',
      backgroundImage: 'radial-gradient(circle at 20% 50%, rgba(0, 166, 255, 0.1) 0%, transparent 50%)',
      py: 3 
    }}>
      <Container maxWidth="xl">
        {/* Hero Alert Section */}
        {activeSignal && activeSignal.urgency === 'CRITICAL' && (
          <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", damping: 20 }}
          >
            <Paper
              sx={{
                mb: 3,
                p: 3,
                background: `linear-gradient(135deg, ${alpha('#00FF88', 0.2)} 0%, ${alpha('#00A6FF', 0.2)} 100%)`,
                border: '2px solid #00FF88',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at center, rgba(0,255,136,0.1) 0%, transparent 70%)',
                  animation: 'pulse 2s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(0.8)', opacity: 0 },
                    '50%': { transform: 'scale(1.2)', opacity: 0.3 },
                    '100%': { transform: 'scale(0.8)', opacity: 0 }
                  }
                }}
              />
              
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ position: 'relative', zIndex: 1 }}>
                <Box>
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Psychology sx={{ fontSize: 48, color: '#00FF88' }} />
                    <Box>
                      <Typography variant="h4" fontWeight="bold">
                        🎯 AI SIGNAL DETECTED
                      </Typography>
                      <Typography variant="h5" sx={{ mt: 1 }}>
                        {activeSignal.symbol} {activeSignal.type} - {activeSignal.confidence}% Confidence
                      </Typography>
                      <Typography variant="body1" sx={{ mt: 1, opacity: 0.9 }}>
                        AI Prediction: {activeSignal.predictedMove} in {activeSignal.timeframe} | Entry: NOW | Strike: ${activeSignal.strike}
                      </Typography>
                    </Box>
                  </Stack>
                </Box>
                
                <Stack spacing={1}>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<BoltOutlined />}
                    onClick={() => handleExecuteTrade(activeSignal)}
                    sx={{
                      backgroundColor: '#00FF88',
                      color: '#000',
                      fontWeight: 'bold',
                      fontSize: '1.1rem',
                      px: 4,
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: '#00FF88',
                        filter: 'brightness(0.9)',
                        transform: 'scale(1.05)'
                      }
                    }}
                  >
                    ONE-CLICK TRADE
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<Notifications />}
                    sx={{
                      borderColor: 'rgba(255,255,255,0.3)',
                      color: '#FFF'
                    }}
                  >
                    Set Alert
                  </Button>
                </Stack>
              </Stack>
            </Paper>
          </motion.div>
        )}

        {/* Main Grid */}
        <Grid container spacing={3}>
          {/* AI Brain Status */}
          {/* <Grid item xs={12}>
            <AIBrainDashboard
              activeAgents={aiMetrics?.activeAgents || 19}
              signalsProcessing={aiMetrics?.signalsProcessing || 0}
              patternsDetected={aiMetrics?.patternsDetected || 0}
              confidenceThreshold={85}
              winRate={aiMetrics?.winRate || 87}
            />
          </Grid> */}

          {/* Active AI Signals */}
          <Grid item xs={12} lg={7}>
            <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
              <Stack direction="row" spacing={1} alignItems="center">
                <SignalCellularAlt sx={{ color: '#00FF88' }} />
                <span>Active AI Signals</span>
                <Chip 
                  label="LIVE" 
                  size="small" 
                  sx={{ 
                    backgroundColor: '#00FF88',
                    color: '#000',
                    fontWeight: 'bold',
                    animation: 'blink 2s ease-in-out infinite',
                    '@keyframes blink': {
                      '0%, 100%': { opacity: 1 },
                      '50%': { opacity: 0.7 }
                    }
                  }} 
                />
              </Stack>
            </Typography>
            
            <Stack spacing={3}>
              <AnimatePresence>
                {aiSignals?.map((signal, index) => (
                  <motion.div
                    key={signal.id}
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 50 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <AISignalCard
                      signal={signal}
                      onExecute={handleExecuteTrade}
                      onDismiss={() => {}}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {!isLoading && (!aiSignals || aiSignals.length === 0) && (
                <Paper sx={{ p: 4, textAlign: 'center', backgroundColor: 'rgba(28, 28, 30, 0.6)' }}>
                  <Psychology sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">
                    AI is scanning markets for high-confidence signals...
                  </Typography>
                </Paper>
              )}
            </Stack>
          </Grid>

          {/* AI Prediction Chart */}
          <Grid item xs={12} lg={5}>
            <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3 }}>
              <Stack direction="row" spacing={1} alignItems="center">
                <AutoGraph sx={{ color: '#8B5CF6' }} />
                <span>AI Prediction Engine</span>
              </Stack>
            </Typography>
            
            <ErrorBoundary>
              <Paper sx={{ 
                p: 3, 
                height: 400,
                backgroundColor: 'rgba(28, 28, 30, 0.6)',
                border: '1px solid rgba(139,92,246,0.3)'
              }}>
                {activeSignal && (
                  <AIPredictionChart
                    symbol={activeSignal.symbol}
                    currentPrice={activeSignal.currentPrice}
                    historicalData={[]}
                    prediction={{
                      target: activeSignal.currentPrice * (1 + (parseFloat(activeSignal.predictedMove) / 100)),
                      confidence: activeSignal.confidence,
                      timeframe: activeSignal.timeframe,
                      upperBand: activeSignal.currentPrice * (1 + (parseFloat(activeSignal.predictedMove) / 100) * 1.1),
                      lowerBand: activeSignal.currentPrice * (1 + (parseFloat(activeSignal.predictedMove) / 100) * 0.9),
                      supportLevels: [activeSignal.currentPrice * 0.95, activeSignal.currentPrice * 0.90],
                      resistanceLevels: [activeSignal.currentPrice * 1.05, activeSignal.currentPrice * 1.10]
                    }}
                  />
                )}
              </Paper>
            </ErrorBoundary>

            {/* AI Performance Stats */}
            <Paper sx={{ 
              mt: 3,
              p: 3,
              backgroundColor: 'rgba(28, 28, 30, 0.6)',
              border: '1px solid rgba(139,92,246,0.3)'
            }}>
              <Typography variant="h6" gutterBottom>
                AI Performance Proof
              </Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}>
                  <Stack spacing={1}>
                    <Typography variant="body2" color="text.secondary">Win Rate</Typography>
                    <Typography variant="h4" fontWeight="bold" color="#00FF88">87%</Typography>
                    <Typography variant="caption" color="text.secondary">Last 30 days</Typography>
                  </Stack>
                </Grid>
                <Grid item xs={6}>
                  <Stack spacing={1}>
                    <Typography variant="body2" color="text.secondary">Avg Return</Typography>
                    <Typography variant="h4" fontWeight="bold" color="#00FF88">+24%</Typography>
                    <Typography variant="caption" color="text.secondary">Per signal</Typography>
                  </Stack>
                </Grid>
                <Grid item xs={6}>
                  <Stack spacing={1}>
                    <Typography variant="body2" color="text.secondary">Total Signals</Typography>
                    <Typography variant="h4" fontWeight="bold">1,247</Typography>
                    <Typography variant="caption" color="text.secondary">This month</Typography>
                  </Stack>
                </Grid>
                <Grid item xs={6}>
                  <Stack spacing={1}>
                    <Typography variant="body2" color="text.secondary">Success Rate</Typography>
                    <Typography variant="h4" fontWeight="bold">91%</Typography>
                    <Typography variant="caption" color="text.secondary">Options trades</Typography>
                  </Stack>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}; 