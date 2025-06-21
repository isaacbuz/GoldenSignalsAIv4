# 🚀 GoldenSignalsAI UI Implementation Plan

## Phase 1: Core Components (Week 1)

### 1. Alert System Foundation

#### A. Create Alert Context & Hook
```typescript
// src/contexts/AlertContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { Howl } from 'howler';

interface Alert {
  id: string;
  type: 'CALL' | 'PUT';
  symbol: string;
  confidence: number;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM';
  timestamp: Date;
  message: string;
  strike?: number;
  expiry?: string;
}

interface AlertContextType {
  alerts: Alert[];
  addAlert: (alert: Alert) => void;
  dismissAlert: (id: string) => void;
  settings: AlertSettings;
  updateSettings: (settings: Partial<AlertSettings>) => void;
}

const AlertContext = createContext<AlertContextType | null>(null);

export const useAlerts = () => {
  const context = useContext(AlertContext);
  if (!context) throw new Error('useAlerts must be used within AlertProvider');
  return context;
};

// Sound management
const sounds = {
  critical: new Howl({ src: ['/sounds/critical-alert.mp3'], volume: 0.8 }),
  high: new Howl({ src: ['/sounds/high-alert.mp3'], volume: 0.6 }),
  medium: new Howl({ src: ['/sounds/notification.mp3'], volume: 0.4 })
};
```

#### B. Signal Alert Component
```typescript
// src/components/Alerts/SignalAlert.tsx
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Box, Paper, Typography, IconButton, LinearProgress, Chip } from '@mui/material';
import { Close, TrendingUp, TrendingDown, Timer } from '@mui/icons-material';

interface SignalAlertProps {
  alert: Alert;
  onDismiss: () => void;
}

export const SignalAlert: React.FC<SignalAlertProps> = ({ alert, onDismiss }) => {
  const [timeLeft, setTimeLeft] = React.useState(30); // 30 second auto-dismiss

  React.useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          onDismiss();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [onDismiss]);

  return (
    <motion.div
      initial={{ x: 400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 400, opacity: 0 }}
      transition={{ type: "spring", damping: 25 }}
    >
      <Paper 
        sx={{ 
          p: 2, 
          mb: 2, 
          background: alert.priority === 'CRITICAL' 
            ? 'linear-gradient(45deg, #FF3B30 0%, #FF6B60 100%)' 
            : 'rgba(28, 28, 30, 0.95)',
          border: alert.priority === 'CRITICAL' ? '2px solid #FF3B30' : '1px solid rgba(255,255,255,0.1)',
          backdropFilter: 'blur(20px)',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <LinearProgress 
          variant="determinate" 
          value={(timeLeft / 30) * 100} 
          sx={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            right: 0, 
            height: 3,
            backgroundColor: 'rgba(255,255,255,0.1)',
            '& .MuiLinearProgress-bar': {
              backgroundColor: alert.priority === 'CRITICAL' ? '#FFF' : '#007AFF'
            }
          }} 
        />
        
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center" gap={2}>
            {alert.type === 'CALL' ? 
              <TrendingUp sx={{ fontSize: 32, color: '#00D632' }} /> : 
              <TrendingDown sx={{ fontSize: 32, color: '#FF3B30' }} />
            }
            
            <Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Typography variant="h6" fontWeight="bold">
                  {alert.symbol} {alert.type}
                </Typography>
                <Chip 
                  label={`${alert.confidence}%`} 
                  size="small" 
                  sx={{ 
                    backgroundColor: alert.confidence > 90 ? '#00D632' : '#FF9500',
                    color: '#000',
                    fontWeight: 'bold'
                  }} 
                />
              </Box>
              
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                {alert.message}
              </Typography>
              
              {alert.strike && (
                <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                  <Timer sx={{ fontSize: 14 }} />
                  <Typography variant="caption">
                    Strike: ${alert.strike} | Exp: {alert.expiry}
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
          
          <IconButton onClick={onDismiss} size="small">
            <Close />
          </IconButton>
        </Box>
      </Paper>
    </motion.div>
  );
};
```

### 2. Trend Predictor Component

```typescript
// src/components/Charts/TrendPredictor.tsx
import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

interface TrendPredictorProps {
  symbol: string;
  data: number[];
  predictions: number[];
  buyMarkers: { x: number; y: number; confidence: number }[];
  sellMarkers: { x: number; y: number; confidence: number }[];
}

export const TrendPredictor: React.FC<TrendPredictorProps> = ({
  symbol,
  data,
  predictions,
  buyMarkers,
  sellMarkers
}) => {
  const chartData = {
    labels: [...Array(data.length + predictions.length)].map((_, i) => i),
    datasets: [
      {
        label: 'Historical Price',
        data: [...data, ...Array(predictions.length).fill(null)],
        borderColor: '#007AFF',
        backgroundColor: 'rgba(0, 122, 255, 0.1)',
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2
      },
      {
        label: 'AI Prediction',
        data: [...Array(data.length - 1).fill(null), data[data.length - 1], ...predictions],
        borderColor: '#5E5CE6',
        backgroundColor: 'rgba(94, 92, 230, 0.1)',
        borderDash: [5, 5],
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
        fill: {
          target: 'origin',
          above: 'rgba(94, 92, 230, 0.1)'
        }
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        display: false
      },
      annotation: {
        annotations: [
          ...buyMarkers.map((marker, i) => ({
            type: 'point' as const,
            xValue: marker.x,
            yValue: marker.y,
            backgroundColor: '#00D632',
            radius: 8,
            borderColor: '#FFF',
            borderWidth: 2,
            label: {
              display: true,
              content: `Buy ${marker.confidence}%`,
              position: 'top' as const,
              backgroundColor: '#00D632',
              color: '#000',
              font: { weight: 'bold' as const }
            }
          })),
          ...sellMarkers.map((marker, i) => ({
            type: 'point' as const,
            xValue: marker.x,
            yValue: marker.y,
            backgroundColor: '#FF3B30',
            radius: 8,
            borderColor: '#FFF',
            borderWidth: 2,
            label: {
              display: true,
              content: `Sell ${marker.confidence}%`,
              position: 'bottom' as const,
              backgroundColor: '#FF3B30',
              color: '#FFF',
              font: { weight: 'bold' as const }
            }
          }))
        ]
      }
    },
    scales: {
      x: {
        display: false
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#A0A0A5'
        }
      }
    }
  };

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <Line data={chartData} options={options} />
    </div>
  );
};
```

### 3. New Dashboard Layout

```typescript
// src/pages/Dashboard/NewDashboard.tsx
import React from 'react';
import { Box, Container, Grid, Paper, Typography, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { TrendPredictor } from '../../components/Charts/TrendPredictor';
import { SignalStream } from '../../components/Signals/SignalStream';
import { PerformanceMetrics } from '../../components/Metrics/PerformanceMetrics';
import { ActiveAlert } from '../../components/Alerts/ActiveAlert';
import { useAlerts } from '../../contexts/AlertContext';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';

export const NewDashboard: React.FC = () => {
  const { alerts } = useAlerts();
  const activeAlert = alerts.find(a => a.priority === 'CRITICAL');

  // Fetch real-time data
  const { data: marketData } = useQuery({
    queryKey: ['dashboard-market-data'],
    queryFn: () => apiClient.getDashboardData(),
    refetchInterval: 5000 // Update every 5 seconds
  });

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: '#000000', pt: 2 }}>
      <Container maxWidth="xl">
        {/* Active Alert Banner */}
        {activeAlert && (
          <motion.div
            initial={{ y: -100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ type: "spring", damping: 20 }}
          >
            <Paper 
              sx={{ 
                p: 3, 
                mb: 3,
                background: 'linear-gradient(90deg, #007AFF 0%, #5E5CE6 100%)',
                border: 'none',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h5" fontWeight="bold" gutterBottom>
                    🎯 Active Signal Alert
                  </Typography>
                  <Typography variant="h6">
                    {activeAlert.symbol} {activeAlert.type} Option - {activeAlert.confidence}% Confidence
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    Entry: Now | Strike: ${activeAlert.strike} | Expires: {activeAlert.expiry}
                  </Typography>
                </Box>
                <Box textAlign="right">
                  <Chip 
                    label="ACT NOW" 
                    sx={{ 
                      backgroundColor: '#FFF', 
                      color: '#007AFF',
                      fontWeight: 'bold',
                      fontSize: '1rem',
                      py: 3,
                      px: 2
                    }} 
                  />
                </Box>
              </Box>
              
              {/* Animated pulse effect */}
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at center, rgba(255,255,255,0.2) 0%, transparent 70%)',
                  animation: 'pulse 2s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(0.8)', opacity: 0 },
                    '50%': { transform: 'scale(1.2)', opacity: 0.3 },
                    '100%': { transform: 'scale(0.8)', opacity: 0 }
                  }
                }}
              />
            </Paper>
          </motion.div>
        )}

        {/* Main Content Grid */}
        <Grid container spacing={3}>
          {/* Trend Predictor - Main Focus */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 3, height: 500, backgroundColor: 'rgba(28, 28, 30, 0.6)' }}>
              <Typography variant="h6" gutterBottom>
                AI Trend Prediction
              </Typography>
              <Box sx={{ height: 'calc(100% - 40px)' }}>
                <TrendPredictor
                  symbol="AAPL"
                  data={marketData?.priceHistory || []}
                  predictions={marketData?.predictions || []}
                  buyMarkers={marketData?.buySignals || []}
                  sellMarkers={marketData?.sellSignals || []}
                />
              </Box>
            </Paper>
          </Grid>

          {/* Signal Stream */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3, height: 500, backgroundColor: 'rgba(28, 28, 30, 0.6)' }}>
              <SignalStream />
            </Paper>
          </Grid>

          {/* Performance Metrics */}
          <Grid item xs={12}>
            <PerformanceMetrics />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};
```

### 4. Confidence Bar Component

```typescript
// src/components/UI/ConfidenceBar.tsx
import React from 'react';
import { Box, Typography } from '@mui/material';
import { motion } from 'framer-motion';

interface ConfidenceBarProps {
  value: number;
  showPulse?: boolean;
  color?: 'success' | 'warning' | 'error';
  height?: number;
  showLabel?: boolean;
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  value,
  showPulse = false,
  color = 'success',
  height = 8,
  showLabel = true
}) => {
  const getColor = () => {
    if (color) {
      return color === 'success' ? '#00D632' : color === 'warning' ? '#FF9500' : '#FF3B30';
    }
    return value >= 80 ? '#00D632' : value >= 60 ? '#FF9500' : '#FF3B30';
  };

  const barColor = getColor();

  return (
    <Box sx={{ width: '100%' }}>
      {showLabel && (
        <Box display="flex" justifyContent="space-between" mb={0.5}>
          <Typography variant="caption" color="text.secondary">
            Confidence
          </Typography>
          <Typography variant="caption" fontWeight="bold" color={barColor}>
            {value}%
          </Typography>
        </Box>
      )}
      <Box 
        sx={{ 
          width: '100%', 
          height, 
          backgroundColor: 'rgba(255,255,255,0.1)', 
          borderRadius: height / 2,
          overflow: 'hidden',
          position: 'relative'
        }}
      >
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          style={{
            height: '100%',
            backgroundColor: barColor,
            borderRadius: height / 2,
            position: 'relative'
          }}
        >
          {showPulse && value >= 80 && (
            <Box
              sx={{
                position: 'absolute',
                right: 0,
                top: '50%',
                transform: 'translateY(-50%)',
                width: height * 2,
                height: height * 2,
                backgroundColor: barColor,
                borderRadius: '50%',
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%': { transform: 'translateY(-50%) scale(0.8)', opacity: 1 },
                  '50%': { transform: 'translateY(-50%) scale(1.5)', opacity: 0.5 },
                  '100%': { transform: 'translateY(-50%) scale(0.8)', opacity: 1 }
                }
              }}
            />
          )}
        </motion.div>
      </Box>
    </Box>
  );
};
```

## Phase 2: Enhanced Signal Cards & Feed

### 1. Option Signal Card
```typescript
// src/components/Signals/OptionSignalCard.tsx
import React from 'react';
import { Card, CardContent, Box, Typography, Button, Chip, Stack } from '@mui/material';
import { TrendingUp, TrendingDown, Timer, Notifications } from '@mui/icons-material';
import { ConfidenceBar } from '../UI/ConfidenceBar';
import { motion } from 'framer-motion';

interface OptionSignal {
  id: string;
  symbol: string;
  type: 'CALL' | 'PUT';
  strike: number;
  expiry: string;
  confidence: number;
  currentPrice: number;
  entryPrice: number;
  targetReturn: string;
  reasoning: string;
  timestamp: Date;
  urgency: 'HIGH' | 'MEDIUM' | 'LOW';
}

export const OptionSignalCard: React.FC<{ signal: OptionSignal }> = ({ signal }) => {
  const Icon = signal.type === 'CALL' ? TrendingUp : TrendingDown;
  const iconColor = signal.type === 'CALL' ? '#00D632' : '#FF3B30';
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.3 }}
    >
      <Card 
        sx={{ 
          backgroundColor: 'rgba(28, 28, 30, 0.8)',
          border: signal.confidence > 85 ? '2px solid' : '1px solid rgba(255,255,255,0.1)',
          borderColor: signal.confidence > 85 ? iconColor : 'rgba(255,255,255,0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {signal.urgency === 'HIGH' && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: 3,
              background: `linear-gradient(90deg, ${iconColor} 0%, transparent 100%)`,
              animation: 'slide 2s linear infinite',
              '@keyframes slide': {
                '0%': { transform: 'translateX(-100%)' },
                '100%': { transform: 'translateX(100%)' }
              }
            }}
          />
        )}
        
        <CardContent>
          <Stack spacing={2}>
            {/* Header */}
            <Box display="flex" justifyContent="space-between" alignItems="flex-start">
              <Box display="flex" alignItems="center" gap={1}>
                <Icon sx={{ fontSize: 28, color: iconColor }} />
                <Box>
                  <Typography variant="h6" fontWeight="bold">
                    {signal.symbol}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {signal.type} Option
                  </Typography>
                </Box>
              </Box>
              
              <Stack direction="row" spacing={1}>
                {signal.urgency === 'HIGH' && (
                  <Chip 
                    label="URGENT" 
                    size="small" 
                    sx={{ 
                      backgroundColor: '#FF3B30',
                      color: '#FFF',
                      fontWeight: 'bold',
                      animation: 'blink 1s ease-in-out infinite',
                      '@keyframes blink': {
                        '0%, 100%': { opacity: 1 },
                        '50%': { opacity: 0.5 }
                      }
                    }} 
                  />
                )}
                <Chip 
                  label={`+${signal.targetReturn}`} 
                  size="small" 
                  sx={{ backgroundColor: iconColor, color: '#000', fontWeight: 'bold' }} 
                />
              </Stack>
            </Box>

            {/* Confidence */}
            <ConfidenceBar value={signal.confidence} showPulse={signal.confidence > 85} />

            {/* Details */}
            <Box display="flex" justifyContent="space-between">
              <Box>
                <Typography variant="caption" color="text.secondary">Strike Price</Typography>
                <Typography variant="body1" fontWeight="bold">${signal.strike}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Entry Price</Typography>
                <Typography variant="body1" fontWeight="bold">${signal.entryPrice}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Expiry</Typography>
                <Typography variant="body1" fontWeight="bold">{signal.expiry}</Typography>
              </Box>
            </Box>

            {/* Reasoning */}
            <Typography variant="body2" color="text.secondary">
              {signal.reasoning}
            </Typography>

            {/* Actions */}
            <Stack direction="row" spacing={1}>
              <Button 
                variant="contained" 
                fullWidth
                startIcon={<Notifications />}
                sx={{ 
                  backgroundColor: iconColor,
                  color: signal.type === 'CALL' ? '#000' : '#FFF',
                  '&:hover': {
                    backgroundColor: iconColor,
                    filter: 'brightness(0.9)'
                  }
                }}
              >
                Set Alert
              </Button>
              <Button 
                variant="outlined" 
                fullWidth
                sx={{ borderColor: 'rgba(255,255,255,0.2)' }}
              >
                View Analysis
              </Button>
            </Stack>

            {/* Time indicator */}
            <Box display="flex" alignItems="center" gap={0.5} justifyContent="center">
              <Timer sx={{ fontSize: 14, color: 'text.secondary' }} />
              <Typography variant="caption" color="text.secondary">
                {new Date(signal.timestamp).toLocaleTimeString()}
              </Typography>
            </Box>
          </Stack>
        </CardContent>
      </Card>
    </motion.div>
  );
};
```

### 2. Signal Feed Settings
```typescript
// src/components/Settings/SignalSettings.tsx
import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Switch,
  Slider,
  FormControlLabel,
  Select,
  MenuItem,
  Divider,
  Stack
} from '@mui/material';

export const SignalSettings: React.FC = () => {
  const [settings, setSettings] = React.useState({
    autoRefresh: true,
    refreshInterval: 60, // seconds
    minConfidence: 70,
    signalTypes: ['CALL', 'PUT'],
    alertSound: true,
    pushNotifications: true,
    emailAlerts: false,
    criticalOnly: false
  });

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Signal Preferences
      </Typography>
      
      <Stack spacing={3} sx={{ mt: 3 }}>
        {/* Auto Refresh */}
        <Box>
          <FormControlLabel
            control={
              <Switch 
                checked={settings.autoRefresh}
                onChange={(e) => setSettings({ ...settings, autoRefresh: e.target.checked })}
              />
            }
            label="Auto-refresh signals"
          />
          {settings.autoRefresh && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                Refresh every {settings.refreshInterval} seconds
              </Typography>
              <Slider
                value={settings.refreshInterval}
                onChange={(_, value) => setSettings({ ...settings, refreshInterval: value as number })}
                min={10}
                max={300}
                step={10}
                marks={[
                  { value: 10, label: '10s' },
                  { value: 60, label: '1m' },
                  { value: 300, label: '5m' }
                ]}
              />
            </Box>
          )}
        </Box>

        <Divider />

        {/* Confidence Threshold */}
        <Box>
          <Typography variant="body2" gutterBottom>
            Minimum confidence level: {settings.minConfidence}%
          </Typography>
          <Slider
            value={settings.minConfidence}
            onChange={(_, value) => setSettings({ ...settings, minConfidence: value as number })}
            min={50}
            max={95}
            step={5}
            marks={[
              { value: 50, label: '50%' },
              { value: 70, label: '70%' },
              { value: 85, label: '85%' },
              { value: 95, label: '95%' }
            ]}
            sx={{
              '& .MuiSlider-mark': {
                backgroundColor: 'rgba(255,255,255,0.2)'
              }
            }}
          />
        </Box>

        <Divider />

        {/* Alert Preferences */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Alert Channels
          </Typography>
          <Stack spacing={1}>
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.alertSound}
                  onChange={(e) => setSettings({ ...settings, alertSound: e.target.checked })}
                />
              }
              label="Sound alerts"
            />
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.pushNotifications}
                  onChange={(e) => setSettings({ ...settings, pushNotifications: e.target.checked })}
                />
              }
              label="Push notifications"
            />
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.emailAlerts}
                  onChange={(e) => setSettings({ ...settings, emailAlerts: e.target.checked })}
                />
              }
              label="Email alerts"
            />
            <FormControlLabel
              control={
                <Switch 
                  checked={settings.criticalOnly}
                  onChange={(e) => setSettings({ ...settings, criticalOnly: e.target.checked })}
                />
              }
              label="Critical alerts only (>90% confidence)"
            />
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
};
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Set up alert system with sound
- [ ] Create new dashboard layout
- [ ] Build confidence bar component
- [ ] Implement basic signal cards

### Week 2: Core Features  
- [ ] Trend predictor with Chart.js
- [ ] Signal feed with filtering
- [ ] Alert management center
- [ ] Settings page

### Week 3: Visualizations
- [ ] Predictive trendlines
- [ ] Buy/sell markers
- [ ] Option chain visualizer
- [ ] Performance charts

### Week 4: Polish
- [ ] Animations & transitions
- [ ] Mobile responsive design
- [ ] PWA configuration
- [ ] Performance optimization

## Next Steps

1. **Install Dependencies**:
```bash
cd frontend
npm install framer-motion howler chart.js react-chartjs-2 chartjs-plugin-annotation
npm install @mui/x-charts web-push
```

2. **Create Alert Sounds**:
- Place sound files in `public/sounds/`
- critical-alert.mp3
- high-alert.mp3
- notification.mp3

3. **Update API Endpoints**:
- Add WebSocket support for real-time signals
- Create alert preferences endpoint
- Add signal filtering parameters

4. **Configure Service Worker**:
- Enable push notifications
- Background sync for signals
- Offline support

This implementation plan provides a clear roadmap to transform GoldenSignalsAI into the AI trading assistant vision! 