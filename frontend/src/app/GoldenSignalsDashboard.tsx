/**
 * Golden Signals Dashboard
 * 
 * Simplified main dashboard that brings together all components
 * in a clean, stable interface with the Golden Eye AI Prophet theme.
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
    Box,
    Container,
    Grid,
    Card,
    CardContent,
    Typography,
    Chip,
    Stack,
    IconButton,
    TextField,
    Autocomplete,
    alpha,
    LinearProgress,
    Paper,
    Avatar,
} from '@mui/material';
import {
    Search as SearchIcon,
    TrendingUp,
    TrendingDown,
    Remove,
    Wifi,
    WifiOff,
    Psychology,
    Speed,
    Timeline,
    Refresh,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

// New simplified components
import { useAppStore, useSignals, useSelectedSymbol, useWSConnected } from '../store/appStore';
import { useStableWebSocket } from '../services/stableWebSocket';
import { goldenEyeColors, goldenEyeGradients, getGlassmorphismStyle, getSignalColor, getConfidenceColor } from '../theme/goldenEye';
import GoldenEyeWidget from '../components/ai-prophet/GoldenEyeWidget';
import GoldenEyeChat from '../components/ai-prophet/GoldenEyeChat';

// === STYLED COMPONENTS ===
const DashboardContainer = styled(Box)({
    minHeight: '100vh',
    background: `linear-gradient(135deg, ${goldenEyeColors.background} 0%, ${alpha(goldenEyeColors.surface, 0.8)} 100%)`,
    paddingTop: '24px',
    paddingBottom: '24px',
});

const HeaderCard = styled(Card)({
    ...getGlassmorphismStyle(0.2),
    marginBottom: '24px',
    padding: '8px',
});

const SignalCard = styled(Card)({
    ...getGlassmorphismStyle(0.1),
    height: '100%',
    transition: 'all 0.3s ease',
    cursor: 'pointer',
    '&:hover': {
        borderColor: goldenEyeColors.primary,
        boxShadow: `0 8px 32px ${alpha(goldenEyeColors.primary, 0.2)}`,
        transform: 'translateY(-4px)',
    },
});

const StatusIndicator = styled(Box)<{ connected: boolean }>(({ connected }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    borderRadius: '20px',
    background: connected
        ? alpha(goldenEyeColors.bullish, 0.1)
        : alpha(goldenEyeColors.bearish, 0.1),
    border: `1px solid ${connected ? goldenEyeColors.bullish : goldenEyeColors.bearish}`,
}));

// === MOCK DATA ===
const POPULAR_SYMBOLS = [
    { symbol: 'SPY', name: 'SPDR S&P 500', price: 456.78 },
    { symbol: 'QQQ', name: 'Invesco QQQ Trust', price: 389.45 },
    { symbol: 'AAPL', name: 'Apple Inc.', price: 189.25 },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 267.89 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.56 },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 489.12 },
];

// === COMPONENT ===
export const GoldenSignalsDashboard: React.FC = () => {
    const [mockSignals, setMockSignals] = useState<any[]>([]);

    // Store state
    const selectedSymbol = useSelectedSymbol();
    const signals = useSignals();
    const wsConnected = useWSConnected();
    const { setSymbol, addSignals } = useAppStore();

    // WebSocket
    const { connect } = useStableWebSocket();

    // Initialize WebSocket connection
    useEffect(() => {
        connect();
    }, [connect]);

    // Generate mock signals for demo
    useEffect(() => {
        const generateMockSignal = () => {
            const types = ['BUY', 'SELL', 'HOLD'] as const;
            const sources = ['AI Prophet', 'Technical Analysis', 'Momentum', 'Volume Analysis'];

            return {
                id: `signal-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                symbol: selectedSymbol,
                type: types[Math.floor(Math.random() * types.length)],
                action: 'Market analysis suggests opportunity',
                confidence: Math.floor(Math.random() * 40) + 60, // 60-100%
                price: 456.78 + (Math.random() - 0.5) * 20,
                timestamp: Date.now(),
                reasoning: `Technical indicators show ${types[Math.floor(Math.random() * types.length)]} signals with strong momentum`,
                source: sources[Math.floor(Math.random() * sources.length)],
                isLive: true,
            };
        };

        // Add initial signals
        const initialSignals = Array.from({ length: 3 }, generateMockSignal);
        addSignals(initialSignals);

        // Generate new signals periodically
        const interval = setInterval(() => {
            if (wsConnected && Math.random() > 0.7) { // 30% chance every 5 seconds
                const newSignal = generateMockSignal();
                addSignals([newSignal]);
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [selectedSymbol, wsConnected, addSignals]);

    // Handle symbol change
    const handleSymbolChange = useCallback((event: any, newValue: any) => {
        if (newValue) {
            setSymbol(newValue.symbol);
        }
    }, [setSymbol]);

    return (
        <DashboardContainer>
            <Container maxWidth="xl">
                {/* Header */}
                <HeaderCard>
                    <CardContent sx={{ py: 2 }}>
                        <Grid container spacing={3} alignItems="center">
                            {/* Title */}
                            <Grid item xs={12} md={4}>
                                <Stack direction="row" alignItems="center" spacing={2}>
                                    <Avatar sx={{
                                        background: goldenEyeGradients.mystical,
                                        width: 48,
                                        height: 48,
                                    }}>
                                        <Psychology />
                                    </Avatar>
                                    <Box>
                                        <Typography variant="h5" sx={{
                                            fontWeight: 700,
                                            color: goldenEyeColors.textPrimary,
                                        }}>
                                            Golden Signals AI
                                        </Typography>
                                        <Typography variant="caption" sx={{
                                            color: goldenEyeColors.textSecondary,
                                        }}>
                                            AI-Powered Market Intelligence
                                        </Typography>
                                    </Box>
                                </Stack>
                            </Grid>

                            {/* Symbol Search */}
                            <Grid item xs={12} md={4}>
                                <Autocomplete
                                    options={POPULAR_SYMBOLS}
                                    getOptionLabel={(option) => `${option.symbol} - ${option.name}`}
                                    value={POPULAR_SYMBOLS.find(s => s.symbol === selectedSymbol) || POPULAR_SYMBOLS[0]}
                                    onChange={handleSymbolChange}
                                    renderInput={(params) => (
                                        <TextField
                                            {...params}
                                            placeholder="Search symbols..."
                                            InputProps={{
                                                ...params.InputProps,
                                                startAdornment: <SearchIcon sx={{ mr: 1, color: goldenEyeColors.textSecondary }} />,
                                            }}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    background: alpha(goldenEyeColors.surface, 0.6),
                                                    '& fieldset': { borderColor: alpha(goldenEyeColors.primary, 0.2) },
                                                    '&:hover fieldset': { borderColor: alpha(goldenEyeColors.primary, 0.4) },
                                                    '&.Mui-focused fieldset': { borderColor: goldenEyeColors.primary },
                                                },
                                                '& .MuiInputBase-input': { color: goldenEyeColors.textPrimary },
                                            }}
                                        />
                                    )}
                                />
                            </Grid>

                            {/* Connection Status */}
                            <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                                <StatusIndicator connected={wsConnected}>
                                    {wsConnected ? <Wifi fontSize="small" /> : <WifiOff fontSize="small" />}
                                    <Typography variant="body2" sx={{
                                        fontWeight: 600,
                                        color: wsConnected ? goldenEyeColors.bullish : goldenEyeColors.bearish,
                                    }}>
                                        {wsConnected ? 'LIVE' : 'DISCONNECTED'}
                                    </Typography>
                                </StatusIndicator>
                            </Grid>
                        </Grid>
                    </CardContent>
                </HeaderCard>

                {/* Main Content */}
                <Grid container spacing={3}>
                    {/* Chart Area */}
                    <Grid item xs={12} lg={8}>
                        <SignalCard>
                            <CardContent sx={{ height: 500 }}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                                    <Typography variant="h6" sx={{
                                        fontWeight: 600,
                                        color: goldenEyeColors.textPrimary,
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 1,
                                    }}>
                                        <Timeline sx={{ color: goldenEyeColors.primary }} />
                                        {selectedSymbol} Chart
                                    </Typography>
                                    <Chip
                                        label="Real-time"
                                        size="small"
                                        sx={{
                                            background: alpha(goldenEyeColors.bullish, 0.2),
                                            color: goldenEyeColors.bullish,
                                            fontWeight: 600,
                                        }}
                                    />
                                </Stack>

                                {/* Simple chart placeholder */}
                                <Box sx={{
                                    height: 400,
                                    background: alpha(goldenEyeColors.surface, 0.3),
                                    borderRadius: 2,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    border: `1px solid ${alpha(goldenEyeColors.primary, 0.1)}`,
                                }}>
                                    <Typography variant="h6" sx={{
                                        color: goldenEyeColors.textSecondary,
                                        textAlign: 'center',
                                    }}>
                                        📈 Chart for {selectedSymbol}
                                        <br />
                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                            Real-time market visualization
                                        </Typography>
                                    </Typography>
                                </Box>
                            </CardContent>
                        </SignalCard>
                    </Grid>

                    {/* Signals Panel */}
                    <Grid item xs={12} lg={4}>
                        <SignalCard>
                            <CardContent>
                                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                                    <Typography variant="h6" sx={{
                                        fontWeight: 600,
                                        color: goldenEyeColors.textPrimary,
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 1,
                                    }}>
                                        <Speed sx={{ color: goldenEyeColors.primary }} />
                                        Live Signals
                                    </Typography>
                                    <Chip
                                        label={`${signals.length} signals`}
                                        size="small"
                                        sx={{
                                            background: alpha(goldenEyeColors.primary, 0.2),
                                            color: goldenEyeColors.primary,
                                            fontWeight: 600,
                                        }}
                                    />
                                </Stack>

                                {/* Signals List */}
                                <Stack spacing={2} sx={{ maxHeight: 400, overflowY: 'auto' }}>
                                    <AnimatePresence>
                                        {signals.map((signal) => (
                                            <motion.div
                                                key={signal.id}
                                                initial={{ opacity: 0, x: 20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                exit={{ opacity: 0, x: -20 }}
                                                transition={{ duration: 0.3 }}
                                            >
                                                <Paper sx={{
                                                    p: 2,
                                                    background: alpha(goldenEyeColors.surface, 0.6),
                                                    border: `1px solid ${alpha(getSignalColor(signal.type), 0.3)}`,
                                                    borderRadius: 2,
                                                    transition: 'all 0.2s ease',
                                                    '&:hover': {
                                                        borderColor: getSignalColor(signal.type),
                                                        transform: 'translateX(4px)',
                                                    },
                                                }}>
                                                    <Stack spacing={1}>
                                                        {/* Header */}
                                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                                            <Typography variant="subtitle2" sx={{
                                                                fontWeight: 600,
                                                                color: goldenEyeColors.textPrimary,
                                                            }}>
                                                                {signal.symbol}
                                                            </Typography>
                                                            <Chip
                                                                label={signal.type}
                                                                size="small"
                                                                sx={{
                                                                    background: alpha(getSignalColor(signal.type), 0.2),
                                                                    color: getSignalColor(signal.type),
                                                                    fontWeight: 600,
                                                                    fontSize: '10px',
                                                                }}
                                                                icon={
                                                                    signal.type === 'BUY' ? <TrendingUp /> :
                                                                        signal.type === 'SELL' ? <TrendingDown /> : <Remove />
                                                                }
                                                            />
                                                        </Stack>

                                                        {/* Confidence */}
                                                        <Box>
                                                            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.5 }}>
                                                                <Typography variant="caption" sx={{ color: goldenEyeColors.textSecondary }}>
                                                                    Confidence
                                                                </Typography>
                                                                <Typography variant="caption" sx={{
                                                                    fontWeight: 600,
                                                                    color: getConfidenceColor(signal.confidence),
                                                                }}>
                                                                    {signal.confidence}%
                                                                </Typography>
                                                            </Stack>
                                                            <LinearProgress
                                                                variant="determinate"
                                                                value={signal.confidence}
                                                                sx={{
                                                                    height: 4,
                                                                    borderRadius: 2,
                                                                    backgroundColor: alpha(goldenEyeColors.surface, 0.3),
                                                                    '& .MuiLinearProgress-bar': {
                                                                        backgroundColor: getConfidenceColor(signal.confidence),
                                                                    },
                                                                }}
                                                            />
                                                        </Box>

                                                        {/* Source & Time */}
                                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                                            <Typography variant="caption" sx={{
                                                                color: goldenEyeColors.primary,
                                                                fontWeight: 500,
                                                            }}>
                                                                {signal.source}
                                                            </Typography>
                                                            <Typography variant="caption" sx={{
                                                                color: goldenEyeColors.textSecondary,
                                                            }}>
                                                                {new Date(signal.timestamp).toLocaleTimeString()}
                                                            </Typography>
                                                        </Stack>
                                                    </Stack>
                                                </Paper>
                                            </motion.div>
                                        ))}
                                    </AnimatePresence>

                                    {signals.length === 0 && (
                                        <Box sx={{
                                            textAlign: 'center',
                                            py: 4,
                                            color: goldenEyeColors.textSecondary,
                                        }}>
                                            <Psychology sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                                            <Typography variant="body2">
                                                Waiting for signals...
                                            </Typography>
                                        </Box>
                                    )}
                                </Stack>
                            </CardContent>
                        </SignalCard>
                    </Grid>
                </Grid>

                {/* Golden Eye Components */}
                <GoldenEyeWidget />
                <GoldenEyeChat />
            </Container>
        </DashboardContainer>
    );
};

export default GoldenSignalsDashboard; 