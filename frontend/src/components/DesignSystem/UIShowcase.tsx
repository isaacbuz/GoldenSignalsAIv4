/**
 * UI Design System Showcase
 *
 * This component demonstrates all the key visual elements
 * of the GoldenSignalsAI professional redesign
 */

import React from 'react';
import {
    Box,
    Grid,
    Paper,
    Typography,
    Stack,
    Chip,
    Button,
    Card,
    CardContent,
    IconButton,
    LinearProgress,
    useTheme,
    alpha,
    Divider,
    Alert,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    Psychology,
    AutoGraph,
    SignalCellularAlt,
    SmartToy,
    Speed,
    Timeline,
    Assessment,
    School,
} from '@mui/icons-material';

// Professional color palette
const colors = {
    background: '#0A0E1A',
    surface: '#131A2A',
    elevated: '#1E293B',
    bullish: '#00D4AA',
    bearish: '#FF4757',
    ai: '#FFD700',
    textPrimary: '#E2E8F0',
    textSecondary: '#94A3B8',
};

export const UIShowcase: React.FC = () => {
    const theme = useTheme();

    return (
        <Box sx={{ p: 4, bgcolor: colors.background, minHeight: '100vh' }}>
            <Stack spacing={4}>
                {/* Header */}
                <Box>
                    <Typography variant="h3" sx={{ color: colors.textPrimary, fontWeight: 800, mb: 1 }}>
                        GoldenSignalsAI Design System
                    </Typography>
                    <Typography variant="body1" sx={{ color: colors.textSecondary }}>
                        Professional UI components for AI-powered signal generation
                    </Typography>
                </Box>

                <Divider sx={{ borderColor: alpha(colors.textPrimary, 0.1) }} />

                {/* Color Palette */}
                <Box>
                    <Typography variant="h5" sx={{ color: colors.textPrimary, fontWeight: 700, mb: 3 }}>
                        Color Palette
                    </Typography>
                    <Grid container spacing={2}>
                        {Object.entries(colors).map(([name, color]) => (
                            <Grid item xs={6} sm={4} md={3} key={name}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        bgcolor: color,
                                        border: `1px solid ${alpha(colors.textPrimary, 0.1)}`,
                                    }}
                                >
                                    <Typography variant="caption" sx={{ color: name.includes('text') ? colors.background : colors.textPrimary }}>
                                        {name}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: name.includes('text') ? colors.background : colors.textPrimary, fontFamily: 'monospace' }}>
                                        {color}
                                    </Typography>
                                </Paper>
                            </Grid>
                        ))}
                    </Grid>
                </Box>

                {/* Typography */}
                <Box>
                    <Typography variant="h5" sx={{ color: colors.textPrimary, fontWeight: 700, mb: 3 }}>
                        Typography
                    </Typography>
                    <Stack spacing={2}>
                        <Typography variant="h1" sx={{ color: colors.textPrimary, fontWeight: 800 }}>
                            Heading 1 - Inter 800
                        </Typography>
                        <Typography variant="h3" sx={{ color: colors.textPrimary, fontWeight: 700 }}>
                            Heading 3 - Inter 700
                        </Typography>
                        <Typography variant="body1" sx={{ color: colors.textPrimary }}>
                            Body text - Inter 400 - Used for general content and descriptions
                        </Typography>
                        <Typography variant="body2" sx={{ color: colors.textSecondary }}>
                            Secondary text - Inter 400 - Used for supporting information
                        </Typography>
                        <Typography sx={{ fontFamily: 'JetBrains Mono', color: colors.bullish, fontSize: '1.5rem' }}>
                            $458.23 - JetBrains Mono for numbers
                        </Typography>
                        <Typography sx={{ fontFamily: 'Bebas Neue', color: colors.ai, fontSize: '3rem' }}>
                            92% - Bebas Neue for confidence
                        </Typography>
                    </Stack>
                </Box>

                {/* Buttons */}
                <Box>
                    <Typography variant="h5" sx={{ color: colors.textPrimary, fontWeight: 700, mb: 3 }}>
                        Buttons
                    </Typography>
                    <Stack direction="row" spacing={2} flexWrap="wrap">
                        <Button variant="contained" sx={{ bgcolor: colors.bullish, color: colors.background }}>
                            Buy Signal
                        </Button>
                        <Button variant="contained" sx={{ bgcolor: colors.bearish }}>
                            Sell Signal
                        </Button>
                        <Button variant="contained" sx={{ bgcolor: colors.ai, color: colors.background }}>
                            AI Analysis
                        </Button>
                        <Button variant="outlined" sx={{ borderColor: colors.bullish, color: colors.bullish }}>
                            View Details
                        </Button>
                        <IconButton sx={{ color: colors.ai }}>
                            <SmartToy />
                        </IconButton>
                    </Stack>
                </Box>

                {/* Signal Cards */}
                <Box>
                    <Typography variant="h5" sx={{ color: colors.textPrimary, fontWeight: 700, mb: 3 }}>
                        Signal Cards
                    </Typography>
                    <Grid container spacing={3}>
                        {/* Bullish Signal Card */}
                        <Grid item xs={12} md={4}>
                            <Card
                                sx={{
                                    bgcolor: alpha(colors.surface, 0.8),
                                    backdropFilter: 'blur(10px)',
                                    border: `1px solid ${alpha(colors.bullish, 0.3)}`,
                                    borderRadius: 2,
                                }}
                            >
                                <CardContent>
                                    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                                        <Stack direction="row" spacing={1} alignItems="center">
                                            <Typography variant="h6" sx={{ color: colors.textPrimary, fontWeight: 700 }}>
                                                AAPL
                                            </Typography>
                                            <Chip
                                                label="BUY CALL"
                                                size="small"
                                                sx={{
                                                    bgcolor: colors.bullish,
                                                    color: colors.background,
                                                    fontWeight: 600,
                                                }}
                                            />
                                        </Stack>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography sx={{ fontFamily: 'Bebas Neue', color: colors.ai, fontSize: '2rem' }}>
                                                92%
                                            </Typography>
                                            <Typography variant="caption" sx={{ color: colors.textSecondary }}>
                                                AI Confidence
                                            </Typography>
                                        </Box>
                                    </Stack>

                                    <Box sx={{ mb: 2 }}>
                                        <LinearProgress
                                            variant="determinate"
                                            value={92}
                                            sx={{
                                                height: 6,
                                                borderRadius: 3,
                                                bgcolor: alpha(colors.textSecondary, 0.2),
                                                '& .MuiLinearProgress-bar': {
                                                    bgcolor: colors.ai,
                                                },
                                            }}
                                        />
                                    </Box>

                                    <Grid container spacing={2}>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" sx={{ color: colors.textSecondary }}>
                                                Entry
                                            </Typography>
                                            <Typography sx={{ color: colors.textPrimary, fontFamily: 'monospace' }}>
                                                $234.56
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" sx={{ color: colors.textSecondary }}>
                                                Target
                                            </Typography>
                                            <Typography sx={{ color: colors.bullish, fontFamily: 'monospace' }}>
                                                $245.00
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" sx={{ color: colors.textSecondary }}>
                                                R:R
                                            </Typography>
                                            <Typography sx={{ color: colors.textPrimary, fontWeight: 600 }}>
                                                1:3.2
                                            </Typography>
                                        </Grid>
                                    </Grid>

                                    <Stack direction="row" spacing={1} mt={2}>
                                        <Chip
                                            icon={<AutoGraph sx={{ fontSize: 14 }} />}
                                            label="MOMENTUM"
                                            size="small"
                                            sx={{
                                                bgcolor: alpha(colors.ai, 0.2),
                                                color: colors.ai,
                                                fontSize: '0.625rem',
                                            }}
                                        />
                                        <Chip
                                            icon={<Timeline sx={{ fontSize: 14 }} />}
                                            label="SWING"
                                            size="small"
                                            sx={{
                                                bgcolor: alpha(colors.textPrimary, 0.1),
                                                color: colors.textSecondary,
                                                fontSize: '0.625rem',
                                            }}
                                        />
                                    </Stack>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Bearish Signal Card */}
                        <Grid item xs={12} md={4}>
                            <Card
                                sx={{
                                    bgcolor: alpha(colors.surface, 0.8),
                                    backdropFilter: 'blur(10px)',
                                    border: `1px solid ${alpha(colors.bearish, 0.3)}`,
                                    borderRadius: 2,
                                }}
                            >
                                <CardContent>
                                    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                                        <Stack direction="row" spacing={1} alignItems="center">
                                            <Typography variant="h6" sx={{ color: colors.textPrimary, fontWeight: 700 }}>
                                                TSLA
                                            </Typography>
                                            <Chip
                                                label="SELL PUT"
                                                size="small"
                                                sx={{
                                                    bgcolor: colors.bearish,
                                                    color: 'white',
                                                    fontWeight: 600,
                                                }}
                                            />
                                        </Stack>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography sx={{ fontFamily: 'Bebas Neue', color: colors.ai, fontSize: '2rem' }}>
                                                87%
                                            </Typography>
                                            <Typography variant="caption" sx={{ color: colors.textSecondary }}>
                                                AI Confidence
                                            </Typography>
                                        </Box>
                                    </Stack>

                                    <Alert
                                        severity="info"
                                        sx={{
                                            bgcolor: alpha(colors.ai, 0.1),
                                            color: colors.textPrimary,
                                            border: `1px solid ${alpha(colors.ai, 0.3)}`,
                                            '& .MuiAlert-icon': {
                                                color: colors.ai,
                                            },
                                        }}
                                    >
                                        <Typography variant="caption">
                                            Pattern: Head & Shoulders detected
                                        </Typography>
                                    </Alert>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* AI Analysis Card */}
                        <Grid item xs={12} md={4}>
                            <Card
                                sx={{
                                    bgcolor: alpha(colors.surface, 0.8),
                                    backdropFilter: 'blur(10px)',
                                    border: `1px solid ${alpha(colors.ai, 0.3)}`,
                                    borderRadius: 2,
                                    background: `linear-gradient(135deg, ${alpha(colors.ai, 0.1)} 0%, ${alpha(colors.surface, 0.8)} 100%)`,
                                }}
                            >
                                <CardContent>
                                    <Stack direction="row" spacing={1} alignItems="center" mb={2}>
                                        <Psychology sx={{ color: colors.ai }} />
                                        <Typography variant="h6" sx={{ color: colors.textPrimary, fontWeight: 700 }}>
                                            AI Market Analysis
                                        </Typography>
                                    </Stack>

                                    <Stack spacing={2}>
                                        <Box>
                                            <Stack direction="row" justifyContent="space-between">
                                                <Typography variant="body2" sx={{ color: colors.textSecondary }}>
                                                    Market Sentiment
                                                </Typography>
                                                <Chip
                                                    label="BULLISH"
                                                    size="small"
                                                    sx={{
                                                        bgcolor: alpha(colors.bullish, 0.2),
                                                        color: colors.bullish,
                                                        fontWeight: 600,
                                                        fontSize: '0.625rem',
                                                    }}
                                                />
                                            </Stack>
                                        </Box>

                                        <Box>
                                            <Stack direction="row" justifyContent="space-between">
                                                <Typography variant="body2" sx={{ color: colors.textSecondary }}>
                                                    Signal Accuracy Today
                                                </Typography>
                                                <Typography variant="body2" sx={{ color: colors.bullish, fontWeight: 600 }}>
                                                    87.3%
                                                </Typography>
                                            </Stack>
                                        </Box>

                                        <Box>
                                            <Stack direction="row" justifyContent="space-between">
                                                <Typography variant="body2" sx={{ color: colors.textSecondary }}>
                                                    Active Signals
                                                </Typography>
                                                <Typography variant="body2" sx={{ color: colors.textPrimary, fontWeight: 600 }}>
                                                    24
                                                </Typography>
                                            </Stack>
                                        </Box>
                                    </Stack>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </Box>

                {/* Status Indicators */}
                <Box>
                    <Typography variant="h5" sx={{ color: colors.textPrimary, fontWeight: 700, mb: 3 }}>
                        Status Indicators
                    </Typography>
                    <Stack direction="row" spacing={2} flexWrap="wrap">
                        <Chip
                            label="MARKET OPEN"
                            size="small"
                            icon={<Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: colors.bullish }} />}
                            sx={{
                                bgcolor: alpha(colors.bullish, 0.2),
                                color: colors.bullish,
                                fontWeight: 600,
                            }}
                        />
                        <Chip
                            label="AI ACTIVE"
                            size="small"
                            icon={<Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: colors.ai, animation: 'pulse 2s infinite' }} />}
                            sx={{
                                bgcolor: alpha(colors.ai, 0.2),
                                color: colors.ai,
                                fontWeight: 600,
                            }}
                        />
                        <Chip
                            label="HIGH VOLATILITY"
                            size="small"
                            icon={<Speed sx={{ fontSize: 14 }} />}
                            sx={{
                                bgcolor: alpha(colors.bearish, 0.2),
                                color: colors.bearish,
                                fontWeight: 600,
                            }}
                        />
                    </Stack>
                </Box>

                {/* Educational Notice */}
                <Alert
                    severity="info"
                    sx={{
                        bgcolor: alpha(colors.ai, 0.1),
                        color: colors.textPrimary,
                        border: `1px solid ${alpha(colors.ai, 0.3)}`,
                        '& .MuiAlert-icon': {
                            color: colors.ai,
                        },
                    }}
                >
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        📚 Educational Platform
                    </Typography>
                    <Typography variant="caption">
                        GoldenSignalsAI generates AI-powered trading signals for educational purposes.
                        We do not execute trades or manage real portfolios. Always do your own research.
                    </Typography>
                </Alert>
            </Stack>

            {/* CSS for animations */}
            <style>
                {`
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                `}
            </style>

            <style>
                {`
                    .demo-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 2rem;
                        margin: 2rem 0;
                    }
                `}
            </style>
        </Box>
    );
};
