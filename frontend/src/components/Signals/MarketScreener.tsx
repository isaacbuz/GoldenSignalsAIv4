import React from 'react';
import {
    Box,
    Typography,
    Stack,
    Chip,
    useTheme,
    alpha,
    LinearProgress,
    Tooltip,
    IconButton,
    Paper,
    Skeleton,
    Card,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    AutoAwesome,
    Refresh,
    Speed,
    Psychology,
    ShowChart,
    Timer,
    Star,
    StarBorder,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { apiClient, MarketOpportunity } from '../../services/api';

interface MarketScreenerProps {
    onSymbolSelect?: (symbol: string) => void;
}

const MarketScreener: React.FC<MarketScreenerProps> = ({ onSymbolSelect }) => {
    const theme = useTheme();

    // Fetch market opportunities from API
    const { data: opportunities = [], isLoading, refetch } = useQuery({
        queryKey: ['marketOpportunities'],
        queryFn: () => apiClient.getMarketOpportunities(),
        // Disabled auto-refresh to prevent constant updating
        staleTime: 300000, // Keep data fresh for 5 minutes
    });

    const getMomentumColor = (momentum: string) => {
        switch (momentum) {
            case 'strong': return theme.palette.success.main;
            case 'moderate': return theme.palette.warning.main;
            case 'building': return theme.palette.info.main;
            default: return theme.palette.text.secondary;
        }
    };

    const getScoreColor = (score: number) => {
        if (score >= 90) return theme.palette.success.main;
        if (score >= 80) return theme.palette.warning.main;
        if (score >= 70) return theme.palette.info.main;
        return theme.palette.text.secondary;
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0 }}>
                <TrendingUp sx={{ color: theme.palette.success.main }} />
                Market Screener
                <Chip
                    label={`${opportunities.length} opportunities`}
                    size="small"
                    color="success"
                    sx={{ ml: 'auto' }}
                />
            </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexShrink: 0 }}>
                AI-analyzed opportunities from 500+ stocks
            </Typography>

            <Box sx={{
                flex: 1,
                overflow: 'auto',
                minHeight: 0,
                '&::-webkit-scrollbar': {
                    width: '8px',
                },
                '&::-webkit-scrollbar-track': {
                    background: alpha(theme.palette.background.paper, 0.1),
                    borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb': {
                    background: alpha(theme.palette.primary.main, 0.3),
                    borderRadius: '4px',
                    '&:hover': {
                        background: alpha(theme.palette.primary.main, 0.5),
                    },
                },
            }}>
                {isLoading ? (
                    <Box sx={{ p: 1 }}>
                        {[1, 2, 3].map(i => (
                            <Skeleton key={i} variant="rectangular" height={80} sx={{ mb: 1, borderRadius: 1 }} />
                        ))}
                    </Box>
                ) : opportunities.length === 0 ? (
                    // Empty state
                    <Box textAlign="center" py={4}>
                        <Typography variant="body2" color="text.secondary">
                            No opportunities found
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Check back in a few minutes
                        </Typography>
                    </Box>
                ) : (
                    <Stack spacing={1.5}>
                        {opportunities.map((opp) => (
                            <Card
                                key={opp.id}
                                elevation={0}
                                sx={{
                                    p: 2,
                                    cursor: 'pointer',
                                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                    transition: 'all 0.2s',
                                    '&:hover': {
                                        borderColor: theme.palette.primary.main,
                                        transform: 'translateX(4px)',
                                        boxShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.2)}`,
                                    },
                                }}
                                onClick={() => onSymbolSelect?.(opp.symbol)}
                            >
                                {/* Row 1: Symbol & Type */}
                                <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                                    <Stack direction="row" alignItems="center" spacing={1}>
                                        <Typography variant="subtitle1" fontWeight="bold">
                                            {opp.symbol}
                                        </Typography>
                                        <Chip
                                            icon={opp.type === 'CALL' ? <TrendingUp /> : <TrendingDown />}
                                            label={opp.type}
                                            size="small"
                                            color={opp.type === 'CALL' ? 'success' : 'error'}
                                            sx={{ height: 24 }}
                                        />
                                    </Stack>
                                    <Stack direction="row" alignItems="center" spacing={0.5}>
                                        <Psychology sx={{ fontSize: 16, color: getScoreColor(opp.aiScore) }} />
                                        <Typography variant="body2" fontWeight="bold" color={getScoreColor(opp.aiScore)}>
                                            {opp.aiScore}
                                        </Typography>
                                    </Stack>
                                </Stack>

                                {/* Row 2: Company Name & Return */}
                                <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                                    <Typography variant="caption" color="text.secondary" sx={{ maxWidth: '60%', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {opp.name}
                                    </Typography>
                                    <Typography variant="body2" fontWeight="medium" color="success.main">
                                        +{opp.potentialReturn}%
                                    </Typography>
                                </Stack>

                                {/* Row 3: Key Reason */}
                                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                                    {opp.keyReason}
                                </Typography>

                                {/* Row 4: Metrics */}
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    {/* Confidence */}
                                    <Chip
                                        icon={<Speed />}
                                        label={`${opp.confidence}%`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ height: 20, fontSize: '0.7rem' }}
                                    />

                                    {/* Timeframe */}
                                    <Chip
                                        icon={<Timer />}
                                        label={opp.timeframe}
                                        size="small"
                                        variant="outlined"
                                        sx={{ height: 20, fontSize: '0.7rem' }}
                                    />

                                    {/* Momentum */}
                                    <Chip
                                        label={opp.momentum}
                                        size="small"
                                        sx={{
                                            height: 20,
                                            fontSize: '0.7rem',
                                            backgroundColor: alpha(getMomentumColor(opp.momentum), 0.1),
                                            color: getMomentumColor(opp.momentum),
                                            border: `1px solid ${alpha(getMomentumColor(opp.momentum), 0.3)}`,
                                        }}
                                    />
                                </Stack>

                                {/* AI Score Progress Bar */}
                                <Box sx={{ mt: 1.5 }}>
                                    <LinearProgress
                                        variant="determinate"
                                        value={opp.aiScore}
                                        sx={{
                                            height: 4,
                                            borderRadius: 2,
                                            backgroundColor: alpha(theme.palette.divider, 0.1),
                                            '& .MuiLinearProgress-bar': {
                                                backgroundColor: getScoreColor(opp.aiScore),
                                                borderRadius: 2,
                                            },
                                        }}
                                    />
                                </Box>
                            </Card>
                        ))}
                    </Stack>
                )}
            </Box>

            {/* Footer Stats */}
            <Box
                sx={{
                    mt: 2,
                    pt: 2,
                    borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexShrink: 0,
                }}
            >
                <Typography variant="caption" color="text.secondary">
                    Last scan: {new Date().toLocaleTimeString()}
                </Typography>
                <Tooltip title="Refresh screener">
                    <IconButton size="small" onClick={() => refetch()}>
                        <Refresh fontSize="small" />
                    </IconButton>
                </Tooltip>
            </Box>
        </Box>
    );
};

export default MarketScreener;
