/**
 * MiniHeatMap Component - Compact Market Overview
 * 
 * A smaller version of the heat map for dashboard integration
 */

import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    useTheme,
    alpha,
    Stack,
    Chip,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    OpenInFull,
    Assessment,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import MarketHeatmapModal from './MarketHeatmapModal';

interface MiniHeatMapProps {
    onSymbolSelect?: (symbol: string) => void;
}

interface SectorSummary {
    name: string;
    change: number;
    topStock: { symbol: string; change: number };
}

const MiniHeatMap: React.FC<MiniHeatMapProps> = ({ onSymbolSelect }) => {
    const theme = useTheme();
    const [heatmapModalOpen, setHeatmapModalOpen] = useState(false);

    // Fetch market summary
    const { data: sectors = [] } = useQuery({
        queryKey: ['market-summary'],
        queryFn: async () => {
            // Mock data - replace with real API
            return [
                { name: 'Tech', change: 2.1, topStock: { symbol: 'NVDA', change: 5.2 } },
                { name: 'Health', change: 1.8, topStock: { symbol: 'LLY', change: 3.8 } },
                { name: 'Finance', change: 1.3, topStock: { symbol: 'JPM', change: 1.5 } },
                { name: 'Consumer', change: 1.2, topStock: { symbol: 'AMZN', change: 2.8 } },
                { name: 'Energy', change: -1.4, topStock: { symbol: 'XOM', change: -1.2 } },
                { name: 'Industrial', change: 2.4, topStock: { symbol: 'BA', change: 3.2 } },
            ] as SectorSummary[];
        },
        // Disabled auto-refresh to prevent constant updating
        staleTime: 300000, // Keep data fresh for 5 minutes
    });

    const getColor = (change: number) => {
        const intensity = Math.min(Math.abs(change) / 3, 1);
        if (change > 0) {
            return alpha(theme.palette.success.main, 0.2 + intensity * 0.4);
        } else if (change < 0) {
            return alpha(theme.palette.error.main, 0.2 + intensity * 0.4);
        }
        return alpha(theme.palette.grey[500], 0.2);
    };

    const handleSymbolSelect = (symbol: string) => {
        setHeatmapModalOpen(false);
        onSymbolSelect?.(symbol);
    };

    return (
        <>
            <Paper
                elevation={0}
                sx={{
                    p: 1.5,
                    background: alpha(theme.palette.background.paper, 0.5),
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    borderRadius: 2,
                    width: '100%',
                    boxSizing: 'border-box',
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1.5}>
                    <Typography variant="subtitle2" fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, fontSize: '0.875rem' }}>
                        <Assessment sx={{ fontSize: 16, color: theme.palette.primary.main }} />
                        Market Overview
                    </Typography>
                    <Tooltip title="Expand to full heatmap">
                        <IconButton
                            size="small"
                            onClick={() => setHeatmapModalOpen(true)}
                            sx={{ padding: 0.5 }}
                        >
                            <OpenInFull sx={{ fontSize: 16 }} />
                        </IconButton>
                    </Tooltip>
                </Stack>

                <Box
                    sx={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(2, 1fr)',
                        gap: 0.75,
                        width: '100%',
                    }}
                >
                    {sectors.map((sector, index) => (
                        <motion.div
                            key={sector.name}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.05 }}
                            style={{ width: '100%' }}
                        >
                            <Paper
                                elevation={0}
                                sx={{
                                    p: 1,
                                    background: getColor(sector.change),
                                    border: `1px solid ${alpha(
                                        sector.change > 0 ? theme.palette.success.main :
                                            sector.change < 0 ? theme.palette.error.main :
                                                theme.palette.grey[500],
                                        0.2
                                    )}`,
                                    cursor: 'pointer',
                                    transition: 'all 0.2s',
                                    '&:hover': {
                                        transform: 'scale(1.02)',
                                        borderColor: theme.palette.primary.main,
                                    },
                                    minHeight: '70px',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    justifyContent: 'space-between',
                                }}
                                onClick={() => setHeatmapModalOpen(true)}
                            >
                                <Typography variant="caption" fontWeight="bold" sx={{ fontSize: '0.7rem' }}>
                                    {sector.name}
                                </Typography>
                                <Typography
                                    variant="body2"
                                    sx={{
                                        fontWeight: 'bold',
                                        fontSize: '0.85rem',
                                        color: sector.change > 0 ? theme.palette.success.main :
                                            sector.change < 0 ? theme.palette.error.main :
                                                theme.palette.text.primary,
                                        display: 'flex',
                                        alignItems: 'center',
                                        mt: 0.25,
                                    }}
                                >
                                    {sector.change > 0 ? <TrendingUp sx={{ fontSize: 12, mr: 0.25 }} /> :
                                        sector.change < 0 ? <TrendingDown sx={{ fontSize: 12, mr: 0.25 }} /> : null}
                                    {sector.change > 0 ? '+' : ''}{sector.change.toFixed(1)}%
                                </Typography>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{
                                        fontSize: '0.6rem',
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                    }}
                                >
                                    {sector.topStock.symbol} {sector.topStock.change > 0 ? '+' : ''}{sector.topStock.change.toFixed(1)}%
                                </Typography>
                            </Paper>
                        </motion.div>
                    ))}
                </Box>

                <Stack direction="row" spacing={0.5} sx={{ mt: 1.5 }}>
                    <Chip
                        size="small"
                        label={`↑ ${sectors.filter(s => s.change > 0).length}`}
                        color="success"
                        variant="outlined"
                        sx={{
                            fontSize: '0.65rem',
                            height: 20,
                            '& .MuiChip-label': { px: 1 }
                        }}
                    />
                    <Chip
                        size="small"
                        label={`↓ ${sectors.filter(s => s.change < 0).length}`}
                        color="error"
                        variant="outlined"
                        sx={{
                            fontSize: '0.65rem',
                            height: 20,
                            '& .MuiChip-label': { px: 1 }
                        }}
                    />
                    <Chip
                        size="small"
                        label={`Avg: ${(sectors.reduce((sum, s) => sum + s.change, 0) / sectors.length).toFixed(1)}%`}
                        variant="outlined"
                        sx={{
                            fontSize: '0.65rem',
                            height: 20,
                            '& .MuiChip-label': { px: 1 }
                        }}
                    />
                </Stack>
            </Paper>

            {/* Market Heatmap Modal */}
            <MarketHeatmapModal
                open={heatmapModalOpen}
                onClose={() => setHeatmapModalOpen(false)}
                onSymbolSelect={handleSymbolSelect}
            />
        </>
    );
};

export default MiniHeatMap; 