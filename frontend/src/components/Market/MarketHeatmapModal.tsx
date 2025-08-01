import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogContent,
    Box,
    Typography,
    IconButton,
    useTheme,
    alpha,
    Tooltip,
    Stack,
    Chip,
    ToggleButton,
    ToggleButtonGroup,
    Fade,
    Zoom,
    Divider,
} from '@mui/material';
import {
    Close,
    TrendingUp,
    TrendingDown,
    Refresh,
    GridView,
    ViewModule,
    ViewComfy,
    FilterList,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';

interface StockData {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    marketCap: number;
    sector: string;
}

interface MarketHeatmapModalProps {
    open: boolean;
    onClose: () => void;
    onSymbolSelect?: (symbol: string) => void;
}

const MarketHeatmapModal: React.FC<MarketHeatmapModalProps> = ({
    open,
    onClose,
    onSymbolSelect,
}) => {
    const theme = useTheme();
    const [viewMode, setViewMode] = useState<'market-cap' | 'equal' | 'volume'>('market-cap');
    const [sector, setSector] = useState<string>('all');
    const [hoveredStock, setHoveredStock] = useState<StockData | null>(null);

    // Mock data - in production, this would come from your API
    const { data: stocks = [], isLoading, refetch } = useQuery({
        queryKey: ['heatmap-stocks', sector],
        queryFn: async () => {
            // Mock data generator
            const sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy', 'Industrial'];
            const mockStocks: StockData[] = [
                // Tech giants
                { symbol: 'AAPL', name: 'Apple Inc.', price: 195.89, change: 2.34, changePercent: 1.21, volume: 54234567, marketCap: 3000000000000, sector: 'Technology' },
                { symbol: 'MSFT', name: 'Microsoft', price: 378.91, change: -1.23, changePercent: -0.32, volume: 23456789, marketCap: 2800000000000, sector: 'Technology' },
                { symbol: 'GOOGL', name: 'Alphabet', price: 141.80, change: 3.45, changePercent: 2.49, volume: 34567890, marketCap: 1800000000000, sector: 'Technology' },
                { symbol: 'AMZN', name: 'Amazon', price: 155.33, change: -2.11, changePercent: -1.34, volume: 45678901, marketCap: 1600000000000, sector: 'Technology' },
                { symbol: 'NVDA', name: 'NVIDIA', price: 495.22, change: 15.67, changePercent: 3.27, volume: 87654321, marketCap: 1200000000000, sector: 'Technology' },
                { symbol: 'META', name: 'Meta', price: 326.49, change: 4.56, changePercent: 1.42, volume: 23456789, marketCap: 850000000000, sector: 'Technology' },
                { symbol: 'TSLA', name: 'Tesla', price: 238.45, change: -5.67, changePercent: -2.32, volume: 98765432, marketCap: 750000000000, sector: 'Consumer' },

                // Finance
                { symbol: 'JPM', name: 'JPMorgan', price: 171.03, change: 0.89, changePercent: 0.52, volume: 12345678, marketCap: 500000000000, sector: 'Finance' },
                { symbol: 'BAC', name: 'Bank of America', price: 35.67, change: -0.23, changePercent: -0.64, volume: 34567890, marketCap: 280000000000, sector: 'Finance' },
                { symbol: 'WFC', name: 'Wells Fargo', price: 45.89, change: 0.34, changePercent: 0.75, volume: 23456789, marketCap: 170000000000, sector: 'Finance' },
                { symbol: 'GS', name: 'Goldman Sachs', price: 391.25, change: 2.45, changePercent: 0.63, volume: 3456789, marketCap: 130000000000, sector: 'Finance' },

                // Healthcare
                { symbol: 'JNJ', name: 'Johnson & Johnson', price: 159.74, change: -0.56, changePercent: -0.35, volume: 8765432, marketCap: 420000000000, sector: 'Healthcare' },
                { symbol: 'UNH', name: 'UnitedHealth', price: 523.45, change: 3.21, changePercent: 0.62, volume: 4567890, marketCap: 480000000000, sector: 'Healthcare' },
                { symbol: 'PFE', name: 'Pfizer', price: 28.91, change: -0.12, changePercent: -0.41, volume: 34567890, marketCap: 160000000000, sector: 'Healthcare' },

                // Energy
                { symbol: 'XOM', name: 'Exxon Mobil', price: 104.37, change: 1.23, changePercent: 1.19, volume: 23456789, marketCap: 420000000000, sector: 'Energy' },
                { symbol: 'CVX', name: 'Chevron', price: 147.59, change: 0.89, changePercent: 0.61, volume: 12345678, marketCap: 280000000000, sector: 'Energy' },

                // Consumer
                { symbol: 'WMT', name: 'Walmart', price: 163.42, change: 0.45, changePercent: 0.28, volume: 9876543, marketCap: 440000000000, sector: 'Consumer' },
                { symbol: 'HD', name: 'Home Depot', price: 346.87, change: -1.23, changePercent: -0.35, volume: 5678901, marketCap: 350000000000, sector: 'Consumer' },
                { symbol: 'MCD', name: "McDonald's", price: 281.40, change: 0.67, changePercent: 0.24, volume: 4567890, marketCap: 200000000000, sector: 'Consumer' },
                { symbol: 'NKE', name: 'Nike', price: 104.88, change: -2.34, changePercent: -2.18, volume: 8765432, marketCap: 160000000000, sector: 'Consumer' },

                // More tech
                { symbol: 'AMD', name: 'AMD', price: 168.92, change: 5.43, changePercent: 3.32, volume: 65432109, marketCap: 270000000000, sector: 'Technology' },
                { symbol: 'INTC', name: 'Intel', price: 43.65, change: -0.87, changePercent: -1.95, volume: 34567890, marketCap: 180000000000, sector: 'Technology' },
                { symbol: 'CRM', name: 'Salesforce', price: 265.79, change: 1.23, changePercent: 0.47, volume: 6789012, marketCap: 260000000000, sector: 'Technology' },
                { symbol: 'ORCL', name: 'Oracle', price: 118.49, change: 0.56, changePercent: 0.48, volume: 8901234, marketCap: 320000000000, sector: 'Technology' },
                { symbol: 'ADBE', name: 'Adobe', price: 589.27, change: -3.45, changePercent: -0.58, volume: 3456789, marketCap: 270000000000, sector: 'Technology' },
                { symbol: 'NFLX', name: 'Netflix', price: 445.73, change: 8.91, changePercent: 2.04, volume: 5678901, marketCap: 200000000000, sector: 'Technology' },
            ];

            if (sector === 'all') {
                return mockStocks;
            }
            return mockStocks.filter(s => s.sector === sector);
        },
        // Disabled auto-refresh to prevent constant updating
        staleTime: 300000, // Keep data fresh for 5 minutes
    });

    const getColorIntensity = (changePercent: number) => {
        const absChange = Math.abs(changePercent);
        const maxChange = 5; // Consider 5% as maximum for color scaling
        const intensity = Math.min(absChange / maxChange, 1);

        if (changePercent > 0) {
            // Green colors
            if (intensity > 0.8) return theme.palette.success.dark;
            if (intensity > 0.6) return theme.palette.success.main;
            if (intensity > 0.4) return alpha(theme.palette.success.main, 0.8);
            if (intensity > 0.2) return alpha(theme.palette.success.main, 0.6);
            return alpha(theme.palette.success.main, 0.4);
        } else {
            // Red colors
            if (intensity > 0.8) return theme.palette.error.dark;
            if (intensity > 0.6) return theme.palette.error.main;
            if (intensity > 0.4) return alpha(theme.palette.error.main, 0.8);
            if (intensity > 0.2) return alpha(theme.palette.error.main, 0.6);
            return alpha(theme.palette.error.main, 0.4);
        }
    };

    const getCellSize = (stock: StockData) => {
        if (viewMode === 'equal') return { width: 150, height: 100 };

        const totalMarketCap = stocks.reduce((sum, s) => sum + s.marketCap, 0);
        const totalVolume = stocks.reduce((sum, s) => sum + s.volume, 0);

        if (viewMode === 'market-cap') {
            const ratio = stock.marketCap / totalMarketCap;
            const size = Math.sqrt(ratio) * 1000; // Scale factor
            return { width: Math.max(80, Math.min(300, size)), height: Math.max(60, Math.min(200, size * 0.7)) };
        } else {
            const ratio = stock.volume / totalVolume;
            const size = Math.sqrt(ratio) * 800; // Scale factor
            return { width: Math.max(80, Math.min(250, size)), height: Math.max(60, Math.min(180, size * 0.7)) };
        }
    };

    const sectors = ['all', 'Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy', 'Industrial'];

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth={false}
            fullScreen
            PaperProps={{
                sx: {
                    backgroundColor: theme.palette.background.default,
                    backgroundImage: 'none',
                }
            }}
        >
            <Box sx={{
                position: 'relative',
                height: '100vh',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
            }}>
                {/* Header */}
                <Box sx={{
                    p: 2,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    backgroundColor: alpha(theme.palette.background.paper, 0.8),
                    backdropFilter: 'blur(10px)',
                    zIndex: 10,
                }}>
                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Stack direction="row" alignItems="center" spacing={3}>
                            <Typography variant="h5" fontWeight="bold">
                                Market Heatmap
                            </Typography>

                            {/* Sector Filter */}
                            <Stack direction="row" spacing={1}>
                                {sectors.map(s => (
                                    <Chip
                                        key={s}
                                        label={s === 'all' ? 'All Sectors' : s}
                                        onClick={() => setSector(s)}
                                        color={sector === s ? 'primary' : 'default'}
                                        variant={sector === s ? 'filled' : 'outlined'}
                                        size="small"
                                    />
                                ))}
                            </Stack>
                        </Stack>

                        <Stack direction="row" alignItems="center" spacing={2}>
                            {/* View Mode Toggle */}
                            <ToggleButtonGroup
                                value={viewMode}
                                exclusive
                                onChange={(_, value) => value && setViewMode(value)}
                                size="small"
                            >
                                <ToggleButton value="market-cap">
                                    <Tooltip title="Size by Market Cap">
                                        <ViewModule />
                                    </Tooltip>
                                </ToggleButton>
                                <ToggleButton value="equal">
                                    <Tooltip title="Equal Size">
                                        <GridView />
                                    </Tooltip>
                                </ToggleButton>
                                <ToggleButton value="volume">
                                    <Tooltip title="Size by Volume">
                                        <ViewComfy />
                                    </Tooltip>
                                </ToggleButton>
                            </ToggleButtonGroup>

                            <IconButton onClick={() => refetch()} size="small">
                                <Refresh />
                            </IconButton>

                            <IconButton onClick={onClose}>
                                <Close />
                            </IconButton>
                        </Stack>
                    </Stack>
                </Box>

                {/* Heatmap Grid */}
                <Box sx={{
                    flex: 1,
                    p: 2,
                    overflow: 'auto',
                    position: 'relative',
                }}>
                    <Box sx={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: 0, // No gap between cells
                        justifyContent: 'center',
                        alignItems: 'flex-start',
                    }}>
                        <AnimatePresence>
                            {stocks.map((stock) => {
                                const size = getCellSize(stock);
                                return (
                                    <motion.div
                                        key={stock.symbol}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.8 }}
                                        transition={{ duration: 0.3 }}
                                        whileHover={{
                                            scale: 1.05,
                                            zIndex: 100,
                                            transition: { duration: 0.2 }
                                        }}
                                        style={{
                                            width: size.width,
                                            height: size.height,
                                            position: 'relative',
                                            cursor: 'pointer',
                                        }}
                                        onClick={() => {
                                            onSymbolSelect?.(stock.symbol);
                                            onClose();
                                        }}
                                        onMouseEnter={() => setHoveredStock(stock)}
                                        onMouseLeave={() => setHoveredStock(null)}
                                    >
                                        <Box
                                            sx={{
                                                width: '100%',
                                                height: '100%',
                                                backgroundColor: getColorIntensity(stock.changePercent),
                                                display: 'flex',
                                                flexDirection: 'column',
                                                justifyContent: 'center',
                                                alignItems: 'center',
                                                border: `1px solid ${alpha(theme.palette.background.default, 0.2)}`,
                                                position: 'relative',
                                                overflow: 'hidden',
                                                '&:hover': {
                                                    boxShadow: `0 0 20px ${alpha(theme.palette.common.white, 0.3)}`,
                                                    border: `1px solid ${alpha(theme.palette.common.white, 0.5)}`,
                                                },
                                            }}
                                        >
                                            <Typography
                                                variant="subtitle2"
                                                fontWeight="bold"
                                                sx={{
                                                    color: Math.abs(stock.changePercent) > 1
                                                        ? theme.palette.common.white
                                                        : theme.palette.text.primary
                                                }}
                                            >
                                                {stock.symbol}
                                            </Typography>
                                            <Stack direction="row" alignItems="center" spacing={0.5}>
                                                {stock.changePercent > 0 ? (
                                                    <TrendingUp sx={{ fontSize: 16, color: 'inherit' }} />
                                                ) : (
                                                    <TrendingDown sx={{ fontSize: 16, color: 'inherit' }} />
                                                )}
                                                <Typography
                                                    variant="body2"
                                                    sx={{
                                                        color: Math.abs(stock.changePercent) > 1
                                                            ? theme.palette.common.white
                                                            : theme.palette.text.primary
                                                    }}
                                                >
                                                    {stock.changePercent > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                                                </Typography>
                                            </Stack>
                                            {size.height > 80 && (
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: Math.abs(stock.changePercent) > 1
                                                            ? alpha(theme.palette.common.white, 0.8)
                                                            : theme.palette.text.secondary,
                                                        mt: 0.5,
                                                    }}
                                                >
                                                    ${stock.price.toFixed(2)}
                                                </Typography>
                                            )}
                                        </Box>
                                    </motion.div>
                                );
                            })}
                        </AnimatePresence>
                    </Box>
                </Box>

                {/* Hover Details */}
                <Fade in={!!hoveredStock}>
                    <Box sx={{
                        position: 'fixed',
                        bottom: 20,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        backgroundColor: alpha(theme.palette.background.paper, 0.95),
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                        borderRadius: 2,
                        p: 2,
                        boxShadow: theme.shadows[10],
                        zIndex: 1000,
                    }}>
                        {hoveredStock && (
                            <Stack direction="row" spacing={3} alignItems="center">
                                <Box>
                                    <Typography variant="h6" fontWeight="bold">
                                        {hoveredStock.symbol}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {hoveredStock.name}
                                    </Typography>
                                </Box>
                                <Divider orientation="vertical" flexItem />
                                <Box>
                                    <Typography variant="body2" color="text.secondary">
                                        Price
                                    </Typography>
                                    <Typography variant="h6">
                                        ${hoveredStock.price.toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="body2" color="text.secondary">
                                        Change
                                    </Typography>
                                    <Typography
                                        variant="h6"
                                        color={hoveredStock.changePercent > 0 ? 'success.main' : 'error.main'}
                                    >
                                        {hoveredStock.changePercent > 0 ? '+' : ''}{hoveredStock.changePercent.toFixed(2)}%
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="body2" color="text.secondary">
                                        Volume
                                    </Typography>
                                    <Typography variant="body1">
                                        {(hoveredStock.volume / 1000000).toFixed(1)}M
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="body2" color="text.secondary">
                                        Market Cap
                                    </Typography>
                                    <Typography variant="body1">
                                        ${(hoveredStock.marketCap / 1000000000).toFixed(0)}B
                                    </Typography>
                                </Box>
                            </Stack>
                        )}
                    </Box>
                </Fade>
            </Box>
        </Dialog>
    );
};

export default MarketHeatmapModal;
