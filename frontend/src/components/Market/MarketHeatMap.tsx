/**
 * MarketHeatMap Component - Interactive Market Visualization
 *
 * A beautiful, interactive heat map showing market sectors and stocks
 * with real-time performance data and drill-down capabilities
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
    Box,
    Paper,
    Typography,
    ToggleButton,
    ToggleButtonGroup,
    Chip,
    IconButton,
    Tooltip,
    useTheme,
    alpha,
    Stack,
    Fade,
    Zoom,
    ButtonGroup,
    Button,
    Menu,
    MenuItem,
    Divider,
} from '@mui/material';
import {
    GridView,
    BubbleChart,
    DonutLarge,
    Refresh,
    ZoomIn,
    ZoomOut,
    Fullscreen,
    TrendingUp,
    TrendingDown,
    Info,
    FilterList,
    Speed,
    ShowChart,
    Assessment,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';

interface MarketHeatMapProps {
    height?: number;
    onSymbolClick?: (symbol: string) => void;
    view?: 'sectors' | 'stocks' | 'sp500';
}

interface HeatMapItem {
    id: string;
    name: string;
    symbol?: string;
    value: number; // Market cap or volume
    change: number; // Percentage change
    volume?: number;
    sector?: string;
    marketCap?: number;
    pe?: number;
    signals?: number; // Number of AI signals
}

interface SectorData {
    name: string;
    change: number;
    stocks: HeatMapItem[];
    marketCap: number;
}

const MarketHeatMap: React.FC<MarketHeatMapProps> = ({
    height = 600,
    onSymbolClick,
    view: propView = 'sectors',
}) => {
    const theme = useTheme();
    const [view, setView] = useState(propView);
    const [selectedSector, setSelectedSector] = useState<string | null>(null);
    const [hoveredItem, setHoveredItem] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState('1d');
    const [sortBy, setSortBy] = useState<'change' | 'volume' | 'marketCap'>('change');
    const [filterMenu, setFilterMenu] = useState<null | HTMLElement>(null);

    // Fetch market data
    const { data: marketData, isLoading, refetch } = useQuery({
        queryKey: ['market-heatmap', view, timeframe],
        queryFn: async () => {
            // In production, fetch real market data
            return generateMockMarketData();
        },
        // Disabled auto-refresh to prevent constant updating
        staleTime: 300000, // Keep data fresh for 5 minutes
    });

    // Generate mock market data
    const generateMockMarketData = (): SectorData[] => {
        const sectors = [
            {
                name: 'Technology',
                stocks: [
                    { symbol: 'AAPL', name: 'Apple Inc.', marketCap: 3000000000000, change: 2.5 },
                    { symbol: 'MSFT', name: 'Microsoft', marketCap: 2800000000000, change: 1.8 },
                    { symbol: 'NVDA', name: 'NVIDIA', marketCap: 1200000000000, change: 5.2 },
                    { symbol: 'GOOGL', name: 'Alphabet', marketCap: 1700000000000, change: -0.5 },
                    { symbol: 'META', name: 'Meta', marketCap: 900000000000, change: 3.1 },
                    { symbol: 'TSLA', name: 'Tesla', marketCap: 800000000000, change: -2.3 },
                ],
            },
            {
                name: 'Healthcare',
                stocks: [
                    { symbol: 'JNJ', name: 'Johnson & Johnson', marketCap: 450000000000, change: 0.8 },
                    { symbol: 'UNH', name: 'UnitedHealth', marketCap: 500000000000, change: 1.2 },
                    { symbol: 'PFE', name: 'Pfizer', marketCap: 280000000000, change: -1.5 },
                    { symbol: 'LLY', name: 'Eli Lilly', marketCap: 420000000000, change: 3.8 },
                ],
            },
            {
                name: 'Financial',
                stocks: [
                    { symbol: 'JPM', name: 'JPMorgan Chase', marketCap: 480000000000, change: 1.5 },
                    { symbol: 'BAC', name: 'Bank of America', marketCap: 280000000000, change: 2.1 },
                    { symbol: 'WFC', name: 'Wells Fargo', marketCap: 180000000000, change: -0.3 },
                    { symbol: 'GS', name: 'Goldman Sachs', marketCap: 120000000000, change: 1.9 },
                ],
            },
            {
                name: 'Consumer',
                stocks: [
                    { symbol: 'AMZN', name: 'Amazon', marketCap: 1500000000000, change: 2.8 },
                    { symbol: 'WMT', name: 'Walmart', marketCap: 420000000000, change: 0.5 },
                    { symbol: 'HD', name: 'Home Depot', marketCap: 350000000000, change: -0.8 },
                    { symbol: 'MCD', name: "McDonald's", marketCap: 200000000000, change: 1.1 },
                ],
            },
            {
                name: 'Energy',
                stocks: [
                    { symbol: 'XOM', name: 'Exxon Mobil', marketCap: 450000000000, change: -1.2 },
                    { symbol: 'CVX', name: 'Chevron', marketCap: 350000000000, change: -0.9 },
                    { symbol: 'COP', name: 'ConocoPhillips', marketCap: 140000000000, change: -2.1 },
                ],
            },
            {
                name: 'Industrial',
                stocks: [
                    { symbol: 'BA', name: 'Boeing', marketCap: 140000000000, change: 3.2 },
                    { symbol: 'CAT', name: 'Caterpillar', marketCap: 150000000000, change: 1.7 },
                    { symbol: 'GE', name: 'General Electric', marketCap: 120000000000, change: 2.4 },
                ],
            },
        ];

        // Calculate sector data
        return sectors.map(sector => ({
            name: sector.name,
            change: sector.stocks.reduce((sum, stock) => sum + stock.change, 0) / sector.stocks.length,
            marketCap: sector.stocks.reduce((sum, stock) => sum + stock.marketCap, 0),
            stocks: sector.stocks.map(stock => ({
                id: stock.symbol,
                name: stock.name,
                symbol: stock.symbol,
                value: stock.marketCap,
                change: stock.change + (Math.random() - 0.5) * 0.5, // Add some randomness
                volume: Math.floor(Math.random() * 50000000) + 10000000,
                sector: sector.name,
                marketCap: stock.marketCap,
                pe: Math.random() * 30 + 10,
                signals: Math.floor(Math.random() * 5),
            })),
        }));
    };

    // Calculate color based on performance
    const getColor = (change: number) => {
        const intensity = Math.min(Math.abs(change) / 5, 1); // Cap at 5% for max intensity

        if (change > 0) {
            // Green shades for positive
            return alpha(theme.palette.success.main, 0.2 + intensity * 0.6);
        } else if (change < 0) {
            // Red shades for negative
            return alpha(theme.palette.error.main, 0.2 + intensity * 0.6);
        } else {
            // Gray for no change
            return alpha(theme.palette.grey[500], 0.3);
        }
    };

    // Calculate box size based on market cap
    const getBoxSize = (marketCap: number, totalMarketCap: number) => {
        const percentage = (marketCap / totalMarketCap) * 100;
        return {
            width: `${Math.max(percentage * 2, 10)}%`, // Min 10% width
            height: `${Math.max(percentage * 3, 60)}px`, // Min 60px height
        };
    };

    // Render individual stock/sector box
    const HeatMapBox: React.FC<{
        item: HeatMapItem | SectorData;
        size?: { width: string; height: string };
        onClick?: () => void;
    }> = ({ item, size, onClick }) => {
        const isHovered = hoveredItem === ('symbol' in item ? item.symbol : item.name);
        const change = 'change' in item ? item.change : 0;
        const hasSignals = 'signals' in item && item.signals && item.signals > 0;

        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2 }}
                style={{
                    ...size,
                    minWidth: '120px',
                    minHeight: '80px',
                    margin: '4px',
                    position: 'relative',
                    cursor: onClick ? 'pointer' : 'default',
                }}
                onMouseEnter={() => setHoveredItem('symbol' in item ? item.symbol! : item.name)}
                onMouseLeave={() => setHoveredItem(null)}
                onClick={onClick}
            >
                <Paper
                    elevation={isHovered ? 8 : 2}
                    sx={{
                        width: '100%',
                        height: '100%',
                        p: 1.5,
                        background: getColor(change),
                        border: `1px solid ${alpha(
                            change > 0 ? theme.palette.success.main :
                                change < 0 ? theme.palette.error.main :
                                    theme.palette.grey[500],
                            0.3
                        )}`,
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        position: 'relative',
                        overflow: 'hidden',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                            borderColor: theme.palette.primary.main,
                            borderWidth: '2px',
                        },
                    }}
                >
                    {/* Background pattern */}
                    <Box
                        sx={{
                            position: 'absolute',
                            top: -20,
                            right: -20,
                            width: 100,
                            height: 100,
                            borderRadius: '50%',
                            background: alpha(
                                change > 0 ? theme.palette.success.main : theme.palette.error.main,
                                0.1
                            ),
                            filter: 'blur(30px)',
                        }}
                    />

                    {/* Content */}
                    <Box sx={{ position: 'relative', zIndex: 1 }}>
                        <Typography
                            variant="subtitle2"
                            sx={{
                                fontWeight: 'bold',
                                color: theme.palette.text.primary,
                                mb: 0.5,
                            }}
                        >
                            {'symbol' in item ? item.symbol : item.name}
                        </Typography>
                        <Typography
                            variant="caption"
                            sx={{
                                color: theme.palette.text.secondary,
                                display: 'block',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                            }}
                        >
                            {item.name}
                        </Typography>
                    </Box>

                    <Box sx={{ position: 'relative', zIndex: 1 }}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                            <Typography
                                variant="h6"
                                sx={{
                                    fontWeight: 'bold',
                                    color: change > 0 ? theme.palette.success.main :
                                        change < 0 ? theme.palette.error.main :
                                            theme.palette.text.primary,
                                    display: 'flex',
                                    alignItems: 'center',
                                }}
                            >
                                {change > 0 ? <TrendingUp sx={{ mr: 0.5, fontSize: 20 }} /> :
                                    change < 0 ? <TrendingDown sx={{ mr: 0.5, fontSize: 20 }} /> : null}
                                {change > 0 ? '+' : ''}{change.toFixed(2)}%
                            </Typography>
                            {hasSignals && (
                                <Chip
                                    size="small"
                                    label={`${(item as HeatMapItem).signals}`}
                                    icon={<Speed sx={{ fontSize: 14 }} />}
                                    sx={{
                                        height: 20,
                                        backgroundColor: alpha(theme.palette.primary.main, 0.2),
                                        color: theme.palette.primary.main,
                                        '& .MuiChip-label': {
                                            px: 0.5,
                                            fontSize: '0.7rem',
                                        },
                                    }}
                                />
                            )}
                        </Stack>

                        {'volume' in item && (
                            <Typography variant="caption" color="text.secondary">
                                Vol: {(item.volume! / 1000000).toFixed(1)}M
                            </Typography>
                        )}
                    </Box>

                    {/* Hover overlay */}
                    <AnimatePresence>
                        {isHovered && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                style={{
                                    position: 'absolute',
                                    bottom: 0,
                                    left: 0,
                                    right: 0,
                                    padding: '8px',
                                    background: alpha(theme.palette.background.paper, 0.95),
                                    backdropFilter: 'blur(10px)',
                                    borderTop: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                                }}
                            >
                                {'marketCap' in item && (
                                    <Stack spacing={0.5}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="caption">Market Cap:</Typography>
                                            <Typography variant="caption" fontWeight="bold">
                                                ${(item.marketCap! / 1000000000).toFixed(0)}B
                                            </Typography>
                                        </Box>
                                        {'pe' in item && (
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography variant="caption">P/E:</Typography>
                                                <Typography variant="caption" fontWeight="bold">
                                                    {item.pe!.toFixed(1)}
                                                </Typography>
                                            </Box>
                                        )}
                                    </Stack>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </Paper>
            </motion.div>
        );
    };

    // Render sector view
    const renderSectorView = () => {
        if (!marketData) return null;

        const totalMarketCap = marketData.reduce((sum, sector) => sum + sector.marketCap, 0);

        return (
            <Box
                sx={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '100%',
                    p: 2,
                }}
            >
                {marketData.map((sector) => (
                    <HeatMapBox
                        key={sector.name}
                        item={sector}
                        size={getBoxSize(sector.marketCap, totalMarketCap)}
                        onClick={() => {
                            setSelectedSector(sector.name);
                            setView('stocks');
                        }}
                    />
                ))}
            </Box>
        );
    };

    // Render stocks view
    const renderStocksView = () => {
        if (!marketData) return null;

        const stocks = selectedSector
            ? marketData.find(s => s.name === selectedSector)?.stocks || []
            : marketData.flatMap(s => s.stocks);

        const sortedStocks = [...stocks].sort((a, b) => {
            switch (sortBy) {
                case 'change':
                    return b.change - a.change;
                case 'volume':
                    return (b.volume || 0) - (a.volume || 0);
                case 'marketCap':
                    return (b.marketCap || 0) - (a.marketCap || 0);
                default:
                    return 0;
            }
        });

        const totalMarketCap = stocks.reduce((sum, stock) => sum + (stock.marketCap || 0), 0);

        return (
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                {selectedSector && (
                    <Box sx={{ p: 2, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                        <Stack direction="row" alignItems="center" spacing={2}>
                            <Button
                                size="small"
                                onClick={() => {
                                    setSelectedSector(null);
                                    setView('sectors');
                                }}
                            >
                                ← Back to Sectors
                            </Button>
                            <Typography variant="h6">{selectedSector}</Typography>
                        </Stack>
                    </Box>
                )}

                <Box
                    sx={{
                        flex: 1,
                        display: 'flex',
                        flexWrap: 'wrap',
                        justifyContent: 'center',
                        alignItems: 'flex-start',
                        overflow: 'auto',
                        p: 2,
                    }}
                >
                    {sortedStocks.map((stock) => (
                        <HeatMapBox
                            key={stock.id}
                            item={stock}
                            size={
                                view === 'sp500'
                                    ? { width: '140px', height: '100px' }
                                    : getBoxSize(stock.marketCap || 0, totalMarketCap)
                            }
                            onClick={() => onSymbolClick?.(stock.symbol!)}
                        />
                    ))}
                </Box>
            </Box>
        );
    };

    return (
        <Paper
            elevation={0}
            sx={{
                height,
                display: 'flex',
                flexDirection: 'column',
                background: alpha(theme.palette.background.paper, 0.5),
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                borderRadius: 2,
                overflow: 'hidden',
            }}
        >
            {/* Header */}
            <Box
                sx={{
                    p: 2,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    background: alpha(theme.palette.background.paper, 0.5),
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Assessment sx={{ color: theme.palette.primary.main }} />
                            Market Heat Map
                        </Typography>

                        <ToggleButtonGroup
                            value={view}
                            exclusive
                            onChange={(e, value) => value && setView(value)}
                            size="small"
                        >
                            <ToggleButton value="sectors">
                                <Tooltip title="Sector View">
                                    <DonutLarge fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                            <ToggleButton value="stocks">
                                <Tooltip title="Stock View">
                                    <GridView fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                            <ToggleButton value="sp500">
                                <Tooltip title="S&P 500">
                                    <BubbleChart fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                        </ToggleButtonGroup>

                        <Divider orientation="vertical" flexItem />

                        <ToggleButtonGroup
                            value={timeframe}
                            exclusive
                            onChange={(e, value) => value && setTimeframe(value)}
                            size="small"
                        >
                            <ToggleButton value="1d">1D</ToggleButton>
                            <ToggleButton value="1w">1W</ToggleButton>
                            <ToggleButton value="1m">1M</ToggleButton>
                            <ToggleButton value="3m">3M</ToggleButton>
                        </ToggleButtonGroup>
                    </Stack>

                    <Stack direction="row" spacing={1}>
                        <Button
                            size="small"
                            startIcon={<FilterList />}
                            onClick={(e) => setFilterMenu(e.currentTarget)}
                        >
                            Sort: {sortBy}
                        </Button>

                        <IconButton size="small" onClick={() => refetch()}>
                            <Refresh />
                        </IconButton>

                        <IconButton size="small">
                            <Fullscreen />
                        </IconButton>
                    </Stack>
                </Stack>

                {/* Summary stats */}
                <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                    <Chip
                        size="small"
                        icon={<TrendingUp />}
                        label={`Gainers: ${marketData?.flatMap(s => s.stocks).filter(s => s.change > 0).length || 0}`}
                        color="success"
                        variant="outlined"
                    />
                    <Chip
                        size="small"
                        icon={<TrendingDown />}
                        label={`Losers: ${marketData?.flatMap(s => s.stocks).filter(s => s.change < 0).length || 0}`}
                        color="error"
                        variant="outlined"
                    />
                    <Chip
                        size="small"
                        icon={<ShowChart />}
                        label={`Avg: ${marketData ? (marketData.reduce((sum, s) => sum + s.change, 0) / marketData.length).toFixed(2) : 0}%`}
                        variant="outlined"
                    />
                </Stack>
            </Box>

            {/* Content */}
            <Box sx={{ flex: 1, position: 'relative' }}>
                {isLoading ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        <Typography variant="body2" color="text.secondary">
                            Loading market data...
                        </Typography>
                    </Box>
                ) : view === 'sectors' ? (
                    renderSectorView()
                ) : (
                    renderStocksView()
                )}
            </Box>

            {/* Sort Menu */}
            <Menu
                anchorEl={filterMenu}
                open={Boolean(filterMenu)}
                onClose={() => setFilterMenu(null)}
            >
                <MenuItem onClick={() => { setSortBy('change'); setFilterMenu(null); }}>
                    Sort by Change %
                </MenuItem>
                <MenuItem onClick={() => { setSortBy('volume'); setFilterMenu(null); }}>
                    Sort by Volume
                </MenuItem>
                <MenuItem onClick={() => { setSortBy('marketCap'); setFilterMenu(null); }}>
                    Sort by Market Cap
                </MenuItem>
            </Menu>
        </Paper>
    );
};

export default MarketHeatMap;
