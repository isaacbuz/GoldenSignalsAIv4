/**
 * ExplodedHeatMap Component - Moomoo-style Market Visualization
 * 
 * A treemap-based heat map where all tiles fit together perfectly like a puzzle
 * with no gaps, similar to moomoo's market heat map
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
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
    Slider,
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
    FullscreenExit,
    Home,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import * as d3 from 'd3';

interface ExplodedHeatMapProps {
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
    children?: HeatMapItem[];
}

interface TreemapNode extends d3.HierarchyRectangularNode<HeatMapItem> {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
}

const ExplodedHeatMap: React.FC<ExplodedHeatMapProps> = ({
    height = 600,
    onSymbolClick,
    view: propView = 'sectors',
}) => {
    const theme = useTheme();
    const containerRef = useRef<HTMLDivElement>(null);
    const [view, setView] = useState(propView);
    const [selectedSector, setSelectedSector] = useState<string | null>(null);
    const [hoveredItem, setHoveredItem] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState('1d');
    const [sortBy, setSortBy] = useState<'change' | 'volume' | 'marketCap'>('marketCap');
    const [filterMenu, setFilterMenu] = useState<null | HTMLElement>(null);
    const [zoomLevel, setZoomLevel] = useState(1);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    // Update dimensions on resize
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const { width, height } = containerRef.current.getBoundingClientRect();
                setDimensions({ width, height });
            }
        };

        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    // Fetch market data
    const { data: marketData, isLoading, refetch } = useQuery({
        queryKey: ['market-heatmap', view, timeframe],
        queryFn: async () => {
            // In production, fetch real market data
            return generateMockMarketData();
        },
        refetchInterval: 30000,
    });

    // Generate mock market data with hierarchical structure
    const generateMockMarketData = (): HeatMapItem => {
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
                    { symbol: 'ORCL', name: 'Oracle', marketCap: 300000000000, change: 1.2 },
                    { symbol: 'ADBE', name: 'Adobe', marketCap: 250000000000, change: -1.8 },
                ],
            },
            {
                name: 'Healthcare',
                stocks: [
                    { symbol: 'JNJ', name: 'Johnson & Johnson', marketCap: 450000000000, change: 0.8 },
                    { symbol: 'UNH', name: 'UnitedHealth', marketCap: 500000000000, change: 1.2 },
                    { symbol: 'PFE', name: 'Pfizer', marketCap: 280000000000, change: -1.5 },
                    { symbol: 'LLY', name: 'Eli Lilly', marketCap: 420000000000, change: 3.8 },
                    { symbol: 'ABBV', name: 'AbbVie', marketCap: 300000000000, change: 0.5 },
                    { symbol: 'MRK', name: 'Merck', marketCap: 280000000000, change: -0.3 },
                ],
            },
            {
                name: 'Financial',
                stocks: [
                    { symbol: 'JPM', name: 'JPMorgan Chase', marketCap: 480000000000, change: 1.5 },
                    { symbol: 'BAC', name: 'Bank of America', marketCap: 280000000000, change: 2.1 },
                    { symbol: 'WFC', name: 'Wells Fargo', marketCap: 180000000000, change: -0.3 },
                    { symbol: 'GS', name: 'Goldman Sachs', marketCap: 120000000000, change: 1.9 },
                    { symbol: 'MS', name: 'Morgan Stanley', marketCap: 150000000000, change: 0.7 },
                    { symbol: 'C', name: 'Citigroup', marketCap: 100000000000, change: -1.2 },
                ],
            },
            {
                name: 'Consumer',
                stocks: [
                    { symbol: 'AMZN', name: 'Amazon', marketCap: 1500000000000, change: 2.8 },
                    { symbol: 'WMT', name: 'Walmart', marketCap: 420000000000, change: 0.5 },
                    { symbol: 'HD', name: 'Home Depot', marketCap: 350000000000, change: -0.8 },
                    { symbol: 'MCD', name: "McDonald's", marketCap: 200000000000, change: 1.1 },
                    { symbol: 'NKE', name: 'Nike', marketCap: 180000000000, change: 2.3 },
                    { symbol: 'SBUX', name: 'Starbucks', marketCap: 120000000000, change: -1.5 },
                ],
            },
            {
                name: 'Energy',
                stocks: [
                    { symbol: 'XOM', name: 'Exxon Mobil', marketCap: 450000000000, change: -1.2 },
                    { symbol: 'CVX', name: 'Chevron', marketCap: 350000000000, change: -0.9 },
                    { symbol: 'COP', name: 'ConocoPhillips', marketCap: 140000000000, change: -2.1 },
                    { symbol: 'SLB', name: 'Schlumberger', marketCap: 80000000000, change: -1.8 },
                ],
            },
            {
                name: 'Industrial',
                stocks: [
                    { symbol: 'BA', name: 'Boeing', marketCap: 140000000000, change: 3.2 },
                    { symbol: 'CAT', name: 'Caterpillar', marketCap: 150000000000, change: 1.7 },
                    { symbol: 'GE', name: 'General Electric', marketCap: 120000000000, change: 2.4 },
                    { symbol: 'HON', name: 'Honeywell', marketCap: 130000000000, change: 0.9 },
                ],
            },
        ];

        // Transform to hierarchical structure
        const children = sectors.map(sector => ({
            id: sector.name,
            name: sector.name,
            value: sector.stocks.reduce((sum, stock) => sum + stock.marketCap, 0),
            change: sector.stocks.reduce((sum, stock) => sum + stock.change, 0) / sector.stocks.length,
            marketCap: sector.stocks.reduce((sum, stock) => sum + stock.marketCap, 0),
            children: sector.stocks.map(stock => ({
                id: stock.symbol,
                name: stock.name,
                symbol: stock.symbol,
                value: stock.marketCap,
                change: stock.change + (Math.random() - 0.5) * 0.5,
                volume: Math.floor(Math.random() * 50000000) + 10000000,
                sector: sector.name,
                marketCap: stock.marketCap,
                pe: Math.random() * 30 + 10,
                signals: Math.floor(Math.random() * 5),
            })),
        }));

        return {
            id: 'root',
            name: 'Market',
            value: children.reduce((sum, child) => sum + child.value, 0),
            change: children.reduce((sum, child) => sum + child.change, 0) / children.length,
            children,
        };
    };

    // Calculate color based on performance
    const getColor = (change: number) => {
        const intensity = Math.min(Math.abs(change) / 5, 1);

        if (change > 2) {
            return theme.palette.success.dark;
        } else if (change > 0) {
            return alpha(theme.palette.success.main, 0.5 + intensity * 0.5);
        } else if (change < -2) {
            return theme.palette.error.dark;
        } else if (change < 0) {
            return alpha(theme.palette.error.main, 0.5 + intensity * 0.5);
        } else {
            return alpha(theme.palette.grey[500], 0.3);
        }
    };

    // Calculate treemap layout
    const treemapData = useMemo(() => {
        if (!marketData || dimensions.width === 0) return null;

        const root = d3.hierarchy(marketData)
            .sum(d => d.value || 0)
            .sort((a, b) => (b.value || 0) - (a.value || 0));

        const treemap = d3.treemap<HeatMapItem>()
            .size([dimensions.width, dimensions.height - 120]) // Account for header
            .paddingInner(2)
            .paddingOuter(4)
            .paddingTop(20)
            .round(true);

        return treemap(root);
    }, [marketData, dimensions, zoomLevel]);

    // Render treemap tile
    const renderTile = (node: TreemapNode, depth: number = 0) => {
        const width = node.x1 - node.x0;
        const height = node.y1 - node.y0;
        const data = node.data;
        const isHovered = hoveredItem === data.id;
        const hasChildren = node.children && node.children.length > 0;

        // Don't render if too small
        if (width < 30 || height < 30) return null;

        // For sector view, only show sectors
        if (view === 'sectors' && depth > 0 && !selectedSector) {
            return null;
        }

        // For selected sector, show its stocks
        if (selectedSector && depth === 0 && data.name !== selectedSector) {
            return null;
        }

        const showLabel = width > 60 && height > 40;
        const showDetails = width > 100 && height > 60;

        return (
            <motion.div
                key={data.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{
                    opacity: 1,
                    scale: 1,
                    x: node.x0 * zoomLevel,
                    y: node.y0 * zoomLevel,
                    width: width * zoomLevel,
                    height: height * zoomLevel,
                }}
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
                style={{
                    position: 'absolute',
                    cursor: hasChildren || data.symbol ? 'pointer' : 'default',
                }}
                onMouseEnter={() => setHoveredItem(data.id)}
                onMouseLeave={() => setHoveredItem(null)}
                onClick={() => {
                    if (data.symbol) {
                        onSymbolClick?.(data.symbol);
                    } else if (hasChildren && view === 'sectors') {
                        setSelectedSector(data.name);
                    }
                }}
            >
                <Box
                    sx={{
                        width: '100%',
                        height: '100%',
                        background: getColor(data.change || 0),
                        border: `1px solid ${alpha(
                            data.change > 0 ? theme.palette.success.main :
                                data.change < 0 ? theme.palette.error.main :
                                    theme.palette.grey[500],
                            isHovered ? 0.8 : 0.3
                        )}`,
                        borderWidth: isHovered ? 2 : 1,
                        borderRadius: 1,
                        p: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        overflow: 'hidden',
                        position: 'relative',
                        transition: 'all 0.2s ease',
                        '&:hover': {
                            zIndex: 10,
                            boxShadow: theme.shadows[8],
                        },
                    }}
                >
                    {/* Background gradient */}
                    <Box
                        sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            background: `linear-gradient(135deg, transparent 0%, ${alpha(
                                data.change > 0 ? theme.palette.success.main : theme.palette.error.main,
                                0.1
                            )} 100%)`,
                            pointerEvents: 'none',
                        }}
                    />

                    {/* Content */}
                    <Box sx={{ position: 'relative', zIndex: 1 }}>
                        {showLabel && (
                            <>
                                <Typography
                                    variant={width > 150 ? 'subtitle2' : 'caption'}
                                    sx={{
                                        fontWeight: 'bold',
                                        color: theme.palette.getContrastText(getColor(data.change || 0)),
                                        lineHeight: 1.2,
                                        mb: 0.5,
                                    }}
                                    noWrap
                                >
                                    {data.symbol || data.name}
                                </Typography>
                                {showDetails && data.name !== data.symbol && (
                                    <Typography
                                        variant="caption"
                                        sx={{
                                            color: alpha(
                                                theme.palette.getContrastText(getColor(data.change || 0)),
                                                0.8
                                            ),
                                            display: 'block',
                                            lineHeight: 1.2,
                                        }}
                                        noWrap
                                    >
                                        {data.name}
                                    </Typography>
                                )}
                            </>
                        )}
                    </Box>

                    {showLabel && (
                        <Box sx={{ position: 'relative', zIndex: 1 }}>
                            <Stack direction="row" alignItems="center" justifyContent="space-between">
                                <Typography
                                    variant={width > 150 ? 'body2' : 'caption'}
                                    sx={{
                                        fontWeight: 'bold',
                                        color: theme.palette.getContrastText(getColor(data.change || 0)),
                                        display: 'flex',
                                        alignItems: 'center',
                                    }}
                                >
                                    {data.change > 0 && '+'}
                                    {data.change.toFixed(2)}%
                                </Typography>
                                {data.signals && data.signals > 0 && showDetails && (
                                    <Chip
                                        size="small"
                                        label={data.signals}
                                        icon={<Speed sx={{ fontSize: 12 }} />}
                                        sx={{
                                            height: 18,
                                            backgroundColor: alpha(theme.palette.primary.main, 0.2),
                                            color: theme.palette.primary.main,
                                            '& .MuiChip-label': {
                                                px: 0.5,
                                                fontSize: '0.65rem',
                                            },
                                        }}
                                    />
                                )}
                            </Stack>
                            {showDetails && data.marketCap && (
                                <Typography
                                    variant="caption"
                                    sx={{
                                        color: alpha(
                                            theme.palette.getContrastText(getColor(data.change || 0)),
                                            0.7
                                        ),
                                        fontSize: '0.65rem',
                                    }}
                                >
                                    ${(data.marketCap / 1000000000).toFixed(0)}B
                                </Typography>
                            )}
                        </Box>
                    )}

                    {/* Hover overlay */}
                    <AnimatePresence>
                        {isHovered && showDetails && (
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
                                <Stack spacing={0.5}>
                                    {data.volume && (
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="caption">Volume:</Typography>
                                            <Typography variant="caption" fontWeight="bold">
                                                {(data.volume / 1000000).toFixed(1)}M
                                            </Typography>
                                        </Box>
                                    )}
                                    {data.pe && (
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="caption">P/E:</Typography>
                                            <Typography variant="caption" fontWeight="bold">
                                                {data.pe.toFixed(1)}
                                            </Typography>
                                        </Box>
                                    )}
                                </Stack>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </Box>
            </motion.div>
        );
    };

    // Render all tiles
    const renderTreemap = () => {
        if (!treemapData) return null;

        const nodes = selectedSector
            ? treemapData.descendants().filter(d =>
                d.depth === 0 || (d.parent && d.parent.data.name === selectedSector)
            )
            : view === 'sectors'
                ? treemapData.children || []
                : treemapData.descendants().filter(d => d.depth > 0);

        return (
            <Box
                sx={{
                    position: 'relative',
                    width: '100%',
                    height: '100%',
                    overflow: 'hidden',
                }}
            >
                {nodes.map((node) => renderTile(node as TreemapNode, node.depth))}
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
                position: 'relative',
            }}
        >
            {/* Header */}
            <Box
                sx={{
                    p: 2,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    background: alpha(theme.palette.background.paper, 0.5),
                    zIndex: 10,
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Assessment sx={{ color: theme.palette.primary.main }} />
                            Market Heat Map
                        </Typography>

                        {selectedSector && (
                            <>
                                <Divider orientation="vertical" flexItem />
                                <Button
                                    size="small"
                                    startIcon={<Home />}
                                    onClick={() => setSelectedSector(null)}
                                >
                                    All Sectors
                                </Button>
                                <Typography variant="subtitle2" color="text.secondary">
                                    / {selectedSector}
                                </Typography>
                            </>
                        )}

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

                    <Stack direction="row" spacing={1} alignItems="center">
                        <Box sx={{ width: 120 }}>
                            <Slider
                                value={zoomLevel}
                                onChange={(e, value) => setZoomLevel(value as number)}
                                min={0.5}
                                max={2}
                                step={0.1}
                                size="small"
                                valueLabelDisplay="auto"
                                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                            />
                        </Box>

                        <IconButton size="small" onClick={() => setZoomLevel(1)}>
                            <Home />
                        </IconButton>

                        <IconButton size="small" onClick={() => refetch()}>
                            <Refresh />
                        </IconButton>

                        <IconButton
                            size="small"
                            onClick={() => setIsFullscreen(!isFullscreen)}
                        >
                            {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                        </IconButton>
                    </Stack>
                </Stack>

                {/* Summary stats */}
                <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                    <Chip
                        size="small"
                        icon={<TrendingUp />}
                        label={`Gainers: ${marketData?.children?.flatMap(s => s.children || [])
                            .filter(s => s.change > 0).length || 0
                            }`}
                        color="success"
                        variant="outlined"
                    />
                    <Chip
                        size="small"
                        icon={<TrendingDown />}
                        label={`Losers: ${marketData?.children?.flatMap(s => s.children || [])
                            .filter(s => s.change < 0).length || 0
                            }`}
                        color="error"
                        variant="outlined"
                    />
                    <Chip
                        size="small"
                        icon={<ShowChart />}
                        label={`Market: ${marketData?.change?.toFixed(2) || 0}%`}
                        variant="outlined"
                    />
                </Stack>
            </Box>

            {/* Content */}
            <Box
                ref={containerRef}
                sx={{
                    flex: 1,
                    position: 'relative',
                    background: theme.palette.background.default,
                }}
            >
                {isLoading ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        <Typography variant="body2" color="text.secondary">
                            Loading market data...
                        </Typography>
                    </Box>
                ) : (
                    renderTreemap()
                )}
            </Box>
        </Paper>
    );
};

export default ExplodedHeatMap; 