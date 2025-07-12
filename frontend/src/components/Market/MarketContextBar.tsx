import React from 'react';
import { AppBar, Toolbar, Typography, Chip, Box, useTheme, alpha } from '@mui/material';
import { TrendingUp, TrendingDown, Schedule, CheckCircle } from '@mui/icons-material';

interface IndexData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
}

const MarketContextBar: React.FC = () => {
    const theme = useTheme();

    // Mock data - replace with real API data
    const indices: IndexData[] = [
        { symbol: 'SPY', price: 478.32, change: 1.24, changePercent: 0.26 },
        { symbol: 'QQQ', price: 402.15, change: -0.89, changePercent: -0.22 },
        { symbol: 'VIX', price: 12.45, change: -0.32, changePercent: -2.51 },
        { symbol: 'DXY', price: 104.32, change: 0.15, changePercent: 0.14 },
    ];

    const getMarketStatus = () => {
        const now = new Date();
        const hours = now.getHours();
        const minutes = now.getMinutes();
        const time = hours * 60 + minutes;
        const day = now.getDay();

        // Weekend check
        if (day === 0 || day === 6) {
            return { status: 'Market Closed', color: 'default' as const, icon: <Schedule /> };
        }

        // EDT/EST timezone approximation (you should use proper timezone library)
        if (time < 570) return { status: 'Pre-Market', color: 'warning' as const, icon: <Schedule /> };
        if (time < 960) return { status: 'Market Open', color: 'success' as const, icon: <CheckCircle /> };
        if (time < 1200) return { status: 'After-Hours', color: 'info' as const, icon: <Schedule /> };
        return { status: 'Market Closed', color: 'default' as const, icon: <Schedule /> };
    };

    const { status, color, icon } = getMarketStatus();

    return (
        <AppBar
            position="sticky"
            sx={{
                top: 112,
                zIndex: 1100,
                backgroundColor: alpha(theme.palette.background.paper, 0.95),
                backdropFilter: 'blur(10px)',
                borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                boxShadow: `0 2px 8px ${alpha(theme.palette.common.black, 0.1)}`
            }}
        >
            <Toolbar variant="dense" sx={{ minHeight: 48 }}>
                <Chip
                    icon={icon}
                    label={status}
                    color={color}
                    size="small"
                    sx={{
                        mr: 3,
                        fontWeight: 'medium',
                        borderRadius: 2
                    }}
                />

                <Box sx={{ display: 'flex', gap: 3, flexGrow: 1, alignItems: 'center' }}>
                    {indices.map((index) => (
                        <Box
                            key={index.symbol}
                            sx={{
                                display: 'flex',
                                alignItems: 'center',
                                '&:hover': {
                                    '& .symbol': {
                                        color: theme.palette.primary.main
                                    }
                                }
                            }}
                        >
                            <Typography
                                variant="body2"
                                className="symbol"
                                sx={{
                                    mr: 1,
                                    fontWeight: 'medium',
                                    color: theme.palette.text.secondary,
                                    transition: 'color 0.2s'
                                }}
                            >
                                {index.symbol}
                            </Typography>
                            <Typography
                                variant="body2"
                                sx={{
                                    fontWeight: 'bold',
                                    color: theme.palette.text.primary,
                                    fontFamily: 'monospace'
                                }}
                            >
                                ${index.price.toFixed(2)}
                            </Typography>
                            <Chip
                                size="small"
                                icon={index.change > 0 ? <TrendingUp fontSize="small" /> : <TrendingDown fontSize="small" />}
                                label={`${index.change > 0 ? '+' : ''}${index.changePercent.toFixed(2)}%`}
                                sx={{
                                    ml: 1,
                                    height: 24,
                                    backgroundColor: index.change > 0
                                        ? alpha(theme.palette.success.main, 0.1)
                                        : alpha(theme.palette.error.main, 0.1),
                                    color: index.change > 0
                                        ? theme.palette.success.main
                                        : theme.palette.error.main,
                                    '& .MuiChip-icon': {
                                        fontSize: 16,
                                        marginLeft: '4px',
                                        marginRight: '-2px'
                                    }
                                }}
                            />
                        </Box>
                    ))}
                </Box>

                <Typography
                    variant="caption"
                    sx={{
                        color: theme.palette.text.secondary,
                        ml: 2
                    }}
                >
                    Last updated: {new Date().toLocaleTimeString()}
                </Typography>
            </Toolbar>
        </AppBar>
    );
};

export default MarketContextBar; 