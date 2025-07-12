import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    IconButton,
    TextField,
    InputAdornment,
    Chip,
    Stack,
    Paper,
    Badge,
} from '@mui/material';
import {
    Search,
    TrendingUp,
    TrendingDown,
    Notifications,
    Circle as LiveIcon,
    ShowChart,
    SmartToy,
} from '@mui/icons-material';

interface MarketTicker {
    symbol: string;
    price: number;
    change: number;
}

interface TradingTopBarProps {
    onAIChatOpen?: () => void;
}

export const TradingTopBar: React.FC<TradingTopBarProps> = ({ onAIChatOpen }) => {
    const [searchValue, setSearchValue] = useState('');
    const [currentTime, setCurrentTime] = useState(new Date());

    // Simplified market data
    const [marketTickers, setMarketTickers] = useState<MarketTicker[]>([
        { symbol: 'SPY', price: 456.78, change: 0.51 },
        { symbol: 'QQQ', price: 398.23, change: -0.36 },
        { symbol: 'DIA', price: 378.91, change: 0.23 },
        { symbol: 'VIX', price: 18.42, change: -3.82 },
    ]);

    // Update time and simulate market data
    useEffect(() => {
        const timer = setInterval(() => {
            setCurrentTime(new Date());
            // Simulate live price updates
            setMarketTickers(prev => prev.map(ticker => ({
                ...ticker,
                price: ticker.price + (Math.random() - 0.5) * 0.5,
                change: ticker.change + (Math.random() - 0.5) * 0.1,
            })));
        }, 3000);
        return () => clearInterval(timer);
    }, []);

    const isMarketOpen = () => {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();
        return day >= 1 && day <= 5 && hour >= 9 && hour < 16;
    };

    return (
        <Box
            sx={{
                bgcolor: '#0D1421',
                borderBottom: '1px solid rgba(255,255,255,0.1)',
                px: 2,
                py: 1,
            }}
        >
            <Stack direction="row" alignItems="center" spacing={3}>
                {/* Logo and App Name */}
                <Stack direction="row" alignItems="center" spacing={1}>
                    <ShowChart sx={{ color: '#8B5CF6', fontSize: 28 }} />
                    <Typography
                        variant="h6"
                        sx={{
                            fontWeight: 700,
                            background: 'linear-gradient(45deg, #8B5CF6, #06B6D4)',
                            backgroundClip: 'text',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                        }}
                    >
                        GoldenSignals AI
                    </Typography>
                </Stack>

                {/* Market Status */}
                <Stack direction="row" alignItems="center" spacing={1}>
                    <LiveIcon
                        sx={{
                            color: isMarketOpen() ? '#00C851' : '#FF4444',
                            fontSize: 12,
                            animation: isMarketOpen() ? 'pulse 2s infinite' : 'none'
                        }}
                    />
                    <Typography variant="caption" sx={{ color: '#8E9BAE' }}>
                        {isMarketOpen() ? 'MARKET OPEN' : 'MARKET CLOSED'}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#8E9BAE' }}>
                        {currentTime.toLocaleTimeString('en-US', {
                            hour12: false,
                            hour: '2-digit',
                            minute: '2-digit'
                        })} ET
                    </Typography>
                </Stack>

                {/* Market Tickers */}
                <Stack direction="row" spacing={2}>
                    {marketTickers.map((ticker) => (
                        <Box key={ticker.symbol} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="caption" sx={{ color: 'white', fontWeight: 'bold' }}>
                                {ticker.symbol}
                            </Typography>
                            <Typography variant="caption" sx={{ color: '#8E9BAE' }}>
                                ${ticker.price.toFixed(2)}
                            </Typography>
                            {ticker.change > 0 ? (
                                <TrendingUp sx={{ fontSize: 12, color: '#00C851' }} />
                            ) : (
                                <TrendingDown sx={{ fontSize: 12, color: '#FF4444' }} />
                            )}
                            <Typography
                                variant="caption"
                                sx={{
                                    color: ticker.change > 0 ? '#00C851' : '#FF4444',
                                    fontWeight: 'bold'
                                }}
                            >
                                {ticker.change > 0 ? '+' : ''}{ticker.change.toFixed(2)}%
                            </Typography>
                        </Box>
                    ))}
                </Stack>

                <Box sx={{ flex: 1 }} />

                {/* Search */}
                <TextField
                    size="small"
                    placeholder="Search symbols..."
                    value={searchValue}
                    onChange={(e) => setSearchValue(e.target.value.toUpperCase())}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <Search sx={{ fontSize: 18, color: '#8E9BAE' }} />
                            </InputAdornment>
                        ),
                    }}
                    sx={{
                        width: 200,
                        '& .MuiOutlinedInput-root': {
                            bgcolor: 'rgba(255,255,255,0.05)',
                            borderColor: 'rgba(255,255,255,0.1)',
                            '& input': {
                                color: 'white',
                                '&::placeholder': {
                                    color: '#8E9BAE',
                                    opacity: 1
                                }
                            },
                            '&:hover .MuiOutlinedInput-notchedOutline': {
                                borderColor: '#8B5CF6',
                            },
                            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                                borderColor: '#8B5CF6',
                            },
                        },
                    }}
                />

                {/* AI Chat Button */}
                <IconButton
                    onClick={onAIChatOpen}
                    sx={{
                        bgcolor: 'rgba(139, 92, 246, 0.1)',
                        border: '1px solid rgba(139, 92, 246, 0.3)',
                        '&:hover': {
                            bgcolor: 'rgba(139, 92, 246, 0.2)',
                        }
                    }}
                >
                    <SmartToy sx={{ color: '#8B5CF6' }} />
                </IconButton>

                {/* Notifications */}
                <Badge badgeContent={3} sx={{ '& .MuiBadge-badge': { bgcolor: '#FF4444' } }}>
                    <IconButton sx={{ color: '#8E9BAE' }}>
                        <Notifications />
                    </IconButton>
                </Badge>
            </Stack>
        </Box>
    );
}; 