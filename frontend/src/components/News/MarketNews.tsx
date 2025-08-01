import React from 'react';
import {
    Box,
    Typography,
    Stack,
    useTheme,
    alpha,
    Chip,
    IconButton,
} from '@mui/material';
import {
    Article,
    TrendingUp,
    TrendingDown,
    Warning,
    Refresh,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

interface NewsItem {
    id: string;
    title: string;
    source: string;
    timestamp: string;
    impact: 'HIGH' | 'MEDIUM' | 'LOW';
    sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    url?: string;
}

interface MarketNewsProps {
    symbol?: string;
    compact?: boolean;
    maxItems?: number;
}

const MarketNews: React.FC<MarketNewsProps> = ({
    symbol = 'SPY',
    compact = true,
    maxItems = 3
}) => {
    const theme = useTheme();

    // Mock news data - in real app this would come from API
    const mockNews: NewsItem[] = [
        {
            id: '1',
            title: 'Fed Maintains Interest Rates, Signals Cautious Approach',
            source: 'Reuters',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            impact: 'HIGH',
            sentiment: 'NEUTRAL',
        },
        {
            id: '2',
            title: 'Tech Stocks Rally on Strong Earnings Reports',
            source: 'CNBC',
            timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
            impact: 'MEDIUM',
            sentiment: 'BULLISH',
        },
        {
            id: '3',
            title: 'Consumer Confidence Index Beats Expectations',
            source: 'Bloomberg',
            timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
            impact: 'MEDIUM',
            sentiment: 'BULLISH',
        },
        {
            id: '4',
            title: 'Oil Prices Rise Amid Supply Chain Concerns',
            source: 'MarketWatch',
            timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
            impact: 'LOW',
            sentiment: 'NEUTRAL',
        },
    ];

    const getImpactColor = (impact: string) => {
        switch (impact) {
            case 'HIGH': return theme.palette.error.main;
            case 'MEDIUM': return theme.palette.warning.main;
            case 'LOW': return theme.palette.info.main;
            default: return theme.palette.text.secondary;
        }
    };

    const getSentimentIcon = (sentiment: string) => {
        switch (sentiment) {
            case 'BULLISH': return <TrendingUp sx={{ fontSize: 12, color: theme.palette.success.main }} />;
            case 'BEARISH': return <TrendingDown sx={{ fontSize: 12, color: theme.palette.error.main }} />;
            default: return <Warning sx={{ fontSize: 12, color: theme.palette.warning.main }} />;
        }
    };

    const formatTimeAgo = (timestamp: string) => {
        const now = new Date();
        const time = new Date(timestamp);
        const diffInHours = Math.floor((now.getTime() - time.getTime()) / (1000 * 60 * 60));

        if (diffInHours < 1) {
            const diffInMinutes = Math.floor((now.getTime() - time.getTime()) / (1000 * 60));
            return `${diffInMinutes}m ago`;
        } else if (diffInHours < 24) {
            return `${diffInHours}h ago`;
        } else {
            const diffInDays = Math.floor(diffInHours / 24);
            return `${diffInDays}d ago`;
        }
    };

    return (
        <Box
            sx={{
                p: 1.5,
                mb: 1.5,
                bgcolor: alpha(theme.palette.background.paper, 0.5),
                backdropFilter: 'blur(10px)',
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            }}
        >
            <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography
                    variant="h6"
                    sx={{
                        fontSize: '0.875rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.5
                    }}
                >
                    <Article sx={{ fontSize: 16, color: theme.palette.primary.main }} />
                    Market News
                </Typography>
                <IconButton size="small" sx={{ padding: 0.25 }}>
                    <Refresh sx={{ fontSize: 14 }} />
                </IconButton>
            </Stack>

            <Stack spacing={1}>
                {mockNews.slice(0, maxItems).map((news) => (
                    <Box
                        key={news.id}
                        sx={{
                            p: 1,
                            borderRadius: 1,
                            bgcolor: alpha(theme.palette.background.default, 0.3),
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': {
                                bgcolor: alpha(theme.palette.primary.main, 0.05),
                                transform: 'translateX(2px)',
                            },
                        }}
                    >
                        <Stack spacing={0.5}>
                            <Typography
                                variant="body2"
                                sx={{
                                    fontSize: '0.7rem',
                                    fontWeight: 500,
                                    lineHeight: 1.2,
                                    display: '-webkit-box',
                                    WebkitLineClamp: 2,
                                    WebkitBoxOrient: 'vertical',
                                    overflow: 'hidden',
                                }}
                            >
                                {news.title}
                            </Typography>

                            <Stack direction="row" alignItems="center" justifyContent="space-between">
                                <Stack direction="row" alignItems="center" spacing={0.5}>
                                    <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{ fontSize: '0.625rem' }}
                                    >
                                        {news.source}
                                    </Typography>
                                    <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{ fontSize: '0.625rem' }}
                                    >
                                        • {formatTimeAgo(news.timestamp)}
                                    </Typography>
                                </Stack>

                                <Stack direction="row" alignItems="center" spacing={0.5}>
                                    {getSentimentIcon(news.sentiment)}
                                    <Chip
                                        label={news.impact}
                                        size="small"
                                        sx={{
                                            fontSize: '0.5rem',
                                            height: 16,
                                            bgcolor: alpha(getImpactColor(news.impact), 0.1),
                                            color: getImpactColor(news.impact),
                                            border: `1px solid ${alpha(getImpactColor(news.impact), 0.3)}`,
                                        }}
                                    />
                                </Stack>
                            </Stack>
                        </Stack>
                    </Box>
                ))}
            </Stack>
        </Box>
    );
};

export default MarketNews;
