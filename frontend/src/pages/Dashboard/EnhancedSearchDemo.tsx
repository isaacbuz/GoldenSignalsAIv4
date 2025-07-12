import React, { useState } from 'react';
import {
    Box,
    Container,
    Typography,
    Paper,
    Grid,
    Card,
    CardContent,
    Stack,
    Chip,
    alpha,
    useTheme,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Divider,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    AutoAwesome,
    Timeline,
    ShowChart,
    Analytics,
    Search,
    PushPin,
    Circle,
} from '@mui/icons-material';
import { useOutletContext } from 'react-router-dom';

interface OutletContext {
    currentSymbol: string;
    onSymbolChange: (symbol: string) => void;
}

const EnhancedSearchDemo: React.FC = () => {
    const theme = useTheme();
    const { currentSymbol, onSymbolChange } = useOutletContext<OutletContext>();
    const [recentSearches] = useState(['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']);

    // Mock data for demonstration
    const features = [
        {
            icon: <Circle />,
            title: 'Market Status Indicator',
            description: 'Real-time market open/closed status with current time',
            color: theme.palette.success.main,
        },
        {
            icon: <Search />,
            title: 'Smart Symbol Search',
            description: 'Search by symbol or company name with autocomplete',
            color: theme.palette.primary.main,
        },
        {
            icon: <PushPin />,
            title: 'Pinned Stocks',
            description: 'Pin up to 6 favorite stocks for quick access',
            color: theme.palette.warning.main,
        },
        {
            icon: <AutoAwesome />,
            title: 'AI Assistant Integration',
            description: 'Ask questions in natural language',
            color: theme.palette.secondary.main,
        },
    ];

    const mockSignals = [
        { symbol: 'AAPL', type: 'BUY', confidence: 87, price: 185.67 },
        { symbol: 'TSLA', type: 'SELL', confidence: 72, price: 245.89 },
        { symbol: 'NVDA', type: 'BUY', confidence: 91, price: 921.45 },
    ];

    return (
        <Container maxWidth="xl" sx={{ py: 4 }}>
            {/* Header */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" gutterBottom fontWeight="bold">
                    Enhanced Search Bar Demo
                </Typography>
                <Typography variant="body1" color="text.secondary">
                    The new search bar is located below the navigation bar and includes market status,
                    pinned stocks, and AI integration for a seamless trading experience.
                </Typography>
            </Box>

            {/* Current Symbol Display */}
            <Paper
                elevation={0}
                sx={{
                    p: 3,
                    mb: 4,
                    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)} 0%, ${alpha(theme.palette.secondary.main, 0.1)} 100%)`,
                    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                }}
            >
                <Stack direction="row" alignItems="center" spacing={2}>
                    <ShowChart sx={{ fontSize: 32, color: theme.palette.primary.main }} />
                    <Box>
                        <Typography variant="overline" color="text.secondary">
                            Currently Viewing
                        </Typography>
                        <Typography variant="h4" fontWeight="bold">
                            {currentSymbol}
                        </Typography>
                    </Box>
                </Stack>
            </Paper>

            {/* Features Grid */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                {features.map((feature, index) => (
                    <Grid item xs={12} sm={6} md={3} key={index}>
                        <Card
                            elevation={0}
                            sx={{
                                height: '100%',
                                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    transform: 'translateY(-4px)',
                                    boxShadow: theme.shadows[8],
                                },
                            }}
                        >
                            <CardContent>
                                <Stack spacing={2}>
                                    <Box
                                        sx={{
                                            width: 48,
                                            height: 48,
                                            borderRadius: 2,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            bgcolor: alpha(feature.color, 0.1),
                                            color: feature.color,
                                        }}
                                    >
                                        {feature.icon}
                                    </Box>
                                    <Box>
                                        <Typography variant="h6" gutterBottom>
                                            {feature.title}
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            {feature.description}
                                        </Typography>
                                    </Box>
                                </Stack>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>

            <Grid container spacing={3}>
                {/* Recent Searches */}
                <Grid item xs={12} md={4}>
                    <Paper elevation={0} sx={{ p: 3, height: '100%', border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                        <Typography variant="h6" gutterBottom>
                            Recent Searches
                        </Typography>
                        <List>
                            {recentSearches.map((symbol, index) => (
                                <React.Fragment key={symbol}>
                                    <ListItem
                                        button
                                        onClick={() => onSymbolChange(symbol)}
                                        sx={{
                                            borderRadius: 1,
                                            '&:hover': {
                                                bgcolor: alpha(theme.palette.primary.main, 0.08),
                                            },
                                        }}
                                    >
                                        <ListItemIcon>
                                            <Timeline sx={{ color: theme.palette.primary.main }} />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={symbol}
                                            secondary="View chart"
                                        />
                                    </ListItem>
                                    {index < recentSearches.length - 1 && <Divider variant="inset" component="li" />}
                                </React.Fragment>
                            ))}
                        </List>
                    </Paper>
                </Grid>

                {/* Active Signals */}
                <Grid item xs={12} md={8}>
                    <Paper elevation={0} sx={{ p: 3, height: '100%', border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                        <Typography variant="h6" gutterBottom>
                            Active AI Signals
                        </Typography>
                        <Stack spacing={2}>
                            {mockSignals.map((signal, index) => (
                                <Box
                                    key={index}
                                    sx={{
                                        p: 2,
                                        borderRadius: 1,
                                        bgcolor: alpha(theme.palette.background.default, 0.5),
                                        border: `1px solid ${alpha(
                                            signal.type === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                                            0.2
                                        )}`,
                                    }}
                                >
                                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                                        <Stack direction="row" alignItems="center" spacing={2}>
                                            {signal.type === 'BUY' ? (
                                                <TrendingUp sx={{ color: theme.palette.success.main }} />
                                            ) : (
                                                <TrendingDown sx={{ color: theme.palette.error.main }} />
                                            )}
                                            <Box>
                                                <Typography variant="subtitle1" fontWeight="bold">
                                                    {signal.symbol}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    ${signal.price}
                                                </Typography>
                                            </Box>
                                        </Stack>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <Chip
                                                label={signal.type}
                                                size="small"
                                                color={signal.type === 'BUY' ? 'success' : 'error'}
                                            />
                                            <Chip
                                                label={`${signal.confidence}% confidence`}
                                                size="small"
                                                variant="outlined"
                                            />
                                        </Stack>
                                    </Stack>
                                </Box>
                            ))}
                        </Stack>
                    </Paper>
                </Grid>
            </Grid>

            {/* Instructions */}
            <Box sx={{ mt: 4, p: 3, bgcolor: alpha(theme.palette.info.main, 0.1), borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom>
                    How to Use the Enhanced Search Bar
                </Typography>
                <List>
                    <ListItem>
                        <ListItemIcon>
                            <Analytics />
                        </ListItemIcon>
                        <ListItemText
                            primary="Search for any stock symbol"
                            secondary="Type in the search bar to find stocks by symbol or company name"
                        />
                    </ListItem>
                    <ListItem>
                        <ListItemIcon>
                            <PushPin />
                        </ListItemIcon>
                        <ListItemText
                            primary="Pin your favorite stocks"
                            secondary="Click the star icon in search results to pin stocks for quick access"
                        />
                    </ListItem>
                    <ListItem>
                        <ListItemIcon>
                            <AutoAwesome />
                        </ListItemIcon>
                        <ListItemText
                            primary="Ask AI questions"
                            secondary="Click the AI button or type natural language queries in the search bar"
                        />
                    </ListItem>
                </List>
            </Box>
        </Container>
    );
};

export default EnhancedSearchDemo; 