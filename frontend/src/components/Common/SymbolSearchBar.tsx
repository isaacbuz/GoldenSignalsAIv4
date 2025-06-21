import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Paper,
    InputBase,
    IconButton,
    Tooltip,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Typography,
    alpha,
    useTheme,
    Popper,
    ClickAwayListener,
    Chip,
    Stack,
} from '@mui/material';
import {
    Search as SearchIcon,
    TrendingUp,
    Star as StarIcon,
    StarBorder as StarBorderIcon,
    ShowChart,
    Timeline,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface SymbolSearchBarProps {
    onSymbolChange?: (symbol: string) => void;
    currentSymbol?: string;
    onAddToFavorites?: (symbol: string) => void;
    favorites?: string[];
}

interface SymbolSuggestion {
    symbol: string;
    name?: string;
    type?: string;
    isFavorite?: boolean;
}

export const SymbolSearchBar: React.FC<SymbolSearchBarProps> = ({
    onSymbolChange,
    currentSymbol = 'SPY',
    onAddToFavorites,
    favorites = [],
}) => {
    const theme = useTheme();
    const navigate = useNavigate();
    const [query, setQuery] = useState('');
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [suggestions, setSuggestions] = useState<SymbolSuggestion[]>([]);

    const inputRef = useRef<HTMLInputElement>(null);
    const anchorEl = useRef<HTMLDivElement>(null);

    // Popular stock symbols with categories
    const stockData: SymbolSuggestion[] = [
        // Major Indices & ETFs
        { symbol: 'SPY', name: 'SPDR S&P 500 ETF', type: 'ETF' },
        { symbol: 'QQQ', name: 'Invesco QQQ Trust', type: 'ETF' },
        { symbol: 'DIA', name: 'SPDR Dow Jones', type: 'ETF' },
        { symbol: 'IWM', name: 'iShares Russell 2000', type: 'ETF' },
        { symbol: 'VTI', name: 'Vanguard Total Stock', type: 'ETF' },

        // Tech Giants
        { symbol: 'AAPL', name: 'Apple Inc.', type: 'Stock' },
        { symbol: 'MSFT', name: 'Microsoft Corporation', type: 'Stock' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', type: 'Stock' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.', type: 'Stock' },
        { symbol: 'TSLA', name: 'Tesla Inc.', type: 'Stock' },
        { symbol: 'META', name: 'Meta Platforms', type: 'Stock' },
        { symbol: 'NVDA', name: 'NVIDIA Corporation', type: 'Stock' },

        // More Tech
        { symbol: 'AMD', name: 'Advanced Micro Devices', type: 'Stock' },
        { symbol: 'NFLX', name: 'Netflix Inc.', type: 'Stock' },
        { symbol: 'PYPL', name: 'PayPal Holdings', type: 'Stock' },
        { symbol: 'ADBE', name: 'Adobe Inc.', type: 'Stock' },
        { symbol: 'CRM', name: 'Salesforce Inc.', type: 'Stock' },
        { symbol: 'INTC', name: 'Intel Corporation', type: 'Stock' },
        { symbol: 'ORCL', name: 'Oracle Corporation', type: 'Stock' },
        { symbol: 'CSCO', name: 'Cisco Systems', type: 'Stock' },

        // Blue Chips
        { symbol: 'BRK.B', name: 'Berkshire Hathaway', type: 'Stock' },
        { symbol: 'JPM', name: 'JPMorgan Chase', type: 'Stock' },
        { symbol: 'V', name: 'Visa Inc.', type: 'Stock' },
        { symbol: 'JNJ', name: 'Johnson & Johnson', type: 'Stock' },
        { symbol: 'WMT', name: 'Walmart Inc.', type: 'Stock' },
        { symbol: 'PG', name: 'Procter & Gamble', type: 'Stock' },
        { symbol: 'MA', name: 'Mastercard Inc.', type: 'Stock' },
        { symbol: 'UNH', name: 'UnitedHealth Group', type: 'Stock' },
    ];

    // Generate suggestions based on query
    useEffect(() => {
        if (!query.trim()) {
            // Show favorites when no query
            const favoriteSuggestions = stockData
                .filter(stock => favorites.includes(stock.symbol))
                .map(stock => ({ ...stock, isFavorite: true }))
                .slice(0, 5);
            setSuggestions(favoriteSuggestions);
            return;
        }

        const q = query.toUpperCase();
        const filtered = stockData
            .filter(stock =>
                stock.symbol.includes(q) ||
                stock.name?.toUpperCase().includes(q)
            )
            .map(stock => ({
                ...stock,
                isFavorite: favorites.includes(stock.symbol)
            }))
            .sort((a, b) => {
                // Prioritize exact matches
                if (a.symbol === q) return -1;
                if (b.symbol === q) return 1;
                // Then favorites
                if (a.isFavorite && !b.isFavorite) return -1;
                if (!a.isFavorite && b.isFavorite) return 1;
                // Then by symbol length
                return a.symbol.length - b.symbol.length;
            })
            .slice(0, 8);

        setSuggestions(filtered);
    }, [query, favorites]);

    const handleSubmit = () => {
        const trimmedQuery = query.trim().toUpperCase();
        if (!trimmedQuery) return;

        // Check if it's a valid symbol from our list or just pass it through
        onSymbolChange?.(trimmedQuery);
        setQuery('');
        setShowSuggestions(false);
        inputRef.current?.blur();
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSubmit();
        } else if (e.key === 'Escape') {
            setShowSuggestions(false);
            inputRef.current?.blur();
        }
    };

    const handleSuggestionClick = (symbol: string) => {
        onSymbolChange?.(symbol);
        setQuery('');
        setShowSuggestions(false);
        inputRef.current?.blur();
    };

    const handleFavoriteToggle = (e: React.MouseEvent, symbol: string) => {
        e.stopPropagation();
        onAddToFavorites?.(symbol);
    };

    return (
        <Box
            ref={anchorEl}
            sx={{
                position: 'relative',
                width: '100%',
                maxWidth: 400,
            }}
        >
            <Paper
                elevation={0}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 2,
                    py: 1,
                    borderRadius: 3,
                    border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                    backgroundColor: alpha(theme.palette.background.paper, 0.9),
                    backdropFilter: 'blur(10px)',
                    transition: 'all 0.2s',
                    '&:hover': {
                        borderColor: alpha(theme.palette.primary.main, 0.3),
                        boxShadow: `0 2px 8px ${alpha(theme.palette.primary.main, 0.1)}`,
                    },
                    '&:focus-within': {
                        borderColor: theme.palette.primary.main,
                        boxShadow: `0 2px 12px ${alpha(theme.palette.primary.main, 0.2)}`,
                    },
                }}
            >
                <SearchIcon
                    sx={{
                        color: theme.palette.text.secondary,
                        mr: 1.5,
                        fontSize: 20,
                    }}
                />

                {/* Current Symbol Chip */}
                {currentSymbol && (
                    <Chip
                        label={currentSymbol}
                        size="small"
                        icon={<ShowChart />}
                        sx={{
                            mr: 1.5,
                            height: 24,
                            backgroundColor: alpha(theme.palette.primary.main, 0.1),
                            color: theme.palette.primary.main,
                            fontWeight: 600,
                            '& .MuiChip-icon': {
                                fontSize: 16,
                                color: theme.palette.primary.main,
                            },
                        }}
                    />
                )}

                <InputBase
                    ref={inputRef}
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value.toUpperCase());
                        setShowSuggestions(true);
                    }}
                    onFocus={() => setShowSuggestions(true)}
                    onKeyDown={handleKeyPress}
                    placeholder="Search symbols..."
                    sx={{
                        flex: 1,
                        fontSize: '0.875rem',
                        fontWeight: 500,
                        '& input': {
                            padding: '4px 0',
                            '&::placeholder': {
                                opacity: 0.6,
                            },
                        },
                    }}
                />

                {query && (
                    <Tooltip title="Search">
                        <IconButton
                            size="small"
                            onClick={handleSubmit}
                            sx={{ ml: 1 }}
                        >
                            <SearchIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                )}
            </Paper>

            {/* Suggestions Dropdown */}
            <Popper
                open={showSuggestions && (suggestions.length > 0 || !query)}
                anchorEl={anchorEl.current}
                placement="bottom-start"
                style={{ width: anchorEl.current?.offsetWidth, zIndex: 1300 }}
            >
                <ClickAwayListener onClickAway={() => setShowSuggestions(false)}>
                    <Paper
                        elevation={4}
                        sx={{
                            mt: 1,
                            maxHeight: 320,
                            overflow: 'auto',
                            borderRadius: 2,
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                        }}
                    >
                        {!query && suggestions.length > 0 && (
                            <Box sx={{ px: 2, py: 1, borderBottom: `1px solid ${theme.palette.divider}` }}>
                                <Typography variant="caption" color="text.secondary" fontWeight={600}>
                                    FAVORITES
                                </Typography>
                            </Box>
                        )}

                        <List dense sx={{ py: 0.5 }}>
                            {suggestions.map((suggestion, index) => (
                                <ListItem
                                    key={index}
                                    button
                                    onClick={() => handleSuggestionClick(suggestion.symbol)}
                                    sx={{
                                        py: 1,
                                        px: 2,
                                        '&:hover': {
                                            backgroundColor: alpha(theme.palette.primary.main, 0.08),
                                        },
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                        <Timeline sx={{ fontSize: 20, color: theme.palette.primary.main }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={
                                            <Stack direction="row" alignItems="center" spacing={1}>
                                                <Typography
                                                    variant="body2"
                                                    fontWeight={600}
                                                    sx={{ color: theme.palette.text.primary }}
                                                >
                                                    {suggestion.symbol}
                                                </Typography>
                                                {suggestion.type && (
                                                    <Chip
                                                        label={suggestion.type}
                                                        size="small"
                                                        sx={{
                                                            height: 18,
                                                            fontSize: '0.7rem',
                                                            backgroundColor: alpha(theme.palette.info.main, 0.1),
                                                            color: theme.palette.info.main,
                                                        }}
                                                    />
                                                )}
                                            </Stack>
                                        }
                                        secondary={suggestion.name}
                                        secondaryTypographyProps={{
                                            fontSize: '0.75rem',
                                            color: 'text.secondary',
                                        }}
                                    />
                                    <IconButton
                                        size="small"
                                        edge="end"
                                        onClick={(e) => handleFavoriteToggle(e, suggestion.symbol)}
                                        sx={{
                                            color: suggestion.isFavorite
                                                ? theme.palette.warning.main
                                                : theme.palette.action.disabled,
                                        }}
                                    >
                                        {suggestion.isFavorite ? (
                                            <StarIcon fontSize="small" />
                                        ) : (
                                            <StarBorderIcon fontSize="small" />
                                        )}
                                    </IconButton>
                                </ListItem>
                            ))}
                        </List>

                        {query && suggestions.length === 0 && (
                            <Box sx={{ p: 2, textAlign: 'center' }}>
                                <Typography variant="body2" color="text.secondary">
                                    No symbols found. Press Enter to search anyway.
                                </Typography>
                            </Box>
                        )}
                    </Paper>
                </ClickAwayListener>
            </Popper>
        </Box>
    );
}; 