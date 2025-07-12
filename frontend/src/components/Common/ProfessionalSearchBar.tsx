import React, { useState, useCallback, useMemo, useRef } from 'react';
import {
    Box,
    TextField,
    Autocomplete,
    Paper,
    Typography,
    Chip,
    IconButton,
    InputAdornment,
    Fade,
    Divider,
} from '@mui/material';
import {
    Search as SearchIcon,
    Clear as ClearIcon,
    TrendingUp as TrendingUpIcon,
    Analytics as AnalyticsIcon,
    Psychology as PsychologyIcon,
    Settings as SettingsIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Professional styled components
const SearchContainer = styled(Box)(({ theme }) => ({
    position: 'relative',
    width: '100%',
    maxWidth: 600,
    margin: '0 auto',
}));

const SearchField = styled(TextField)(({ theme }) => ({
    '& .MuiOutlinedInput-root': {
        backgroundColor: theme.palette.background.paper,
        borderRadius: 8,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
            backgroundColor: theme.palette.action?.hover,
        },
        '&.Mui-focused': {
            backgroundColor: theme.palette.background.paper,
            '& fieldset': {
                borderColor: theme.palette.primary.main,
                borderWidth: 2,
            },
        },
    },
    '& .MuiInputBase-input': {
        padding: '12px 16px',
        fontSize: '1rem',
        fontWeight: 500,
    },
}));

const SuggestionPaper = styled(Paper)(({ theme }) => ({
    backgroundColor: theme.palette.background.paper,
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: 8,
    marginTop: 4,
    maxHeight: 400,
    overflow: 'auto',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
}));

const SuggestionItem = styled(Box)(({ theme }) => ({
    padding: '12px 16px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    transition: 'all 0.2s ease-in-out',
    '&:hover': {
        backgroundColor: theme.palette.action?.hover,
    },
    '&:not(:last-child)': {
        borderBottom: `1px solid ${theme.palette.divider}`,
    },
}));

const CategoryChip = styled(Chip)(({ theme }) => ({
    height: 24,
    fontSize: '0.75rem',
    fontWeight: 500,
    '& .MuiChip-label': {
        padding: '0 8px',
    },
}));

// Static data to prevent re-creation on every render
const SEARCH_CATEGORIES = [
    { id: 'symbols', label: 'Symbols', icon: TrendingUpIcon, color: 'primary' },
    { id: 'signals', label: 'Signals', icon: AnalyticsIcon, color: 'success' },
    { id: 'ai', label: 'AI Insights', icon: PsychologyIcon, color: 'warning' },
    { id: 'settings', label: 'Settings', icon: SettingsIcon, color: 'secondary' },
] as const;

const POPULAR_SYMBOLS = [
    'SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'QQQ', 'IWM'
] as const;

const QUICK_ACTIONS = [
    { id: 'market-overview', label: 'Market Overview', category: 'signals' },
    { id: 'top-signals', label: 'Top Signals', category: 'signals' },
    { id: 'ai-analysis', label: 'AI Analysis', category: 'ai' },
    { id: 'portfolio', label: 'Portfolio', category: 'settings' },
] as const;

interface SearchSuggestion {
    id: string;
    label: string;
    category: string;
    description?: string;
    icon?: React.ComponentType;
}

interface ProfessionalSearchBarProps {
    onSearch?: (query: string) => void;
    onSelect?: (suggestion: SearchSuggestion) => void;
    placeholder?: string;
    disabled?: boolean;
}

export const ProfessionalSearchBar: React.FC<ProfessionalSearchBarProps> = ({
    onSearch,
    onSelect,
    placeholder = "Search symbols, signals, or ask AI...",
    disabled = false,
}) => {
    const [query, setQuery] = useState('');
    const [isOpen, setIsOpen] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const inputRef = useRef<HTMLInputElement>(null);

    // Memoized suggestions to prevent re-computation
    const suggestions = useMemo(() => {
        if (!query.trim()) {
            return [
                ...QUICK_ACTIONS.map(action => ({
                    id: action.id,
                    label: action.label,
                    category: action.category,
                    description: `Quick access to ${action.label.toLowerCase()}`,
                })),
                ...POPULAR_SYMBOLS.map(symbol => ({
                    id: symbol,
                    label: symbol,
                    category: 'symbols',
                    description: `View ${symbol} analysis and signals`,
                })),
            ];
        }

        const filtered: SearchSuggestion[] = [];
        const lowerQuery = query.toLowerCase();

        // Search symbols
        POPULAR_SYMBOLS.forEach(symbol => {
            if (symbol.toLowerCase().includes(lowerQuery)) {
                filtered.push({
                    id: symbol,
                    label: symbol,
                    category: 'symbols',
                    description: `View ${symbol} analysis and signals`,
                });
            }
        });

        // Search quick actions
        QUICK_ACTIONS.forEach(action => {
            if (action.label.toLowerCase().includes(lowerQuery)) {
                filtered.push({
                    id: action.id,
                    label: action.label,
                    category: action.category,
                    description: `Quick access to ${action.label.toLowerCase()}`,
                });
            }
        });

        // AI-related searches
        if (lowerQuery.includes('ai') || lowerQuery.includes('analysis')) {
            filtered.push({
                id: 'ai-chat',
                label: 'Golden Eye AI Prophet',
                category: 'ai',
                description: 'Chat with AI for market insights',
            });
        }

        return filtered.slice(0, 8); // Limit results
    }, [query]);

    // Stable callback functions
    const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        setQuery(value);
        setSelectedIndex(-1);
        setIsOpen(value.length > 0 || true); // Keep open for quick actions
    }, []);

    const handleClear = useCallback(() => {
        setQuery('');
        setSelectedIndex(-1);
        setIsOpen(false);
        inputRef.current?.focus();
    }, []);

    const handleSuggestionClick = useCallback((suggestion: SearchSuggestion) => {
        setQuery(suggestion.label);
        setIsOpen(false);
        onSelect?.(suggestion);
    }, [onSelect]);

    const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
        if (!isOpen) return;

        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                setSelectedIndex(prev => Math.min(prev + 1, suggestions.length - 1));
                break;
            case 'ArrowUp':
                event.preventDefault();
                setSelectedIndex(prev => Math.max(prev - 1, -1));
                break;
            case 'Enter':
                event.preventDefault();
                if (selectedIndex >= 0) {
                    handleSuggestionClick(suggestions[selectedIndex]);
                } else if (query.trim()) {
                    onSearch?.(query);
                    setIsOpen(false);
                }
                break;
            case 'Escape':
                setIsOpen(false);
                setSelectedIndex(-1);
                break;
        }
    }, [isOpen, selectedIndex, suggestions, query, onSearch, handleSuggestionClick]);

    const handleFocus = useCallback(() => {
        setIsOpen(true);
    }, []);

    const handleBlur = useCallback(() => {
        // Delay closing to allow clicking on suggestions
        setTimeout(() => setIsOpen(false), 150);
    }, []);

    const getCategoryIcon = useCallback((category: string) => {
        const categoryConfig = SEARCH_CATEGORIES.find(cat => cat.id === category);
        return categoryConfig?.icon || SearchIcon;
    }, []);

    const getCategoryColor = useCallback((category: string) => {
        const categoryConfig = SEARCH_CATEGORIES.find(cat => cat.id === category);
        return categoryConfig?.color || 'default';
    }, []);

    return (
        <SearchContainer>
            <SearchField
                ref={inputRef}
                fullWidth
                value={query}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                onFocus={handleFocus}
                onBlur={handleBlur}
                placeholder={placeholder}
                disabled={disabled}
                InputProps={{
                    startAdornment: (
                        <InputAdornment position="start">
                            <SearchIcon sx={{ color: 'text.secondary' }} />
                        </InputAdornment>
                    ),
                    endAdornment: query && (
                        <InputAdornment position="end">
                            <IconButton
                                size="small"
                                onClick={handleClear}
                                sx={{ color: 'text.secondary' }}
                            >
                                <ClearIcon />
                            </IconButton>
                        </InputAdornment>
                    ),
                }}
            />

            <Fade in={isOpen} timeout={200}>
                <SuggestionPaper>
                    {suggestions.length > 0 ? (
                        suggestions.map((suggestion, index) => {
                            const IconComponent = getCategoryIcon(suggestion.category);
                            const isSelected = index === selectedIndex;

                            return (
                                <SuggestionItem
                                    key={suggestion.id}
                                    onClick={() => handleSuggestionClick(suggestion)}
                                    sx={{
                                        backgroundColor: isSelected ? 'action.selected' : 'transparent',
                                    }}
                                >
                                    <IconComponent sx={{ fontSize: 20, color: 'text.secondary' }} />
                                    <Box sx={{ flex: 1, minWidth: 0 }}>
                                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                            {suggestion.label}
                                        </Typography>
                                        {suggestion.description && (
                                            <Typography variant="caption" color="text.secondary">
                                                {suggestion.description}
                                            </Typography>
                                        )}
                                    </Box>
                                    <CategoryChip
                                        label={suggestion.category}
                                        size="small"
                                        color={getCategoryColor(suggestion.category) as any}
                                        variant="outlined"
                                    />
                                </SuggestionItem>
                            );
                        })
                    ) : (
                        <SuggestionItem>
                            <SearchIcon sx={{ fontSize: 20, color: 'text.secondary' }} />
                            <Typography variant="body2" color="text.secondary">
                                No results found for "{query}"
                            </Typography>
                        </SuggestionItem>
                    )}
                </SuggestionPaper>
            </Fade>
        </SearchContainer>
    );
};

export default ProfessionalSearchBar; 