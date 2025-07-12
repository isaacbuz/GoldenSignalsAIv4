import React, { useState, useCallback, useMemo } from 'react';
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
} from '@mui/material';
import {
    Search as SearchIcon,
    Clear as ClearIcon,
    TrendingUp as TrendingUpIcon,
    Analytics as AnalyticsIcon,
    Psychology as PsychologyIcon,
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
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        borderRadius: '12px',
        border: '1px solid rgba(255, 215, 0, 0.3)',
        color: '#E2E8F0',
        '&:hover': {
            borderColor: 'rgba(255, 215, 0, 0.5)',
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
        },
        '&.Mui-focused': {
            borderColor: '#FFD700',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            boxShadow: '0 0 0 2px rgba(255, 215, 0, 0.2)',
        },
    },
    '& .MuiOutlinedInput-input': {
        color: '#E2E8F0',
        fontSize: '16px',
        '&::placeholder': {
            color: '#94A3B8',
            opacity: 1,
        },
    },
}));

const SearchPaper = styled(Paper)(({ theme }) => ({
    backgroundColor: '#1E293B',
    border: '1px solid rgba(255, 215, 0, 0.3)',
    borderRadius: '12px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
    maxHeight: 400,
    overflow: 'auto',
}));

const SuggestionItem = styled(Box)(({ theme }) => ({
    padding: '12px 16px',
    cursor: 'pointer',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    '&:hover': {
        backgroundColor: 'rgba(255, 215, 0, 0.1)',
    },
    '&:last-child': {
        borderBottom: 'none',
    },
}));

// Static data to prevent re-creation
const SEARCH_CATEGORIES = [
    { id: 'symbols', label: 'Symbols', icon: <TrendingUpIcon /> },
    { id: 'agents', label: 'Agents', icon: <PsychologyIcon /> },
    { id: 'signals', label: 'Signals', icon: <AnalyticsIcon /> },
];

const SAMPLE_SUGGESTIONS = [
    { id: '1', type: 'symbol', label: 'AAPL', description: 'Apple Inc.' },
    { id: '2', type: 'symbol', label: 'GOOGL', description: 'Alphabet Inc.' },
    { id: '3', type: 'symbol', label: 'MSFT', description: 'Microsoft Corp.' },
    { id: '4', type: 'symbol', label: 'TSLA', description: 'Tesla Inc.' },
    { id: '5', type: 'symbol', label: 'AMZN', description: 'Amazon.com Inc.' },
    { id: '6', type: 'agent', label: 'Sentiment Prophet', description: 'AI Sentiment Analysis' },
    { id: '7', type: 'agent', label: 'Technical Wizard', description: 'Technical Analysis AI' },
    { id: '8', type: 'agent', label: 'Risk Guardian', description: 'Risk Management AI' },
    { id: '9', type: 'signal', label: 'Bullish Momentum', description: 'Strong buy signals' },
    { id: '10', type: 'signal', label: 'Bearish Reversal', description: 'Potential sell signals' },
];

interface UnifiedSearchBarProps {
    onSearch?: (query: string) => void;
    onSelect?: (item: any) => void;
    placeholder?: string;
    className?: string;
}

const UnifiedSearchBar: React.FC<UnifiedSearchBarProps> = ({
    onSearch,
    onSelect,
    placeholder = "Search symbols, agents, or signals...",
    className,
}) => {
    const [searchValue, setSearchValue] = useState('');
    const [isOpen, setIsOpen] = useState(false);

    // Memoized filtered suggestions to prevent re-computation
    const filteredSuggestions = useMemo(() => {
        if (!searchValue.trim()) {
            return SAMPLE_SUGGESTIONS.slice(0, 8);
        }

        return SAMPLE_SUGGESTIONS.filter(item =>
            item.label.toLowerCase().includes(searchValue.toLowerCase()) ||
            item.description.toLowerCase().includes(searchValue.toLowerCase())
        ).slice(0, 10);
    }, [searchValue]);

    // Stable callbacks to prevent re-renders
    const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        setSearchValue(value);
        setIsOpen(value.length > 0);
    }, []);

    const handleClear = useCallback(() => {
        setSearchValue('');
        setIsOpen(false);
    }, []);

    const handleSelect = useCallback((item: any) => {
        setSearchValue(item.label);
        setIsOpen(false);
        onSelect?.(item);
    }, [onSelect]);

    const handleSearch = useCallback(() => {
        if (searchValue.trim()) {
            onSearch?.(searchValue);
            setIsOpen(false);
        }
    }, [searchValue, onSearch]);

    const handleKeyPress = useCallback((event: React.KeyboardEvent) => {
        if (event.key === 'Enter') {
            handleSearch();
        } else if (event.key === 'Escape') {
            setIsOpen(false);
        }
    }, [handleSearch]);

    const getTypeIcon = useCallback((type: string) => {
        switch (type) {
            case 'symbol': return <TrendingUpIcon sx={{ fontSize: 16, color: '#FFD700' }} />;
            case 'agent': return <PsychologyIcon sx={{ fontSize: 16, color: '#00D4AA' }} />;
            case 'signal': return <AnalyticsIcon sx={{ fontSize: 16, color: '#2196F3' }} />;
            default: return <SearchIcon sx={{ fontSize: 16, color: '#94A3B8' }} />;
        }
    }, []);

    const getTypeColor = useCallback((type: string) => {
        switch (type) {
            case 'symbol': return '#FFD700';
            case 'agent': return '#00D4AA';
            case 'signal': return '#2196F3';
            default: return '#94A3B8';
        }
    }, []);

    return (
        <SearchContainer className={className}>
            <SearchField
                fullWidth
                value={searchValue}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder={placeholder}
                variant="outlined"
                InputProps={{
                    startAdornment: (
                        <InputAdornment position="start">
                            <SearchIcon sx={{ color: '#FFD700' }} />
                        </InputAdornment>
                    ),
                    endAdornment: searchValue && (
                        <InputAdornment position="end">
                            <IconButton
                                size="small"
                                onClick={handleClear}
                                sx={{ color: '#94A3B8' }}
                            >
                                <ClearIcon />
                            </IconButton>
                        </InputAdornment>
                    ),
                }}
            />

            {isOpen && (
                <Fade in={isOpen}>
                    <SearchPaper
                        sx={{
                            position: 'absolute',
                            top: '100%',
                            left: 0,
                            right: 0,
                            zIndex: 1000,
                            mt: 1,
                        }}
                    >
                        {filteredSuggestions.length > 0 ? (
                            <>
                                <Box sx={{ p: 2, borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                                    <Typography variant="body2" sx={{ color: '#94A3B8', fontWeight: 'bold' }}>
                                        Search Results
                                    </Typography>
                                </Box>
                                {filteredSuggestions.map((item) => (
                                    <SuggestionItem
                                        key={item.id}
                                        onClick={() => handleSelect(item)}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            {getTypeIcon(item.type)}
                                            <Box sx={{ flex: 1 }}>
                                                <Typography variant="body2" sx={{ color: '#E2E8F0', fontWeight: 'bold' }}>
                                                    {item.label}
                                                </Typography>
                                                <Typography variant="caption" sx={{ color: '#94A3B8' }}>
                                                    {item.description}
                                                </Typography>
                                            </Box>
                                            <Chip
                                                label={item.type}
                                                size="small"
                                                sx={{
                                                    backgroundColor: `${getTypeColor(item.type)}20`,
                                                    color: getTypeColor(item.type),
                                                    border: `1px solid ${getTypeColor(item.type)}`,
                                                    fontSize: '0.7rem',
                                                }}
                                            />
                                        </Box>
                                    </SuggestionItem>
                                ))}
                            </>
                        ) : (
                            <Box sx={{ p: 3, textAlign: 'center' }}>
                                <Typography variant="body2" sx={{ color: '#94A3B8' }}>
                                    No results found for "{searchValue}"
                                </Typography>
                            </Box>
                        )}
                    </SearchPaper>
                </Fade>
            )}
        </SearchContainer>
    );
};

export default UnifiedSearchBar; 