import React, { useState, useCallback } from 'react';
import {
    Box,
    InputBase,
    Paper,
    IconButton,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Search as SearchIcon,
    Clear as ClearIcon,
} from '@mui/icons-material';

interface SimpleSearchBarProps {
    currentSymbol?: string;
    onSymbolChange?: (symbol: string) => void;
    placeholder?: string;
    width?: number | string;
    maxWidth?: number | string;
    className?: string;
}

export const SimpleSearchBar: React.FC<SimpleSearchBarProps> = ({
    currentSymbol = 'SPY',
    onSymbolChange,
    placeholder = 'Search stocks...',
    width = '100%',
    maxWidth = 600,
    className,
}) => {
    const theme = useTheme();
    const [query, setQuery] = useState('');

    const handleSubmit = useCallback((e: React.FormEvent) => {
        e.preventDefault();
        if (query.trim()) {
            onSymbolChange?.(query.trim().toUpperCase());
            setQuery('');
        }
    }, [query, onSymbolChange]);

    const handleClear = useCallback(() => {
        setQuery('');
    }, []);

    return (
        <Box
            sx={{
                position: 'relative',
                width,
                maxWidth,
            }}
            className={className}
        >
            <Paper
                component="form"
                onSubmit={handleSubmit}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    borderRadius: 3,
                    border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                    backgroundColor: alpha(theme.palette.background.paper, 0.9),
                    backdropFilter: 'blur(10px)',
                    px: 2,
                    py: 1,
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

                <InputBase
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={placeholder}
                    sx={{
                        flex: 1,
                        '& .MuiInputBase-input': {
                            fontSize: '0.95rem',
                            fontWeight: 500,
                            color: theme.palette.text.primary,
                            '&::placeholder': {
                                color: theme.palette.text.secondary,
                                opacity: 0.7,
                            },
                        },
                    }}
                />

                {query && (
                    <IconButton
                        size="small"
                        onClick={handleClear}
                        sx={{
                            color: theme.palette.text.secondary,
                            ml: 1,
                        }}
                    >
                        <ClearIcon fontSize="small" />
                    </IconButton>
                )}
            </Paper>
        </Box>
    );
}; 