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
                elevation={0}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    borderRadius: '12px',
                    border: `1px solid ${alpha('#FFD700', 0.15)}`,
                    backgroundColor: alpha(theme.palette.background.paper, 0.6),
                    backdropFilter: 'blur(16px)',
                    px: 2.5,
                    py: 1.25,
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    background: `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.8)} 0%, ${alpha(theme.palette.background.paper, 0.6)} 100%)`,
                    '&:hover': {
                        borderColor: alpha(theme.palette.primary.main, 0.4),
                        boxShadow: `0 4px 16px ${alpha(theme.palette.primary.main, 0.15)}`,
                        transform: 'translateY(-1px)',
                    },
                    '&:focus-within': {
                        borderColor: theme.palette.primary.main,
                        boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.25)}, inset 0 0 20px ${alpha(theme.palette.primary.main, 0.05)}`,
                        transform: 'translateY(-1px)',
                    },
                }}
            >
                <SearchIcon
                    sx={{
                        color: alpha(theme.palette.primary.main, 0.6),
                        mr: 1.5,
                        fontSize: 22,
                        transition: 'color 0.3s',
                    }}
                />

                <InputBase
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={placeholder}
                    sx={{
                        flex: 1,
                        '& .MuiInputBase-input': {
                            fontSize: '0.925rem',
                            fontWeight: 500,
                            letterSpacing: '0.01em',
                            color: theme.palette.text.primary,
                            '&::placeholder': {
                                color: alpha(theme.palette.text.secondary, 0.6),
                                opacity: 1,
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
