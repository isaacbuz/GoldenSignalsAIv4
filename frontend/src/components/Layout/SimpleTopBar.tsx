import React from 'react';
import {
    AppBar,
    Toolbar,
    Box,
    Typography,
    IconButton,
    useTheme,
} from '@mui/material';
import {
    SmartToy,
    ShowChart,
} from '@mui/icons-material';

interface SimpleTopBarProps {
    onAIOpen: () => void;
    currentSymbol: string;
    onSymbolChange: (symbol: string) => void;
}

export const SimpleTopBar: React.FC<SimpleTopBarProps> = ({
    onAIOpen,
    currentSymbol,
    onSymbolChange,
}) => {
    const theme = useTheme();

    return (
        <AppBar
            position="static"
            elevation={0}
            sx={{
                bgcolor: theme.palette.background.paper,
                borderBottom: `1px solid ${theme.palette.divider}`,
                zIndex: 1200,
            }}
        >
            <Toolbar sx={{ px: 3, py: 1, minHeight: '72px !important' }}>
                {/* Brand & Logo */}
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
                    <ShowChart sx={{ color: theme.palette.primary.main, fontSize: 28, mr: 1 }} />
                    <Typography
                        variant="h6"
                        sx={{
                            fontWeight: 700,
                            color: theme.palette.text.primary,
                        }}
                    >
                        GoldenSignals AI
                    </Typography>
                </Box>

                <Box sx={{ flex: 1 }} />

                {/* Current Symbol */}
                <Typography variant="h6" sx={{ color: theme.palette.text.primary, mr: 2 }}>
                    {currentSymbol}
                </Typography>

                {/* AI Assistant */}
                <IconButton
                    onClick={onAIOpen}
                    sx={{
                        bgcolor: theme.palette.primary.main,
                        color: 'white',
                        '&:hover': {
                            bgcolor: theme.palette.primary.dark,
                        }
                    }}
                >
                    <SmartToy />
                </IconButton>
            </Toolbar>
        </AppBar>
    );
};
