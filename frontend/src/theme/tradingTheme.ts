import { createTheme } from '@mui/material/styles';

// Clean Trading Platform Color Palette
const tradingColors = {
    // Market Colors
    bullish: '#00C851',     // Green for upward movement
    bearish: '#FF4444',     // Red for downward movement
    neutral: '#6C757D',     // Gray for neutral

    // Background Colors
    primary: '#0D1421',     // Deep dark blue (main background)
    secondary: '#1A1F2E',   // Slightly lighter (panels)
    tertiary: '#252A3A',    // Card backgrounds

    // Accent Colors
    accent: '#2E86AB',      // Blue accent
    warning: '#FFB74D',     // Orange for warnings
    success: '#4CAF50',     // Success green
    error: '#F44336',       // Error red

    // Text Colors
    text: {
        primary: '#FFFFFF',
        secondary: '#B8BCC8',
        muted: '#6C757D',
        inverse: '#0D1421',
    },

    // Border Colors
    border: '#2C3E50',
    divider: '#34495E',

    // AI Colors
    ai: {
        primary: '#8B5CF6',   // Purple for AI elements
        secondary: '#A78BFA', // Light purple
        accent: '#C4B5FD',    // Very light purple
    }
};

export const tradingTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: tradingColors.accent,
            dark: '#1E5A8A',
            light: '#60A5FA',
            contrastText: tradingColors.text.primary,
        },
        secondary: {
            main: tradingColors.ai.primary,
            dark: tradingColors.ai.secondary,
            light: tradingColors.ai.accent,
            contrastText: tradingColors.text.primary,
        },
        background: {
            default: tradingColors.primary,
            paper: tradingColors.secondary,
        },
        text: {
            primary: tradingColors.text.primary,
            secondary: tradingColors.text.secondary,
        },
        success: {
            main: tradingColors.bullish,
            dark: '#00A73D',
            light: '#5DCEA5',
        },
        error: {
            main: tradingColors.bearish,
            dark: '#CC2936',
            light: '#FF7F7F',
        },
        warning: {
            main: tradingColors.warning,
            dark: '#F57C00',
            light: '#FFD54F',
        },
        divider: tradingColors.divider,
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica Neue", Arial, sans-serif',
        h4: { fontSize: '1.5rem', fontWeight: 600 },
        h5: { fontSize: '1.25rem', fontWeight: 600 },
        h6: { fontSize: '1.125rem', fontWeight: 600 },
        body1: { fontSize: '0.875rem', lineHeight: 1.5 },
        body2: { fontSize: '0.75rem', lineHeight: 1.4 },
        caption: { fontSize: '0.6875rem', fontWeight: 500 },
        button: { fontSize: '0.75rem', fontWeight: 600, textTransform: 'none' },
    },
    shape: { borderRadius: 4 },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: tradingColors.tertiary,
                    border: `1px solid ${tradingColors.border}`,
                    backgroundImage: 'none',
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: 4,
                    fontWeight: 600,
                },
            },
        },
    },
});

// Trading-specific utility functions
export const getMarketColor = (value: number) => {
    if (value > 0) return tradingColors.bullish;
    if (value < 0) return tradingColors.bearish;
    return tradingColors.neutral;
};

export const formatMarketValue = (value: number, prefix = '', suffix = '') => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${prefix}${value.toFixed(2)}${suffix}`;
};

export { tradingColors };
