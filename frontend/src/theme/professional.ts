import { createTheme, ThemeOptions } from '@mui/material/styles';

// Professional Color Palette
const colors = {
    // Primary Colors
    golden: '#FFD700',
    background: '#0A0E1A',
    surface: '#131A2A',
    elevated: '#1E293B',

    // Market Colors
    bullish: '#00D4AA',
    bearish: '#FF4757',
    neutral: '#94A3B8',

    // Text Colors
    textPrimary: '#E2E8F0',
    textSecondary: '#94A3B8',
    textMuted: '#64748B',

    // Accent Colors
    info: '#2196F3',
    warning: '#FFA500',
    success: '#00D4AA',
    error: '#FF4757',

    // Border Colors
    border: '#334155',
    borderLight: '#475569',

    // Hover States
    hoverSurface: '#1E293B',
    hoverElevated: '#334155',
};

// Professional Typography
const typography = {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    monoFamily: '"JetBrains Mono", "Fira Code", monospace',

    // Font Sizes
    xs: '0.75rem',    // 12px
    sm: '0.875rem',   // 14px
    base: '1rem',     // 16px
    lg: '1.125rem',   // 18px
    xl: '1.25rem',    // 20px
    '2xl': '1.5rem',  // 24px
    '3xl': '1.875rem', // 30px

    // Font Weights
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
};

// Professional Spacing System (8px grid)
const spacing = {
    xs: '0.25rem',  // 4px
    sm: '0.5rem',   // 8px
    md: '0.75rem',  // 12px
    lg: '1rem',     // 16px
    xl: '1.5rem',   // 24px
    '2xl': '2rem',  // 32px
    '3xl': '3rem',  // 48px
    '4xl': '4rem',  // 64px
};

// Professional Theme Configuration
export const professionalTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: colors.golden,
            light: '#FFF44F',
            dark: '#B8860B',
            contrastText: colors.background,
        },
        secondary: {
            main: colors.neutral,
            light: '#CBD5E1',
            dark: '#64748B',
            contrastText: colors.textPrimary,
        },
        background: {
            default: colors.background,
            paper: colors.surface,
        },
        surface: {
            main: colors.surface,
            elevated: colors.elevated,
        },
        text: {
            primary: colors.textPrimary,
            secondary: colors.textSecondary,
            disabled: colors.textMuted,
        },
        success: {
            main: colors.bullish,
            light: '#33E6C4',
            dark: '#00A085',
            contrastText: colors.background,
        },
        error: {
            main: colors.bearish,
            light: '#FF6B7F',
            dark: '#CC3945',
            contrastText: colors.textPrimary,
        },
        warning: {
            main: colors.warning,
            light: '#FFB733',
            dark: '#CC8400',
            contrastText: colors.background,
        },
        info: {
            main: colors.info,
            light: '#4FC3F7',
            dark: '#1976D2',
            contrastText: colors.textPrimary,
        },
        divider: colors.border,
    },

    typography: {
        fontFamily: typography.fontFamily,
        fontSize: 16,

        h1: {
            fontSize: typography['3xl'],
            fontWeight: typography.bold,
            lineHeight: 1.2,
            color: colors.textPrimary,
        },
        h2: {
            fontSize: typography['2xl'],
            fontWeight: typography.semibold,
            lineHeight: 1.3,
            color: colors.textPrimary,
        },
        h3: {
            fontSize: typography.xl,
            fontWeight: typography.semibold,
            lineHeight: 1.4,
            color: colors.textPrimary,
        },
        h4: {
            fontSize: typography.lg,
            fontWeight: typography.medium,
            lineHeight: 1.4,
            color: colors.textPrimary,
        },
        h5: {
            fontSize: typography.base,
            fontWeight: typography.medium,
            lineHeight: 1.5,
            color: colors.textPrimary,
        },
        h6: {
            fontSize: typography.sm,
            fontWeight: typography.medium,
            lineHeight: 1.5,
            color: colors.textSecondary,
        },
        body1: {
            fontSize: typography.base,
            fontWeight: typography.normal,
            lineHeight: 1.5,
            color: colors.textPrimary,
        },
        body2: {
            fontSize: typography.sm,
            fontWeight: typography.normal,
            lineHeight: 1.5,
            color: colors.textSecondary,
        },
        caption: {
            fontSize: typography.xs,
            fontWeight: typography.normal,
            lineHeight: 1.4,
            color: colors.textMuted,
        },
        button: {
            fontSize: typography.sm,
            fontWeight: typography.medium,
            textTransform: 'none',
            letterSpacing: 0,
        },
        overline: {
            fontSize: typography.xs,
            fontWeight: typography.medium,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: colors.textMuted,
        },
    },

    spacing: 8, // 8px base unit

    shape: {
        borderRadius: 4,
    },

    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    backgroundColor: colors.background,
                    color: colors.textPrimary,
                    fontFamily: typography.fontFamily,
                },
                '*::-webkit-scrollbar': {
                    width: '8px',
                },
                '*::-webkit-scrollbar-track': {
                    backgroundColor: colors.surface,
                },
                '*::-webkit-scrollbar-thumb': {
                    backgroundColor: colors.border,
                    borderRadius: '4px',
                    '&:hover': {
                        backgroundColor: colors.borderLight,
                    },
                },
            },
        },

        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: colors.surface,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 8,
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    '&:hover': {
                        backgroundColor: colors.hoverSurface,
                        borderColor: colors.borderLight,
                    },
                },
            },
        },

        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 6,
                    textTransform: 'none',
                    fontWeight: typography.medium,
                    padding: '8px 16px',
                    transition: 'all 0.2s ease-in-out',
                },
                contained: {
                    boxShadow: '0 2px 4px -1px rgba(0, 0, 0, 0.2)',
                    '&:hover': {
                        boxShadow: '0 4px 8px -2px rgba(0, 0, 0, 0.3)',
                    },
                },
                outlined: {
                    borderColor: colors.border,
                    '&:hover': {
                        borderColor: colors.borderLight,
                        backgroundColor: colors.hoverSurface,
                    },
                },
            },
        },

        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: colors.surface,
                        borderRadius: 6,
                        '& fieldset': {
                            borderColor: colors.border,
                        },
                        '&:hover fieldset': {
                            borderColor: colors.borderLight,
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: colors.golden,
                        },
                    },
                    '& .MuiInputLabel-root': {
                        color: colors.textSecondary,
                        '&.Mui-focused': {
                            color: colors.golden,
                        },
                    },
                    '& .MuiInputBase-input': {
                        color: colors.textPrimary,
                    },
                },
            },
        },

        MuiChip: {
            styleOverrides: {
                root: {
                    backgroundColor: colors.surface,
                    color: colors.textPrimary,
                    border: `1px solid ${colors.border}`,
                    '&:hover': {
                        backgroundColor: colors.hoverSurface,
                    },
                },
                filled: {
                    '&.MuiChip-colorSuccess': {
                        backgroundColor: colors.bullish,
                        color: colors.background,
                    },
                    '&.MuiChip-colorError': {
                        backgroundColor: colors.bearish,
                        color: colors.textPrimary,
                    },
                    '&.MuiChip-colorWarning': {
                        backgroundColor: colors.warning,
                        color: colors.background,
                    },
                },
            },
        },

        MuiAppBar: {
            styleOverrides: {
                root: {
                    backgroundColor: colors.surface,
                    borderBottom: `1px solid ${colors.border}`,
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                },
            },
        },

        MuiDrawer: {
            styleOverrides: {
                paper: {
                    backgroundColor: colors.surface,
                    borderRight: `1px solid ${colors.border}`,
                },
            },
        },

        MuiTableCell: {
            styleOverrides: {
                root: {
                    borderBottom: `1px solid ${colors.border}`,
                    color: colors.textPrimary,
                },
                head: {
                    backgroundColor: colors.elevated,
                    color: colors.textSecondary,
                    fontWeight: typography.medium,
                },
            },
        },

        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: colors.elevated,
                    color: colors.textPrimary,
                    border: `1px solid ${colors.border}`,
                    fontSize: typography.sm,
                },
            },
        },
    },
} as ThemeOptions);

// Utility functions for theme usage
export const getMarketColor = (value: number, neutral = 0) => {
    if (value > neutral) return colors.bullish;
    if (value < neutral) return colors.bearish;
    return colors.neutral;
};

export const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return colors.bullish;
    if (confidence >= 60) return colors.warning;
    return colors.bearish;
};

export const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
};

export const formatPercentage = (value: number) => {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value / 100);
};

export default professionalTheme; 