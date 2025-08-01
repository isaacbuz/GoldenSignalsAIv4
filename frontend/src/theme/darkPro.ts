import { createTheme } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';

// Dark Pro Theme - Professional Trading Platform
export const darkProTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#FFD700', // Gold
            light: '#FFED4E',
            dark: '#FFA000',
            contrastText: '#000000',
        },
        secondary: {
            main: '#FFC107', // Amber Gold
            light: '#FFE082',
            dark: '#FF8F00',
            contrastText: '#000000',
        },
        background: {
            default: '#0a0a0a',
            paper: '#121212',
        },
        text: {
            primary: '#ffffff',
            secondary: '#a0a0a0',
        },
        success: {
            main: '#69f0ae',
            light: '#9fffe0',
            dark: '#2bbd7e',
        },
        error: {
            main: '#ff5252',
            light: '#ff867f',
            dark: '#c50e29',
        },
        warning: {
            main: '#FFD700', // Gold for warnings
            light: '#FFED4E',
            dark: '#FFA000',
        },
        info: {
            main: '#40c4ff',
            light: '#82f7ff',
            dark: '#0094cc',
        },
        divider: alpha('#FFD700', 0.12),
        grey: {
            50: '#FAFAFA',
            100: '#F5F5F5',
            200: '#E5E5E5',
            300: '#D4D4D4',
            400: '#A3A3A3',
            500: '#737373',
            600: '#525252',
            700: '#404040',
            800: '#262626',
            900: '#171717',
        },
    },
    typography: {
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        h1: {
            fontWeight: 700,
            letterSpacing: '-0.02em',
        },
        h2: {
            fontWeight: 600,
            letterSpacing: '-0.01em',
        },
        h3: {
            fontWeight: 600,
        },
        h4: {
            fontWeight: 600,
        },
        h5: {
            fontWeight: 500,
        },
        h6: {
            fontWeight: 500,
        },
        subtitle1: {
            fontSize: '1rem',
            fontWeight: 500,
            letterSpacing: '0.01em',
            color: '#86868B',
        },
        subtitle2: {
            fontSize: '0.875rem',
            fontWeight: 500,
            letterSpacing: '0.01em',
            color: '#86868B',
        },
        body1: {
            fontSize: '1rem',
            fontWeight: 400,
            letterSpacing: '0.005em',
            lineHeight: 1.7,
        },
        body2: {
            fontSize: '0.875rem',
            fontWeight: 400,
            letterSpacing: '0.005em',
            lineHeight: 1.6,
        },
        caption: {
            fontSize: '0.75rem',
            fontWeight: 400,
            letterSpacing: '0.02em',
            color: '#86868B',
        },
        overline: {
            fontSize: '0.75rem',
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: '#86868B',
        },
        button: {
            textTransform: 'none',
            fontWeight: 600,
        },
    },
    shape: {
        borderRadius: 12,
    },
    shadows: [
        'none',
        '0px 2px 4px rgba(0, 0, 0, 0.6)',
        '0px 4px 8px rgba(0, 0, 0, 0.6)',
        '0px 8px 16px rgba(0, 0, 0, 0.6)',
        '0px 12px 24px rgba(0, 0, 0, 0.6)',
        '0px 16px 32px rgba(0, 0, 0, 0.6)',
        '0px 20px 40px rgba(0, 0, 0, 0.6)',
        '0px 24px 48px rgba(0, 0, 0, 0.6)',
        '0px 28px 56px rgba(0, 0, 0, 0.6)',
        '0px 32px 64px rgba(0, 0, 0, 0.6)',
        '0px 36px 72px rgba(0, 0, 0, 0.6)',
        '0px 40px 80px rgba(0, 0, 0, 0.6)',
        '0px 44px 88px rgba(0, 0, 0, 0.6)',
        '0px 48px 96px rgba(0, 0, 0, 0.6)',
        '0px 52px 104px rgba(0, 0, 0, 0.6)',
        '0px 56px 112px rgba(0, 0, 0, 0.6)',
        '0px 60px 120px rgba(0, 0, 0, 0.6)',
        '0px 64px 128px rgba(0, 0, 0, 0.6)',
        '0px 68px 136px rgba(0, 0, 0, 0.6)',
        '0px 72px 144px rgba(0, 0, 0, 0.6)',
        '0px 76px 152px rgba(0, 0, 0, 0.6)',
        '0px 80px 160px rgba(0, 0, 0, 0.6)',
        '0px 84px 168px rgba(0, 0, 0, 0.6)',
        '0px 88px 176px rgba(0, 0, 0, 0.6)',
        '0px 92px 184px rgba(0, 0, 0, 0.6)',
    ],
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    scrollbarColor: '#525252 #141416',
                    '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
                        width: 8,
                        height: 8,
                    },
                    '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
                        borderRadius: 8,
                        backgroundColor: '#525252',
                        border: '2px solid #141416',
                    },
                    '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
                        backgroundColor: '#141416',
                    },
                },
            },
        },
        MuiPaper: {
            defaultProps: {
                elevation: 0,
            },
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                    backgroundColor: '#121212',
                    border: '1px solid rgba(255, 215, 0, 0.1)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        borderColor: 'rgba(255, 215, 0, 0.3)',
                        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.8), 0 0 20px rgba(255, 215, 0, 0.1)',
                    },
                },
                elevation1: {
                    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.5), 0 1px 2px rgba(0, 0, 0, 0.3)',
                },
                elevation2: {
                    boxShadow: '0 3px 6px rgba(0, 0, 0, 0.5), 0 3px 6px rgba(0, 0, 0, 0.3)',
                },
                elevation3: {
                    boxShadow: '0 10px 20px rgba(0, 0, 0, 0.5), 0 6px 6px rgba(0, 0, 0, 0.3)',
                },
            },
        },
        MuiCard: {
            defaultProps: {
                elevation: 0,
            },
            styleOverrides: {
                root: {
                    backgroundColor: '#121212',
                    border: '1px solid rgba(255, 215, 0, 0.1)',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.8), 0 0 20px rgba(255, 215, 0, 0.2)',
                        borderColor: 'rgba(255, 215, 0, 0.3)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    padding: '8px 16px',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 4px 20px rgba(255, 215, 0, 0.3)',
                    },
                },
                containedPrimary: {
                    background: 'linear-gradient(135deg, #FFD700 0%, #FFA000 100%)',
                    color: '#000000',
                    fontWeight: 600,
                    '&:hover': {
                        background: 'linear-gradient(135deg, #FFED4E 0%, #FFB300 100%)',
                    },
                },
                containedSecondary: {
                    background: 'linear-gradient(135deg, #5E5CE6 0%, #4A48B3 100%)',
                    boxShadow: '0 4px 12px rgba(94, 92, 230, 0.3)',
                    '&:hover': {
                        background: 'linear-gradient(135deg, #7C7AFF 0%, #5E5CE6 100%)',
                        boxShadow: '0 6px 16px rgba(94, 92, 230, 0.4)',
                        transform: 'translateY(-1px)',
                    },
                },
                outlined: {
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    '&:hover': {
                        borderColor: 'rgba(255, 255, 255, 0.4)',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    },
                },
                text: {
                    '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    },
                },
            },
        },
        MuiTextField: {
            defaultProps: {
                variant: 'outlined',
            },
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: '#1a1a1a',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                            backgroundColor: '#1e1e1e',
                            '& .MuiOutlinedInput-notchedOutline': {
                                borderColor: '#FFD700',
                            },
                        },
                        '&.Mui-focused': {
                            backgroundColor: '#1e1e1e',
                            '& .MuiOutlinedInput-notchedOutline': {
                                borderColor: '#FFD700',
                                borderWidth: 2,
                            },
                        },
                    },
                    '& .MuiInputLabel-root': {
                        color: '#86868B',
                        '&.Mui-focused': {
                            color: '#0A84FF',
                        },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    fontWeight: 600,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        transform: 'scale(1.05)',
                        boxShadow: '0 2px 8px rgba(255, 215, 0, 0.3)',
                    },
                },
                colorPrimary: {
                    backgroundColor: alpha('#FFD700', 0.2),
                    color: '#FFD700',
                    border: '1px solid rgba(255, 215, 0, 0.3)',
                },
                colorSecondary: {
                    backgroundColor: 'rgba(94, 92, 230, 0.2)',
                    color: '#7C7AFF',
                    '&:hover': {
                        backgroundColor: 'rgba(94, 92, 230, 0.3)',
                    },
                },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: '#1a1a1a',
                    color: '#ffffff',
                    border: '1px solid rgba(255, 215, 0, 0.2)',
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.8)',
                },
                arrow: {
                    color: '#262626',
                },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: {
                    borderColor: alpha('#FFD700', 0.12),
                },
            },
        },
        MuiSwitch: {
            styleOverrides: {
                root: {
                    width: 42,
                    height: 26,
                    padding: 0,
                    '& .MuiSwitch-switchBase': {
                        padding: 0,
                        margin: 2,
                        transitionDuration: '300ms',
                        '&.Mui-checked': {
                            transform: 'translateX(16px)',
                            color: '#fff',
                            '& + .MuiSwitch-track': {
                                backgroundColor: '#0A84FF',
                                opacity: 1,
                                border: 0,
                            },
                        },
                    },
                    '& .MuiSwitch-thumb': {
                        boxSizing: 'border-box',
                        width: 22,
                        height: 22,
                    },
                    '& .MuiSwitch-track': {
                        borderRadius: 26 / 2,
                        backgroundColor: '#525252',
                        opacity: 1,
                        transition: 'background-color 300ms',
                    },
                },
            },
        },
        MuiAlert: {
            styleOverrides: {
                root: {
                    border: '1px solid',
                    borderRadius: 8,
                },
                standardSuccess: {
                    backgroundColor: alpha('#69f0ae', 0.1),
                    borderColor: '#69f0ae',
                },
                standardError: {
                    backgroundColor: alpha('#ff5252', 0.1),
                    borderColor: '#ff5252',
                },
                standardWarning: {
                    backgroundColor: alpha('#FFD700', 0.1),
                    borderColor: '#FFD700',
                    color: '#FFD700',
                },
                standardInfo: {
                    backgroundColor: alpha('#40c4ff', 0.1),
                    borderColor: '#40c4ff',
                },
            },
        },
        MuiTableCell: {
            styleOverrides: {
                root: {
                    borderBottom: '1px solid rgba(255, 215, 0, 0.08)',
                },
                head: {
                    backgroundColor: '#1a1a1a',
                    fontWeight: 600,
                    color: '#FFD700',
                },
            },
        },
        MuiTableRow: {
            styleOverrides: {
                root: {
                    '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.02)',
                    },
                },
            },
        },
        MuiAppBar: {
            styleOverrides: {
                root: {
                    backgroundColor: alpha('#0a0a0a', 0.95),
                    borderBottom: '1px solid rgba(255, 215, 0, 0.2)',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.8), 0 0 20px rgba(255, 215, 0, 0.05)',
                },
            },
        },
        MuiBadge: {
            styleOverrides: {
                colorError: {
                    backgroundColor: '#FFD700',
                    color: '#000000',
                    fontWeight: 700,
                },
            },
        },
        MuiIconButton: {
            styleOverrides: {
                root: {
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        backgroundColor: alpha('#FFD700', 0.08),
                        transform: 'scale(1.1)',
                    },
                },
            },
        },
    },
});

// Custom color utilities for charts and data visualization
export const chartColors = {
    primary: ['#0A84FF', '#409CFF', '#0066CC'],
    success: ['#30D158', '#4ADE80', '#22C55E'],
    error: ['#FF453A', '#FF6B6B', '#DC2626'],
    warning: ['#FFD60A', '#FACC15', '#EAB308'],
    info: ['#64D2FF', '#7DD3FC', '#0EA5E9'],
    purple: ['#5E5CE6', '#7C7AFF', '#4A48B3'],
    gradient: {
        blue: 'linear-gradient(135deg, #0A84FF 0%, #0066CC 100%)',
        green: 'linear-gradient(135deg, #30D158 0%, #22C55E 100%)',
        red: 'linear-gradient(135deg, #FF453A 0%, #DC2626 100%)',
        purple: 'linear-gradient(135deg, #5E5CE6 0%, #4A48B3 100%)',
        gold: 'linear-gradient(135deg, #FFD60A 0%, #EAB308 100%)',
    },
};

// Utility classes for common styles
export const darkProStyles = {
    glassmorphism: {
        background: 'rgba(20, 20, 22, 0.7)',
        backdropFilter: 'blur(20px) saturate(180%)',
        border: '1px solid rgba(255, 255, 255, 0.05)',
    },
    glow: {
        primary: '0 0 20px rgba(10, 132, 255, 0.5)',
        success: '0 0 20px rgba(48, 209, 88, 0.5)',
        error: '0 0 20px rgba(255, 69, 58, 0.5)',
    },
    animation: {
        fadeIn: 'fadeIn 0.3s ease-in-out',
        slideUp: 'slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        pulse: 'pulse 2s infinite',
    },
};
