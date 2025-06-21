import { createTheme } from '@mui/material/styles';

// Dark Pro Theme - Professional Trading Platform
export const darkProTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#0A84FF', // Vibrant blue for primary actions
            light: '#409CFF',
            dark: '#0066CC',
            contrastText: '#FFFFFF',
        },
        secondary: {
            main: '#5E5CE6', // Elegant purple for secondary elements
            light: '#7C7AFF',
            dark: '#4A48B3',
            contrastText: '#FFFFFF',
        },
        background: {
            default: '#0A0A0B', // Ultra dark background
            paper: '#141416', // Slightly lighter for cards
        },
        text: {
            primary: '#F5F5F7', // High contrast white
            secondary: '#86868B', // Muted gray for secondary text
        },
        success: {
            main: '#30D158', // Vibrant green
            light: '#4ADE80',
            dark: '#22C55E',
        },
        error: {
            main: '#FF453A', // Bright red
            light: '#FF6B6B',
            dark: '#DC2626',
        },
        warning: {
            main: '#FFD60A', // Golden yellow
            light: '#FACC15',
            dark: '#EAB308',
        },
        info: {
            main: '#64D2FF', // Cyan blue
            light: '#7DD3FC',
            dark: '#0EA5E9',
        },
        divider: 'rgba(255, 255, 255, 0.08)',
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
        fontFamily: [
            'SF Pro Display',
            '-apple-system',
            'BlinkMacSystemFont',
            'Inter',
            'Segoe UI',
            'Roboto',
            'Helvetica Neue',
            'Arial',
            'sans-serif',
        ].join(','),
        h1: {
            fontSize: '3.5rem',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            lineHeight: 1.1,
        },
        h2: {
            fontSize: '2.75rem',
            fontWeight: 700,
            letterSpacing: '-0.015em',
            lineHeight: 1.2,
        },
        h3: {
            fontSize: '2.25rem',
            fontWeight: 600,
            letterSpacing: '-0.01em',
            lineHeight: 1.3,
        },
        h4: {
            fontSize: '1.75rem',
            fontWeight: 600,
            letterSpacing: '-0.005em',
            lineHeight: 1.4,
        },
        h5: {
            fontSize: '1.375rem',
            fontWeight: 600,
            lineHeight: 1.5,
        },
        h6: {
            fontSize: '1.125rem',
            fontWeight: 600,
            lineHeight: 1.6,
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
            letterSpacing: '0.02em',
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
                    backgroundColor: '#141416',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
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
                    backgroundColor: '#1A1A1C',
                    backgroundImage: 'linear-gradient(135deg, #1A1A1C 0%, #141416 100%)',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    borderRadius: 16,
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 12px 24px rgba(0, 0, 0, 0.4)',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 10,
                    padding: '10px 20px',
                    fontSize: '0.875rem',
                    fontWeight: 600,
                    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                    textTransform: 'none',
                },
                containedPrimary: {
                    background: 'linear-gradient(135deg, #0A84FF 0%, #0066CC 100%)',
                    boxShadow: '0 4px 12px rgba(10, 132, 255, 0.3)',
                    '&:hover': {
                        background: 'linear-gradient(135deg, #409CFF 0%, #0A84FF 100%)',
                        boxShadow: '0 6px 16px rgba(10, 132, 255, 0.4)',
                        transform: 'translateY(-1px)',
                    },
                    '&:active': {
                        transform: 'translateY(0)',
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
                        backgroundColor: 'rgba(255, 255, 255, 0.02)',
                        borderRadius: 10,
                        transition: 'all 0.2s ease',
                        '& fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            transition: 'all 0.2s ease',
                        },
                        '&:hover fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: '#0A84FF',
                            borderWidth: '2px',
                        },
                        '&.Mui-focused': {
                            backgroundColor: 'rgba(10, 132, 255, 0.05)',
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
                    fontWeight: 500,
                    borderRadius: 8,
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.12)',
                    },
                },
                colorPrimary: {
                    backgroundColor: 'rgba(10, 132, 255, 0.2)',
                    color: '#409CFF',
                    '&:hover': {
                        backgroundColor: 'rgba(10, 132, 255, 0.3)',
                    },
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
                    backgroundColor: '#262626',
                    color: '#F5F5F7',
                    fontSize: '0.75rem',
                    fontWeight: 500,
                    padding: '8px 12px',
                    borderRadius: 8,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
                },
                arrow: {
                    color: '#262626',
                },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: {
                    borderColor: 'rgba(255, 255, 255, 0.08)',
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
                    borderRadius: 10,
                    fontWeight: 500,
                },
                standardSuccess: {
                    backgroundColor: 'rgba(48, 209, 88, 0.1)',
                    color: '#30D158',
                    '& .MuiAlert-icon': {
                        color: '#30D158',
                    },
                },
                standardError: {
                    backgroundColor: 'rgba(255, 69, 58, 0.1)',
                    color: '#FF453A',
                    '& .MuiAlert-icon': {
                        color: '#FF453A',
                    },
                },
                standardWarning: {
                    backgroundColor: 'rgba(255, 214, 10, 0.1)',
                    color: '#FFD60A',
                    '& .MuiAlert-icon': {
                        color: '#FFD60A',
                    },
                },
                standardInfo: {
                    backgroundColor: 'rgba(100, 210, 255, 0.1)',
                    color: '#64D2FF',
                    '& .MuiAlert-icon': {
                        color: '#64D2FF',
                    },
                },
            },
        },
        MuiTableCell: {
            styleOverrides: {
                root: {
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                },
                head: {
                    backgroundColor: '#0A0A0B',
                    fontWeight: 600,
                    color: '#86868B',
                    fontSize: '0.75rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
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