import { createTheme, ThemeOptions } from '@mui/material/styles';

// Golden color palette
const goldenPalette = {
    primary: {
        main: '#FFD700', // Gold
        light: '#FFED4E',
        dark: '#C5A200',
        contrastText: '#000000',
    },
    secondary: {
        main: '#FFA500', // Orange Gold
        light: '#FFB84D',
        dark: '#CC8400',
        contrastText: '#000000',
    },
    background: {
        default: '#0D1117', // Darker background
        paper: '#0A0E27', // Dark background
        elevated: '#161B33', // Slightly lighter for elevation
    },
    success: {
        main: '#4CAF50',
        light: '#81C784',
        dark: '#388E3C',
    },
    error: {
        main: '#F44336',
        light: '#E57373',
        dark: '#D32F2F',
    },
    warning: {
        main: '#FFA500',
        light: '#FFB84D',
        dark: '#F57C00',
    },
    info: {
        main: '#2196F3',
        light: '#64B5F6',
        dark: '#1976D2',
    },
    text: {
        primary: '#FFFFFF',
        secondary: 'rgba(255, 255, 255, 0.7)',
        disabled: 'rgba(255, 255, 255, 0.5)',
    },
    divider: 'rgba(255, 215, 0, 0.12)',
    action: {
        active: '#FFD700',
        hover: 'rgba(255, 215, 0, 0.08)',
        selected: 'rgba(255, 215, 0, 0.16)',
        disabled: 'rgba(255, 255, 255, 0.3)',
        disabledBackground: 'rgba(255, 255, 255, 0.12)',
    },
};

// Typography configuration
const typography = {
    fontFamily: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    h1: {
        fontSize: '2.5rem',
        fontWeight: 700,
        letterSpacing: '-0.02em',
        lineHeight: 1.2,
    },
    h2: {
        fontSize: '2rem',
        fontWeight: 600,
        letterSpacing: '-0.01em',
        lineHeight: 1.3,
    },
    h3: {
        fontSize: '1.75rem',
        fontWeight: 600,
        letterSpacing: '-0.01em',
        lineHeight: 1.4,
    },
    h4: {
        fontSize: '1.5rem',
        fontWeight: 600,
        lineHeight: 1.4,
    },
    h5: {
        fontSize: '1.25rem',
        fontWeight: 600,
        lineHeight: 1.5,
    },
    h6: {
        fontSize: '1.125rem',
        fontWeight: 600,
        lineHeight: 1.5,
    },
    body1: {
        fontSize: '1rem',
        lineHeight: 1.6,
    },
    body2: {
        fontSize: '0.875rem',
        lineHeight: 1.6,
    },
    button: {
        textTransform: 'none' as const,
        fontWeight: 600,
    },
    caption: {
        fontSize: '0.75rem',
        lineHeight: 1.6,
    },
    overline: {
        fontSize: '0.75rem',
        fontWeight: 600,
        letterSpacing: '0.1em',
        textTransform: 'uppercase' as const,
    },
    // Custom variants for data display
    dataLarge: {
        fontFamily: '"JetBrains Mono", "SF Mono", monospace',
        fontSize: '2rem',
        fontWeight: 700,
        letterSpacing: '-0.02em',
    },
    dataMedium: {
        fontFamily: '"JetBrains Mono", "SF Mono", monospace',
        fontSize: '1.25rem',
        fontWeight: 600,
    },
    dataSmall: {
        fontFamily: '"JetBrains Mono", "SF Mono", monospace',
        fontSize: '0.875rem',
        fontWeight: 500,
    },
};

// Component overrides
const components = {
    MuiCssBaseline: {
        styleOverrides: {
            body: {
                scrollbarColor: '#FFD700 #0A0E27',
                '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
                    width: '8px',
                    height: '8px',
                },
                '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
                    borderRadius: 8,
                    backgroundColor: '#FFD700',
                    minHeight: 24,
                },
                '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
                    backgroundColor: '#0A0E27',
                },
            },
        },
    },
    MuiButton: {
        styleOverrides: {
            root: {
                borderRadius: 8,
                padding: '8px 16px',
                transition: 'all 0.2s ease-in-out',
            },
            containedPrimary: {
                background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
                boxShadow: '0 3px 10px 2px rgba(255, 215, 0, .3)',
                '&:hover': {
                    background: 'linear-gradient(45deg, #FFA500 30%, #FFD700 90%)',
                    boxShadow: '0 3px 15px 3px rgba(255, 215, 0, .4)',
                },
            },
        },
    },
    MuiCard: {
        styleOverrides: {
            root: {
                backgroundImage: 'none',
                backgroundColor: 'rgba(10, 14, 39, 0.8)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 215, 0, 0.1)',
                borderRadius: 12,
                transition: 'all 0.3s ease-in-out',
                '&:hover': {
                    borderColor: 'rgba(255, 215, 0, 0.3)',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 8px 24px rgba(255, 215, 0, 0.2)',
                },
            },
        },
    },
    MuiPaper: {
        styleOverrides: {
            root: {
                backgroundImage: 'none',
                backgroundColor: '#0A0E27',
                borderRadius: 8,
            },
            elevation1: {
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
            },
            elevation2: {
                boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
            },
            elevation3: {
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
            },
        },
    },
    MuiChip: {
        styleOverrides: {
            root: {
                borderRadius: 6,
                fontWeight: 600,
            },
            colorPrimary: {
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                color: '#FFD700',
                border: '1px solid rgba(255, 215, 0, 0.3)',
            },
        },
    },
    MuiLinearProgress: {
        styleOverrides: {
            root: {
                borderRadius: 4,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
            barColorPrimary: {
                backgroundColor: '#FFD700',
                borderRadius: 4,
            },
        },
    },
    MuiAlert: {
        styleOverrides: {
            root: {
                borderRadius: 8,
                backdropFilter: 'blur(10px)',
            },
            standardSuccess: {
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                color: '#4CAF50',
                border: '1px solid rgba(76, 175, 80, 0.3)',
            },
            standardError: {
                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                color: '#F44336',
                border: '1px solid rgba(244, 67, 54, 0.3)',
            },
            standardWarning: {
                backgroundColor: 'rgba(255, 165, 0, 0.1)',
                color: '#FFA500',
                border: '1px solid rgba(255, 165, 0, 0.3)',
            },
            standardInfo: {
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                color: '#2196F3',
                border: '1px solid rgba(33, 150, 243, 0.3)',
            },
        },
    },
    MuiTooltip: {
        styleOverrides: {
            tooltip: {
                backgroundColor: '#161B33',
                color: '#FFFFFF',
                border: '1px solid rgba(255, 215, 0, 0.2)',
                borderRadius: 6,
                fontSize: '0.875rem',
                padding: '8px 12px',
            },
            arrow: {
                color: '#161B33',
                '&:before': {
                    border: '1px solid rgba(255, 215, 0, 0.2)',
                },
            },
        },
    },
    MuiSwitch: {
        styleOverrides: {
            root: {
                width: 42,
                height: 26,
                padding: 0,
            },
            switchBase: {
                padding: 0,
                margin: 2,
                transitionDuration: '300ms',
                '&.Mui-checked': {
                    transform: 'translateX(16px)',
                    color: '#fff',
                    '& + .MuiSwitch-track': {
                        backgroundColor: '#FFD700',
                        opacity: 1,
                        border: 0,
                    },
                },
            },
            thumb: {
                boxSizing: 'border-box',
                width: 22,
                height: 22,
            },
            track: {
                borderRadius: 26 / 2,
                backgroundColor: 'rgba(255, 255, 255, 0.3)',
                opacity: 1,
            },
        },
    },
};

// Create theme options
const themeOptions: ThemeOptions = {
    palette: goldenPalette,
    typography,
    components,
    shape: {
        borderRadius: 8,
    },
    transitions: {
        duration: {
            shortest: 150,
            shorter: 200,
            short: 250,
            standard: 300,
            complex: 375,
            enteringScreen: 225,
            leavingScreen: 195,
        },
    },
};

// Create and export the theme
export const goldenTheme = createTheme(themeOptions);

// Custom animations for the theme
export const animations = {
    pulse: `
    @keyframes pulse {
      0% { opacity: 1; }
      50% { opacity: 0.6; }
      100% { opacity: 1; }
    }
  `,
    spin: `
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `,
    glow: `
    @keyframes glow {
      0% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.5); }
      50% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.8), 0 0 30px rgba(255, 215, 0, 0.6); }
      100% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.5); }
    }
  `,
    slideIn: `
    @keyframes slideIn {
      from { transform: translateX(-100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `,
    fadeIn: `
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  `,
};

// Export utility classes
export const utilityClasses = {
    glassmorphism: {
        background: 'rgba(10, 14, 39, 0.8)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 215, 0, 0.1)',
    },
    goldenGradient: {
        background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
    },
    textGradient: {
        background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
    },
    aiProcessing: {
        animation: 'pulse 2s infinite',
    },
    glowEffect: {
        animation: 'glow 2s infinite',
    },
}; 