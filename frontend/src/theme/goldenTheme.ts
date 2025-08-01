import { createTheme } from '@mui/material/styles';

// Golden theme utility classes
export const utilityClasses = {
    goldText: {
        color: '#FFD700',
        fontWeight: 'bold',
    },
    darkBackground: {
        backgroundColor: '#0A0A0B',
    },
    glassEffect: {
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 215, 0, 0.2)',
    },
    neonGlow: {
        boxShadow: '0 0 20px rgba(255, 215, 0, 0.3)',
    },
};

// Golden theme
export const goldenTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#FFD700',
            light: '#FFED4E',
            dark: '#B8860B',
        },
        secondary: {
            main: '#FF6B35',
            light: '#FF8A65',
            dark: '#E64A19',
        },
        background: {
            default: '#0A0A0B',
            paper: '#1A1A1B',
        },
        text: {
            primary: '#FFFFFF',
            secondary: '#B8860B',
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
            fontWeight: 700,
            color: '#FFD700',
        },
        h2: {
            fontWeight: 600,
            color: '#FFD700',
        },
    },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(26, 26, 27, 0.8)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 215, 0, 0.2)',
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    textTransform: 'none',
                    fontWeight: 600,
                },
            },
        },
    },
});

export default goldenTheme;
