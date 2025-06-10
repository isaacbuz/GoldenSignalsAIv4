/**
 * GoldenSignals AI - Premium Trading Platform
 * 
 * My Vision: A sophisticated, data-driven trading interface that combines
 * the elegance of Apple's design system with the power of professional trading tools.
 * 
 * Design Philosophy:
 * - Clarity over complexity
 * - Data-first approach
 * - Subtle elegance
 * - Professional aesthetics
 */

import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { AppRoutes } from './AppRoutes';

// Create a client
const queryClient = new QueryClient();

// Professional Trading Platform Theme
export const tradingTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#007AFF', // Apple's classic blue
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#5E5CE6', // Apple's purple
    },
    background: {
      default: '#000000',
      paper: 'rgba(28, 28, 30, 0.8)', // Slightly off-black for cards
    },
    text: {
      primary: '#E5E5E7',
      secondary: '#A0A0A5',
    },
    success: {
      main: '#34C759', // Apple's green
    },
    error: {
      main: '#FF3B30', // Apple's red
    },
    warning: {
      main: '#FF9500', // Apple's orange
    },
    divider: 'rgba(255, 255, 255, 0.12)',
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
    h1: { fontSize: '3rem', fontWeight: 700, letterSpacing: '-0.015em' },
    h2: { fontSize: '2.5rem', fontWeight: 700, letterSpacing: '-0.01em' },
    h3: { fontSize: '2rem', fontWeight: 600, letterSpacing: '-0.005em' },
    h4: { fontSize: '1.5rem', fontWeight: 600 },
    h5: { fontSize: '1.25rem', fontWeight: 600 },
    h6: { fontSize: '1.1rem', fontWeight: 600 },
    subtitle1: { fontSize: '1rem', fontWeight: 500, color: '#A0A0A5' },
    body1: { fontSize: '1rem', fontWeight: 400 },
    body2: { fontSize: '0.875rem', fontWeight: 400 },
    caption: { fontSize: '0.75rem', color: '#A0A0A5' },
    button: { textTransform: 'none', fontWeight: 600 },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backdropFilter: 'blur(20px)',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: '1px',
          borderStyle: 'solid',
        },
      },
    },
    MuiCard: {
      defaultProps: {
        elevation: 0,
      },
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(28, 28, 30, 0.7)',
          backdropFilter: 'blur(30px) saturate(180%)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 16,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          transition: 'all 0.2s ease-in-out',
        },
        containedPrimary: {
          '&:hover': {
            backgroundColor: '#0059b3',
            transform: 'scale(1.02)',
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
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            borderRadius: 8,
            '& fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.2)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.4)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#007AFF',
              borderWidth: '2px',
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={tradingTheme}>
        <CssBaseline />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#333',
              color: '#fff',
            },
          }}
        />
        <AppRoutes />
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App; 