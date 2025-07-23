/**
 * GoldenSignals AI - Professional Trading Application
 *
 * Uses the comprehensive AITradingChart component with:
 * - Real-time AI agent analysis
 * - Auto-analyze on symbol/timeframe change
 * - Multi-agent consensus building
 * - Professional charting features
 */

import React from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { AITradingChart } from './components/AIChart/AITradingChart';
import { createTheme } from '@mui/material/styles';

// Professional dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#FFD700', // Golden accent
    },
    secondary: {
      main: '#00FF88',
    },
    background: {
      default: '#000000',
      paper: '#0a0a0a',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#B8BCC8',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#000000',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <AITradingChart height="100%" />
      </div>
    </ThemeProvider>
  );
};

export default App;
