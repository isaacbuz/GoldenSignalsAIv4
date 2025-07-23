/**
 * Minimal Chart-Only Application
 *
 * Using the new IntelligentChart component
 */

import React, { useState } from 'react';
import { Box, ThemeProvider, createTheme } from '@mui/material';
import { IntelligentChart } from './components/Chart/IntelligentChart';

// Dark theme for AI-powered interface
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#FFD700',
    },
    background: {
      default: '#0a0a0a',
      paper: '#101010',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.6)',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
});

const MinimalChartApp: React.FC = () => {
  const [symbol, setSymbol] = useState('TSLA');

  return (
    <ThemeProvider theme={darkTheme}>
      <Box sx={{ width: '100vw', height: '100vh', bgcolor: '#0a0a0a' }}>
        <IntelligentChart symbol={symbol} onSymbolChange={setSymbol} />
      </Box>
    </ThemeProvider>
  );
};

export default MinimalChartApp;
