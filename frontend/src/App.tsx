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

import React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { Box } from '@mui/material';
import { BrowserRouter as Router } from 'react-router-dom';
import { AppRoutes } from './AppRoutes';
import { AlertProvider } from './contexts/AlertContext';
import AIChatButton from './components/AI/AIChatButton';
import { CommandPalette } from './components/Common/CommandPalette';
import { darkProTheme } from './theme/darkPro';

// Create a client
const queryClient = new QueryClient();

// Professional Trading Platform Theme
export const tradingTheme = darkProTheme;

function AppContent() {
  return (
    <>
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

      {/* Main App Content */}
      <AppRoutes />

      {/* Command Palette */}
      <CommandPalette />

      {/* AI Chat */}
      <AIChatButton />
    </>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={tradingTheme}>
        <Router
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
          <AlertProvider>
            <AppContent />
          </AlertProvider>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App; 