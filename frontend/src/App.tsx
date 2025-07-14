/**
 * GoldenSignals AI - Professional Trading Platform
 * 
 * Simplified version that focuses on the working TradingSignalsApp
 * to avoid import conflicts and dependency issues.
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Provider } from 'react-redux';
import { Toaster } from 'react-hot-toast';

// Professional theme
import professionalTheme from './theme/professional';

// Store
import { store } from './store';

// Contexts
import { AlertProvider } from './contexts/AlertContext';
import { ErrorProvider } from './contexts/ErrorContext';

// Main application component
import TradingSignalsApp from './pages/TradingSignals/TradingSignalsApp';
import TradingDashboard from './pages/TradingDashboard';

// Create query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// Service worker cleanup
if ('serviceWorker' in navigator) {
  console.log('Starting cleanup of service workers and caches...');

  // Clear storage
  try {
    localStorage.clear();
    sessionStorage.clear();
    console.log('Cleared local and session storage');
  } catch (error) {
    console.error('Error clearing storage:', error);
  }

  // Unregister service workers
  navigator.serviceWorker.getRegistrations().then(function (registrations) {
    console.log(`Found ${registrations.length} service worker(s)`);
    for (let registration of registrations) {
      registration.unregister();
    }
  });

  // Clear caches
  if ('caches' in window) {
    caches.keys().then(function (names) {
      console.log(`Found ${names.length} cache(s)`);
      for (let name of names) {
        caches.delete(name);
      }
    });
  }
}

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={professionalTheme}>
          <CssBaseline />
          <ErrorProvider>
            <AlertProvider>
              <Router>
                <Routes>
                  <Route path="/" element={<TradingDashboard />} />
                  <Route path="/dashboard" element={<TradingDashboard />} />
                  <Route path="/trading-signals" element={<TradingSignalsApp />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Router>
              <Toaster
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: professionalTheme.palette.background.paper,
                    color: professionalTheme.palette.text.primary,
                    border: `1px solid ${professionalTheme.palette.divider}`,
                  },
                }}
              />
            </AlertProvider>
          </ErrorProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </Provider>
  );
};

export default App; 