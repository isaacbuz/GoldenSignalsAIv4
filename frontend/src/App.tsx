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
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { Provider } from 'react-redux';
import { store } from './store/store';
import { goldenTheme } from './theme/goldenTheme';

// AI Signal Platform Pages
import AICommandCenter from './pages/AICommandCenter/AICommandCenter';
import SignalStream from './pages/SignalStream/SignalStream';
import AIAssistant from './pages/AIAssistant/AIAssistant';
import SignalAnalytics from './pages/SignalAnalytics/SignalAnalytics';
import ModelDashboard from './pages/ModelDashboard/ModelDashboard';
import MarketIntelligence from './pages/MarketIntelligence/MarketIntelligence';
import SignalHistory from './pages/SignalHistory/SignalHistory';
import AdminPanel from './pages/Admin/AdminPanel';
import Settings from './pages/Settings/Settings';

// Layout
import MainLayout from './components/Layout/MainLayout';

// Notifications
import { NotificationProvider } from './components/Notifications/NotificationProvider';

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <ThemeProvider theme={goldenTheme}>
        <CssBaseline />
        <Router>
          <NotificationProvider>
            <MainLayout>
              <Routes>
                {/* Default route - AI Command Center */}
                <Route path="/" element={<Navigate to="/command-center" replace />} />

                {/* AI Signal Platform Routes */}
                <Route path="/command-center" element={<AICommandCenter />} />
                <Route path="/signals" element={<SignalStream />} />
                <Route path="/ai-assistant" element={<AIAssistant />} />
                <Route path="/analytics" element={<SignalAnalytics />} />
                <Route path="/models" element={<ModelDashboard />} />
                <Route path="/intelligence" element={<MarketIntelligence />} />
                <Route path="/history" element={<SignalHistory />} />
                <Route path="/admin" element={<AdminPanel />} />
                <Route path="/settings" element={<Settings />} />

                {/* Catch all - redirect to command center */}
                <Route path="*" element={<Navigate to="/command-center" replace />} />
              </Routes>
            </MainLayout>
          </NotificationProvider>
        </Router>
      </ThemeProvider>
    </Provider>
  );
};

export default App; 