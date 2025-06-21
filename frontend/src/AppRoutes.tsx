import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import Layout from './components/Layout/Layout';
import { DashboardPage } from './pages/Dashboard/DashboardPage';
import SignalsDashboard from './pages/SignalsDashboard/SignalsDashboard';
import AnalyticsPage from './pages/Analytics/AnalyticsPage';
import PortfolioPage from './pages/Portfolio/PortfolioPage';
import SettingsPage from './pages/Settings/SettingsPage';
import AgentsPage from './pages/Agents/AgentsPage';
import AISignalProphetPage from './pages/AISignalProphet/AISignalProphetPage';
import AICommandCenter from './pages/Dashboard/AICommandCenter';

// Lazy load pages
const LazyPortfolioPage = lazy(() => import('./pages/Portfolio/PortfolioPage'));

const LoadingFallback = () => (
  <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
    <CircularProgress />
  </Box>
);

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route element={<Layout />}>
        {/* Main Dashboard */}
        <Route path="/" element={<Navigate to="/signals" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />

        {/* Signals & Trading */}
        <Route path="/signals" element={<SignalsDashboard />} />
        <Route path="/signals/:symbol" element={<SignalsDashboard />} />

        {/* AI Features */}
        <Route path="/ai-command" element={<AICommandCenter />} />
        <Route path="/ai-prophet" element={<AISignalProphetPage />} />
        <Route path="/ai-lab" element={<Navigate to="/ai-prophet" replace />} />

        {/* Analytics & Insights */}
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/agents" element={<AgentsPage />} />

        {/* Portfolio & Settings */}
        <Route path="/portfolio" element={
          <Suspense fallback={<LoadingFallback />}>
            <LazyPortfolioPage />
          </Suspense>
        } />
        <Route path="/settings" element={<SettingsPage />} />

        {/* Catch all - redirect to signals */}
        <Route path="*" element={<Navigate to="/signals" replace />} />
      </Route>
    </Routes>
  );
};

export default AppRoutes; 