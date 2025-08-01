import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import Layout from './components/Layout/Layout';
import { DashboardPage } from './pages/Dashboard/DashboardPage';
import { SignalsPageFixed } from './pages/Signals/SignalsPageFixed';
import AnalyticsPage from './pages/Analytics/AnalyticsPage';
import PortfolioPage from './pages/Portfolio/PortfolioPage';
import SettingsPage from './pages/Settings/SettingsPage';
import AgentsPage from './pages/Agents/AgentsPage';
import AISignalProphetPage from './pages/AISignalProphet/AISignalProphetPage';
import AICommandCenter from './pages/Dashboard/AICommandCenter';
import AIChartPage from './pages/AIChart/AIChartPage';
import { WebSocketTest } from './pages/WebSocketTest';
import { LiveChartTest } from './pages/LiveChartTest';
import { SimplePage } from './pages/SimplePage';
import { DebugPage } from './pages/DebugPage';
import { TestDirectRoute } from './pages/TestDirectRoute';
import { TestMinimal } from './pages/TestMinimal';
import { ErrorTest } from './pages/ErrorTest';
import { ChartSelector } from './pages/ChartSelector';

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
      {/* Test route without Layout */}
      <Route path="/test-direct" element={<TestDirectRoute />} />
      <Route path="/test-minimal" element={<TestMinimal />} />
      <Route path="/error-test" element={<ErrorTest />} />
      <Route path="/charts" element={<ChartSelector />} />

      <Route element={<Layout />}>
        {/* Main Dashboard */}
        <Route path="/" element={<Navigate to="/signals" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />

        {/* Signals & Trading */}
        <Route path="/signals" element={<SignalsPageFixed />} />
        <Route path="/signals/:symbol" element={<SignalsPageFixed />} />
        <Route path="/chart" element={<SignalsPageFixed />} />
        <Route path="/chart/:symbol" element={<SignalsPageFixed />} />

        {/* AI Features */}
        <Route path="/ai-command" element={<AICommandCenter />} />
        <Route path="/ai-prophet" element={<AISignalProphetPage />} />
        <Route path="/ai-chart" element={<AIChartPage />} />
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

        {/* Test Routes */}
        <Route path="/test/websocket" element={<WebSocketTest />} />
        <Route path="/test/chart" element={<LiveChartTest />} />
        <Route path="/debug" element={<DebugPage />} />

        {/* Catch all - redirect to signals */}
        <Route path="*" element={<Navigate to="/signals" replace />} />
      </Route>
    </Routes>
  );
};

export default AppRoutes;
