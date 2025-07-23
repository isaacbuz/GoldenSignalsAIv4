import React, { Suspense, lazy } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { ErrorBoundary } from 'react-error-boundary';

// Lazy load the chart - using simplified version for now
const AITradingChart = lazy(() => import('./components/AIChart/AITradingChartSimple'));

const ErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = ({ error, resetErrorBoundary }) => {
  console.error('Chart Error:', error);

  return (
    <Box sx={{
      width: '100%',
      height: '100vh',
      backgroundColor: '#000',
      color: '#fff',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      p: 4
    }}>
      <Typography variant="h4" sx={{ color: '#FF3B30', mb: 2 }}>
        Chart Loading Error
      </Typography>
      <Box sx={{
        backgroundColor: '#111',
        p: 3,
        borderRadius: 2,
        maxWidth: '80%',
        mb: 3
      }}>
        <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#ff6b6b' }}>
          {error.message}
        </Typography>
        {error.stack && (
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem', mt: 2, color: '#999' }}>
            {error.stack.split('\n').slice(0, 5).join('\n')}
          </Typography>
        )}
      </Box>
      <button
        onClick={resetErrorBoundary}
        style={{
          padding: '10px 20px',
          backgroundColor: '#007AFF',
          color: '#fff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        Try Again
      </button>
    </Box>
  );
};

const LoadingFallback = () => (
  <Box sx={{
    width: '100%',
    height: '100vh',
    backgroundColor: '#000',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  }}>
    <CircularProgress sx={{ color: '#FFD700' }} />
  </Box>
);

const SafeAITradingChart: React.FC = () => {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <Suspense fallback={<LoadingFallback />}>
        <AITradingChart />
      </Suspense>
    </ErrorBoundary>
  );
};

export default SafeAITradingChart;
