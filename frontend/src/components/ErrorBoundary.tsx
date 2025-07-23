import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Button, Typography, Paper, Stack } from '@mui/material';
import { Refresh, Warning } from '@mui/icons-material';
import logger from '../services/logger';


interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Only log errors in development mode to prevent console spam
    if (process.env.NODE_ENV === 'development') {
      logger.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    this.setState({ errorInfo });

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log to error reporting service in production
    if (process.env.NODE_ENV === 'production') {
      // Send to error tracking service (e.g., Sentry, LogRocket)
      // Only log critical errors to avoid console spam
      if (error.name !== 'ChunkLoadError' && error.name !== 'NetworkError') {
        logger.error('Production error:', {
          error: error.toString(),
          stack: errorInfo.componentStack,
          timestamp: new Date().toISOString(),
        });
      }
    }
  }

  private handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  public render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return <>{this.props.fallback}</>;
      }

      // Default error UI
      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '400px',
            p: 3,
          }}
        >
          <Paper
            elevation={3}
            sx={{
              p: 4,
              maxWidth: 600,
              textAlign: 'center',
              borderRadius: 2,
            }}
          >
            <Stack spacing={3} alignItems="center">
              <Warning sx={{ fontSize: 64, color: 'warning.main' }} />

              <Typography variant="h5" gutterBottom>
                Oops! Something went wrong
              </Typography>

              <Typography variant="body1" color="text.secondary" paragraph>
                We encountered an unexpected error. The issue has been logged and our team will investigate.
              </Typography>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    bgcolor: 'grey.100',
                    maxWidth: '100%',
                    overflow: 'auto',
                  }}
                >
                  <Typography variant="caption" component="pre" sx={{ textAlign: 'left' }}>
                    {this.state.error.toString()}
                    {this.state.errorInfo && this.state.errorInfo.componentStack}
                  </Typography>
                </Paper>
              )}

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<Refresh />}
                  onClick={this.handleReset}
                >
                  Try Again
                </Button>

                <Button
                  variant="outlined"
                  onClick={() => window.location.href = '/'}
                >
                  Go to Dashboard
                </Button>
              </Stack>
            </Stack>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
