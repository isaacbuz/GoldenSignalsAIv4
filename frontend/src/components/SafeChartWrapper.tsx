import React, { Component, ReactNode } from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import { styled } from '@mui/material/styles';

const ErrorContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  margin: theme.spacing(2),
  backgroundColor: '#f5f5f5',
  borderRadius: theme.spacing(1),
  textAlign: 'center',
}));

const ErrorTitle = styled(Typography)({
  color: '#d32f2f',
  marginBottom: '16px',
  fontWeight: 600,
});

const ErrorMessage = styled(Typography)({
  color: '#666',
  marginBottom: '24px',
  fontFamily: 'monospace',
  fontSize: '14px',
  backgroundColor: '#fff',
  padding: '12px',
  borderRadius: '4px',
  border: '1px solid #e0e0e0',
  textAlign: 'left',
  maxWidth: '600px',
  margin: '0 auto 24px',
  wordBreak: 'break-word',
});

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class SafeChartWrapper extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Chart component error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return <>{this.props.fallback}</>;
      }

      return (
        <ErrorContainer elevation={0}>
          <ErrorTitle variant="h5">
            Chart Loading Error
          </ErrorTitle>
          <ErrorMessage>
            {this.state.error?.message || 'An unexpected error occurred'}
          </ErrorMessage>
          <Button
            variant="contained"
            color="primary"
            onClick={this.handleReset}
            sx={{
              backgroundColor: '#00c853',
              '&:hover': { backgroundColor: '#00a041' }
            }}
          >
            Try Again
          </Button>
        </ErrorContainer>
      );
    }

    return this.props.children;
  }
}

export default SafeChartWrapper;
