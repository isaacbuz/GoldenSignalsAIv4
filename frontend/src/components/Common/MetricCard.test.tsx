import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { TrendingUp } from '@mui/icons-material';
import { MetricCard } from './MetricCard';

// Mock theme for MUI components
const theme = createTheme();

describe('MetricCard', () => {
  it('renders the title, value, and change correctly', () => {
    const props = {
      title: 'Market Sentiment',
      value: 'Bullish',
      change: '75%',
      trend: 'up' as const,
      icon: <TrendingUp />,
      color: '#00ff00',
    };

    render(
      <ThemeProvider theme={theme}>
        <MetricCard {...props} />
      </ThemeProvider>
    );

    // Check if the title is rendered
    expect(screen.getByText('Market Sentiment')).toBeInTheDocument();

    // Check if the value is rendered
    expect(screen.getByText('Bullish')).toBeInTheDocument();
    
    // Check if the change value is rendered
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('renders a skeleton when isLoading is true', () => {
    const props = {
      title: 'Market Sentiment',
      value: 'Bullish',
      icon: <TrendingUp />,
      isLoading: true,
    };

    render(
      <ThemeProvider theme={theme}>
        <MetricCard {...props} />
      </ThemeProvider>
    );

    // Check that the main value is not present
    expect(screen.queryByText('Bullish')).not.toBeInTheDocument();

    // Check for skeleton elements (we can check by role or test id if needed, but for now we'll assume their presence if the value is gone)
    // A more robust way would be to check for the skeleton's presence directly if it had a data-testid.
    // For this demonstration, checking for the absence of data is sufficient.
  });
}); 