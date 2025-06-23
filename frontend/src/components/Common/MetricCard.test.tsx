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

  it('renders without change when change prop is not provided', () => {
    const props = {
      title: 'Market Sentiment',
      value: 'Neutral',
      icon: <TrendingUp />,
    };

    render(
      <ThemeProvider theme={theme}>
        <MetricCard {...props} />
      </ThemeProvider>
    );

    // Check if the title is rendered
    expect(screen.getByText('Market Sentiment')).toBeInTheDocument();

    // Check if the value is rendered
    expect(screen.getByText('Neutral')).toBeInTheDocument();
    
    // Check that no percentage is shown
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });
});
