import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SignalCard from '../Signals/SignalCard';
import { Signal, SignalType } from '../../types/signals';
import { ThemeProvider, createTheme } from '@mui/material/styles';

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

const mockBuySignal: Signal = {
  id: 'signal-1',
  symbol: 'AAPL',
  type: SignalType.BUY,
  confidence: 0.85,
  price: 150.25,
  timestamp: '2024-06-24T10:30:00Z',
  agentId: 'technical-agent-001',
  agentType: 'technical',
  reasoning: 'Strong bullish momentum detected',
  supportingData: {
    rsi: 28,
    macd: { signal: 1.2, histogram: 0.5 },
    volume: 1250000
  }
};

const mockSellSignal: Signal = {
  id: 'signal-2',
  symbol: 'GOOGL',
  type: SignalType.SELL,
  confidence: 0.72,
  price: 2750.50,
  timestamp: '2024-06-24T10:35:00Z',
  agentId: 'sentiment-agent-001',
  agentType: 'sentiment',
  reasoning: 'Negative sentiment detected in social media',
  supportingData: {
    sentimentScore: -0.65,
    mentionsCount: 450,
    newsCount: 12
  }
};

describe('SignalCard', () => {
  const mockOnClick = jest.fn();
  const mockOnAction = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders buy signal correctly', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} />
    );

    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('$150.25')).toBeInTheDocument();
    expect(screen.getByText('Technical Analysis')).toBeInTheDocument();
  });

  test('renders sell signal correctly', () => {
    renderWithTheme(
      <SignalCard signal={mockSellSignal} onClick={mockOnClick} />
    );

    expect(screen.getByText('GOOGL')).toBeInTheDocument();
    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.getByText('72%')).toBeInTheDocument();
    expect(screen.getByText('$2,750.50')).toBeInTheDocument();
    expect(screen.getByText('Sentiment Analysis')).toBeInTheDocument();
  });

  test('applies correct styling for buy signal', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} />
    );

    const signalChip = screen.getByText('BUY').closest('.MuiChip-root');
    expect(signalChip).toHaveClass('MuiChip-colorSuccess');
  });

  test('applies correct styling for sell signal', () => {
    renderWithTheme(
      <SignalCard signal={mockSellSignal} onClick={mockOnClick} />
    );

    const signalChip = screen.getByText('SELL').closest('.MuiChip-root');
    expect(signalChip).toHaveClass('MuiChip-colorError');
  });

  test('handles click event', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} />
    );

    const card = screen.getByRole('article');
    fireEvent.click(card);

    expect(mockOnClick).toHaveBeenCalledWith(mockBuySignal);
  });

  test('displays confidence indicator correctly', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} />
    );

    const confidenceBar = screen.getByRole('progressbar');
    expect(confidenceBar).toHaveAttribute('aria-valuenow', '85');
    expect(confidenceBar).toHaveAttribute('aria-label', 'Signal confidence: 85%');
  });

  test('shows action buttons when provided', () => {
    renderWithTheme(
      <SignalCard
        signal={mockBuySignal}
        onClick={mockOnClick}
        onAction={mockOnAction}
        showActions
      />
    );

    expect(screen.getByRole('button', { name: /Execute/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Dismiss/i })).toBeInTheDocument();
  });

  test('handles execute action', async () => {
    renderWithTheme(
      <SignalCard
        signal={mockBuySignal}
        onClick={mockOnClick}
        onAction={mockOnAction}
        showActions
      />
    );

    const executeButton = screen.getByRole('button', { name: /Execute/i });
    fireEvent.click(executeButton);

    expect(mockOnAction).toHaveBeenCalledWith('execute', mockBuySignal);
  });

  test('handles dismiss action', async () => {
    renderWithTheme(
      <SignalCard
        signal={mockBuySignal}
        onClick={mockOnClick}
        onAction={mockOnAction}
        showActions
      />
    );

    const dismissButton = screen.getByRole('button', { name: /Dismiss/i });
    fireEvent.click(dismissButton);

    expect(mockOnAction).toHaveBeenCalledWith('dismiss', mockBuySignal);
  });

  test('prevents card click when action button is clicked', () => {
    renderWithTheme(
      <SignalCard
        signal={mockBuySignal}
        onClick={mockOnClick}
        onAction={mockOnAction}
        showActions
      />
    );

    const executeButton = screen.getByRole('button', { name: /Execute/i });
    fireEvent.click(executeButton);

    expect(mockOnAction).toHaveBeenCalled();
    expect(mockOnClick).not.toHaveBeenCalled();
  });

  test('formats timestamp correctly', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} />
    );

    // Should show relative time
    expect(screen.getByText(/ago/i)).toBeInTheDocument();
  });

  test('shows reasoning when expanded', async () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} expandable />
    );

    const expandButton = screen.getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    await waitFor(() => {
      expect(screen.getByText('Strong bullish momentum detected')).toBeInTheDocument();
    });
  });

  test('displays supporting data when available', async () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} expandable />
    );

    const expandButton = screen.getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    await waitFor(() => {
      expect(screen.getByText(/RSI: 28/i)).toBeInTheDocument();
      expect(screen.getByText(/Volume: 1.25M/i)).toBeInTheDocument();
    });
  });

  test('handles missing optional data gracefully', () => {
    const minimalSignal: Signal = {
      id: 'signal-3',
      symbol: 'TSLA',
      type: SignalType.BUY,
      confidence: 0.65,
      price: 250.00,
      timestamp: new Date().toISOString(),
      agentId: 'agent-001',
      agentType: 'technical'
    };

    renderWithTheme(
      <SignalCard signal={minimalSignal} onClick={mockOnClick} />
    );

    expect(screen.getByText('TSLA')).toBeInTheDocument();
    expect(screen.queryByText(/reasoning/i)).not.toBeInTheDocument();
  });

  test('highlights high confidence signals', () => {
    const highConfidenceSignal = { ...mockBuySignal, confidence: 0.95 };

    renderWithTheme(
      <SignalCard signal={highConfidenceSignal} onClick={mockOnClick} />
    );

    const card = screen.getByRole('article');
    expect(card).toHaveClass('high-confidence');
  });

  test('shows warning for low confidence signals', () => {
    const lowConfidenceSignal = { ...mockBuySignal, confidence: 0.45 };

    renderWithTheme(
      <SignalCard signal={lowConfidenceSignal} onClick={mockOnClick} />
    );

    expect(screen.getByTitle(/Low confidence signal/i)).toBeInTheDocument();
  });

  test('handles keyboard navigation', async () => {
    const user = userEvent.setup();

    renderWithTheme(
      <SignalCard
        signal={mockBuySignal}
        onClick={mockOnClick}
        showActions
        onAction={mockOnAction}
      />
    );

    // Tab to card
    await user.tab();
    expect(screen.getByRole('article')).toHaveFocus();

    // Press Enter to click
    await user.keyboard('{Enter}');
    expect(mockOnClick).toHaveBeenCalled();

    // Tab to execute button
    await user.tab();
    expect(screen.getByRole('button', { name: /Execute/i })).toHaveFocus();
  });

  test('shows agent icon based on type', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={mockOnClick} />
    );

    expect(screen.getByTestId('TrendingUpIcon')).toBeInTheDocument();
  });

  test('handles hold signal type', () => {
    const holdSignal = { ...mockBuySignal, type: SignalType.HOLD };

    renderWithTheme(
      <SignalCard signal={holdSignal} onClick={mockOnClick} />
    );

    expect(screen.getByText('HOLD')).toBeInTheDocument();
    const signalChip = screen.getByText('HOLD').closest('.MuiChip-root');
    expect(signalChip).toHaveClass('MuiChip-colorDefault');
  });

  test('shows status badge when signal is executed', () => {
    const executedSignal = { ...mockBuySignal, status: 'executed' };

    renderWithTheme(
      <SignalCard signal={executedSignal} onClick={mockOnClick} />
    );

    expect(screen.getByText('Executed')).toBeInTheDocument();
    expect(screen.getByTestId('CheckCircleIcon')).toBeInTheDocument();
  });

  test('applies disabled styling for dismissed signals', () => {
    const dismissedSignal = { ...mockBuySignal, status: 'dismissed' };

    renderWithTheme(
      <SignalCard signal={dismissedSignal} onClick={mockOnClick} />
    );

    const card = screen.getByRole('article');
    expect(card).toHaveClass('dismissed');
    expect(card).toHaveStyle({ opacity: '0.6' });
  });
});

describe('SignalCard Accessibility', () => {
  test('has proper ARIA attributes', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={jest.fn()} />
    );

    const card = screen.getByRole('article');
    expect(card).toHaveAttribute('aria-label', 'Trading signal for AAPL');

    const confidenceBar = screen.getByRole('progressbar');
    expect(confidenceBar).toHaveAttribute('aria-label', 'Signal confidence: 85%');
  });

  test('announces signal type to screen readers', () => {
    renderWithTheme(
      <SignalCard signal={mockBuySignal} onClick={jest.fn()} />
    );

    const signalType = screen.getByText('BUY');
    expect(signalType).toHaveAttribute('role', 'status');
    expect(signalType).toHaveAttribute('aria-live', 'polite');
  });

  test('provides descriptive button labels', () => {
    renderWithTheme(
      <SignalCard
        signal={mockBuySignal}
        onClick={jest.fn()}
        onAction={jest.fn()}
        showActions
      />
    );

    expect(screen.getByRole('button', { name: /Execute BUY signal for AAPL/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Dismiss BUY signal for AAPL/i })).toBeInTheDocument();
  });
});
