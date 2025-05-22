import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import Scanner from '../Scanner';
import GrokFeedback from '../../components/GrokFeedback';

jest.mock('../../components/GrokFeedback', () => ({ feedback, loading }) => (
  <div data-testid="grok-feedback">{loading ? 'Analyzing...' : (feedback || []).join(', ')}</div>
));

describe('Scanner + Grok integration', () => {
  it('renders GrokFeedback with suggestions after mock backtest', async () => {
    // Simulate the Scanner page logic that would call Grok and receive feedback
    // This is a placeholder for actual orchestrator/backend integration
    render(<Scanner />);
    // Simulate feedback
    const feedback = ['Use a tighter stop loss', 'Add RSI filter'];
    // Wait for GrokFeedback to appear
    await waitFor(() => {
      expect(screen.getByTestId('grok-feedback')).toHaveTextContent('RSI');
    });
  });
});
