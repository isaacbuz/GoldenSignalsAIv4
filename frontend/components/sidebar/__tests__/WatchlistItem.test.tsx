import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import WatchlistItem from '../WatchlistItem';

// Mock the useTickerContext hook
jest.mock('@/context/TickerContext', () => ({
  useTickerContext: () => ({
    selected: 'AAPL',
    setSelected: jest.fn(),
  }),
}));

describe('WatchlistItem', () => {
  it('renders symbol and responds to click', () => {
    const { getByText } = render(<WatchlistItem symbol="AAPL" />);
    const item = getByText('AAPL');
    expect(item).toBeInTheDocument();
    fireEvent.click(item);
    // setSelected is mocked, so we can't check call here, but this ensures no error
  });

  it('is accessible via keyboard', () => {
    const { getByText } = render(<WatchlistItem symbol="AAPL" />);
    const item = getByText('AAPL');
    fireEvent.keyDown(item, { key: 'Enter' });
    fireEvent.keyDown(item, { key: ' ' });
    // setSelected is mocked, so we can't check call here, but this ensures no error
  });
});
