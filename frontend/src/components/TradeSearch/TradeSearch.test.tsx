import { render, screen } from '@testing-library/react';
import { TradeSearch } from './TradeSearch';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <TradeSearch {...props} />
    </TestProviders>
  );
};

describe('TradeSearch', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('tradesearch')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('TradeSearch Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('tradesearch');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <TradeSearch>
          <div>{childText}</div>
        </TradeSearch>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
