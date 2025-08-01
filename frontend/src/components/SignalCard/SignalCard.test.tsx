import { render, screen } from '@testing-library/react';
import { SignalCard } from './SignalCard';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <SignalCard {...props} />
    </TestProviders>
  );
};

describe('SignalCard', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('signalcard')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('SignalCard Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('signalcard');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <SignalCard>
          <div>{childText}</div>
        </SignalCard>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
