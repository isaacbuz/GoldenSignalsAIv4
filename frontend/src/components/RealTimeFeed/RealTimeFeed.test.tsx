import { render, screen } from '@testing-library/react';
import { RealTimeFeed } from './RealTimeFeed';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <RealTimeFeed {...props} />
    </TestProviders>
  );
};

describe('RealTimeFeed', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('realtimefeed')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('RealTimeFeed Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('realtimefeed');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <RealTimeFeed>
          <div>{childText}</div>
        </RealTimeFeed>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
