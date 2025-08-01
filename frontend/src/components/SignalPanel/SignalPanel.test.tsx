import { render, screen } from '@testing-library/react';
import { SignalPanel } from './SignalPanel';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <SignalPanel {...props} />
    </TestProviders>
  );
};

describe('SignalPanel', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('signalpanel')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('SignalPanel Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('signalpanel');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <SignalPanel>
          <div>{childText}</div>
        </SignalPanel>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
