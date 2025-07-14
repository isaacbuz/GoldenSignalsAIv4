import { render, screen } from '@testing-library/react';
import { FloatingOrb } from './FloatingOrb';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <FloatingOrb {...props} />
    </TestProviders>
  );
};

describe('FloatingOrb', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('floatingorb')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('FloatingOrb Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('floatingorb');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <FloatingOrb>
          <div>{childText}</div>
        </FloatingOrb>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});