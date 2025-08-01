import { render, screen } from '@testing-library/react';
import { FloatingOrbAssistant } from './FloatingOrbAssistant';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <FloatingOrbAssistant {...props} />
    </TestProviders>
  );
};

describe('FloatingOrbAssistant', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('floatingorbassistant')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('FloatingOrbAssistant Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('floatingorbassistant');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <FloatingOrbAssistant>
          <div>{childText}</div>
        </FloatingOrbAssistant>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
