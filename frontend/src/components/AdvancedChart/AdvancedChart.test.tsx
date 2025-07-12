import { render, screen } from '@testing-library/react';
import { AdvancedChart } from './AdvancedChart';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <AdvancedChart {...props} />
    </TestProviders>
  );
};

describe('AdvancedChart', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('advancedchart')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('AdvancedChart Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('advancedchart');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <AdvancedChart>
          <div>{childText}</div>
        </AdvancedChart>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});