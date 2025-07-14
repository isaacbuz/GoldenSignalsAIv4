import { render, screen } from '@testing-library/react';
import { CentralChart } from './CentralChart';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <CentralChart {...props} />
    </TestProviders>
  );
};

describe('CentralChart', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('centralchart')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('CentralChart Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('centralchart');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <CentralChart>
          <div>{childText}</div>
        </CentralChart>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});