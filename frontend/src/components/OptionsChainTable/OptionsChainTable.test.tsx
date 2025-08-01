import { render, screen } from '@testing-library/react';
import { OptionsChainTable } from './OptionsChainTable';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <OptionsChainTable {...props} />
    </TestProviders>
  );
};

describe('OptionsChainTable', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('optionschaintable')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('OptionsChainTable Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('optionschaintable');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <OptionsChainTable>
          <div>{childText}</div>
        </OptionsChainTable>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
