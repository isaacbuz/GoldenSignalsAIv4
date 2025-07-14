import { render, screen } from '@testing-library/react';
import { OptionsPanel } from './OptionsPanel';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <OptionsPanel {...props} />
    </TestProviders>
  );
};

describe('OptionsPanel', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('optionspanel')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('OptionsPanel Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('optionspanel');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <OptionsPanel>
          <div>{childText}</div>
        </OptionsPanel>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});