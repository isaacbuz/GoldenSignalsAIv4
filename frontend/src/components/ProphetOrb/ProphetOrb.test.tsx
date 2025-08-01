import { render, screen } from '@testing-library/react';
import { ProphetOrb } from './ProphetOrb';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <ProphetOrb {...props} />
    </TestProviders>
  );
};

describe('ProphetOrb', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('prophetorb')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('ProphetOrb Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('prophetorb');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <ProphetOrb>
          <div>{childText}</div>
        </ProphetOrb>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});
