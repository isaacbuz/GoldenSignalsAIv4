import { render, screen } from '@testing-library/react';
import { AnalysisLegend } from './AnalysisLegend';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <AnalysisLegend {...props} />
    </TestProviders>
  );
};

describe('AnalysisLegend', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('analysislegend')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('AnalysisLegend Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('analysislegend');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <AnalysisLegend>
          <div>{childText}</div>
        </AnalysisLegend>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});