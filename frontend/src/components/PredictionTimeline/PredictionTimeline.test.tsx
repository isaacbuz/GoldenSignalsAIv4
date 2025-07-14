import { render, screen } from '@testing-library/react';
import { PredictionTimeline } from './PredictionTimeline';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <PredictionTimeline {...props} />
    </TestProviders>
  );
};

describe('PredictionTimeline', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('predictiontimeline')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('PredictionTimeline Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('predictiontimeline');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <PredictionTimeline>
          <div>{childText}</div>
        </PredictionTimeline>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});