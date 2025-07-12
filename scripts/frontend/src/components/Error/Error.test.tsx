/**
 * Error Component Tests
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Error } from './Error';
import { testIds } from '@/utils/test-utils';

describe('Error', () => {
  it('renders without crashing', () => {
    render(<Error />);
    expect(screen.getByTestId('error')).toBeInTheDocument();
  });

  it('accepts custom data-testid', () => {
    const customTestId = 'custom-test-id';
    render(<Error data-testid={customTestId} />);
    expect(screen.getByTestId(customTestId)).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    render(<Error className={customClass} />);
    const element = screen.getByTestId('error');
    expect(element).toHaveClass(customClass);
  });

  it('renders children', () => {
    const childText = 'Test child content';
    render(
      <Error>
        <span>{childText}</span>
      </Error>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });

  // TODO: Add more specific tests for your component
});
