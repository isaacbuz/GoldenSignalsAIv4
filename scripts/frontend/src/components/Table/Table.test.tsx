/**
 * Table Component Tests
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Table } from './Table';
import { testIds } from '@/utils/test-utils';

describe('Table', () => {
  it('renders without crashing', () => {
    render(<Table />);
    expect(screen.getByTestId('table')).toBeInTheDocument();
  });

  it('accepts custom data-testid', () => {
    const customTestId = 'custom-test-id';
    render(<Table data-testid={customTestId} />);
    expect(screen.getByTestId(customTestId)).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    render(<Table className={customClass} />);
    const element = screen.getByTestId('table');
    expect(element).toHaveClass(customClass);
  });

  it('renders children', () => {
    const childText = 'Test child content';
    render(
      <Table>
        <span>{childText}</span>
      </Table>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });

  // TODO: Add more specific tests for your component
});
