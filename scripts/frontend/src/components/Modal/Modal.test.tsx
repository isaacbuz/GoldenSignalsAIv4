/**
 * Modal Component Tests
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Modal } from './Modal';
import { testIds } from '@/utils/test-utils';

describe('Modal', () => {
  it('renders without crashing', () => {
    render(<Modal />);
    expect(screen.getByTestId('modal')).toBeInTheDocument();
  });

  it('accepts custom data-testid', () => {
    const customTestId = 'custom-test-id';
    render(<Modal data-testid={customTestId} />);
    expect(screen.getByTestId(customTestId)).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    render(<Modal className={customClass} />);
    const element = screen.getByTestId('modal');
    expect(element).toHaveClass(customClass);
  });

  it('renders children', () => {
    const childText = 'Test child content';
    render(
      <Modal>
        <span>{childText}</span>
      </Modal>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });

  // TODO: Add more specific tests for your component
});
