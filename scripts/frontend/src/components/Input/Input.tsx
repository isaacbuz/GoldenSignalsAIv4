/**
 * Input Component
 *
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './Input.module.css';

export interface InputProps {
  /**
   * Component ID for testing
   */
  'data-testid'?: string;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Children elements
   */
  children?: React.ReactNode;
  // TODO: Add more props
}

export const Input: React.FC<InputProps> = ({
  'data-testid': testId = 'input',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('Input');

  // Log render
  logger.debug('Rendering', { props });

  return (
    <div
      className={`${styles.container} ${className || ''}`}
      data-testid={testId}
      {...props}
    >
      {children}
    </div>
  );
};

Input.displayName = 'Input';
