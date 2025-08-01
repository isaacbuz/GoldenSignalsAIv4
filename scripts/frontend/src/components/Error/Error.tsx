/**
 * Error Component
 *
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './Error.module.css';

export interface ErrorProps {
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

export const Error: React.FC<ErrorProps> = ({
  'data-testid': testId = 'error',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('Error');

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

Error.displayName = 'Error';
