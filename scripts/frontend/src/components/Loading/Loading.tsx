/**
 * Loading Component
 * 
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './Loading.module.css';

export interface LoadingProps {
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

export const Loading: React.FC<LoadingProps> = ({
  'data-testid': testId = 'loading',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('Loading');

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

Loading.displayName = 'Loading';
