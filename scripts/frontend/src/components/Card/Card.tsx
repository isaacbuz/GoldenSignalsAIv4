/**
 * Card Component
 *
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './Card.module.css';

export interface CardProps {
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

export const Card: React.FC<CardProps> = ({
  'data-testid': testId = 'card',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('Card');

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

Card.displayName = 'Card';
