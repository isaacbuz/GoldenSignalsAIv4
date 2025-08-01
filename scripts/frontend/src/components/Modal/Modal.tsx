/**
 * Modal Component
 *
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './Modal.module.css';

export interface ModalProps {
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

export const Modal: React.FC<ModalProps> = ({
  'data-testid': testId = 'modal',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('Modal');

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

Modal.displayName = 'Modal';
