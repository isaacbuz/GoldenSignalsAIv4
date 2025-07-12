/**
 * Table Component
 * 
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './Table.module.css';

export interface TableProps {
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

export const Table: React.FC<TableProps> = ({
  'data-testid': testId = 'table',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('Table');

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

Table.displayName = 'Table';
