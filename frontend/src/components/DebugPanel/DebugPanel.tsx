/**
 * DebugPanel Component
 *
 * Development tool for viewing frontend logs and errors in real-time
 */

import React, { useState, useEffect } from 'react';
import { useComponentLogger, logger, LogEntry } from '@/services/logging/logger';
import { ENV } from '@/config/environment';
import { testIds } from '@/utils/test-utils';
import styles from './DebugPanel.module.css';

export interface DebugPanelProps {
  /**
   * Component ID for testing
   */
  'data-testid'?: string;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Maximum number of log entries to display
   */
  maxEntries?: number;
  /**
   * Position of the panel
   */
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
}

export const DebugPanel: React.FC<DebugPanelProps> = ({
  'data-testid': testId = 'debug-panel',
  className,
  maxEntries = 50,
  position = 'bottom-right',
  ...props
}) => {
  const componentLogger = useComponentLogger('DebugPanel');
  const [isOpen, setIsOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<'all' | 'error' | 'warn' | 'info' | 'debug'>('all');

  // Only show in development
  if (!ENV.IS_DEVELOPMENT) {
    return null;
  }

  useEffect(() => {
    // Poll for recent logs
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/logs/frontend/recent?limit=' + maxEntries);
        if (response.ok) {
          const recentLogs = await response.json();
          setLogs(recentLogs);
        }
      } catch (error) {
        // Use local logs if API fails
        setLogs(logger.getRecentErrors(maxEntries));
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [maxEntries]);

  const errorCount = logs.filter(log => log.level === 'error').length;
  const warnCount = logs.filter(log => log.level === 'warn').length;

  const filteredLogs = filter === 'all'
    ? logs
    : logs.filter(log => log.level === filter);

  const handleClearLogs = () => {
    logger.clearLogs();
    setLogs([]);
  };

  const handleDownloadLogs = () => {
    logger.downloadLogs();
  };

  return (
    <div
      className={`${styles.container} ${styles[position]} ${className || ''}`}
      data-testid={testId}
      {...props}
    >
      {!isOpen ? (
        <button
          className={styles.toggleButton}
          onClick={() => setIsOpen(true)}
          data-testid={`${testId}-toggle`}
        >
          <span className={styles.icon}>🐛</span>
          Debug
          {errorCount > 0 && (
            <span className={styles.errorBadge}>{errorCount}</span>
          )}
        </button>
      ) : (
        <div className={styles.panel}>
          <div className={styles.header}>
            <h3>Debug Panel</h3>
            <div className={styles.stats}>
              <span className={styles.errorCount}>
                Errors: {errorCount}
              </span>
              <span className={styles.warnCount}>
                Warnings: {warnCount}
              </span>
            </div>
            <button
              className={styles.closeButton}
              onClick={() => setIsOpen(false)}
              data-testid={`${testId}-close`}
            >
              ✕
            </button>
          </div>

          <div className={styles.toolbar}>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as any)}
              className={styles.filterSelect}
              data-testid={`${testId}-filter`}
            >
              <option value="all">All Logs</option>
              <option value="error">Errors Only</option>
              <option value="warn">Warnings Only</option>
              <option value="info">Info Only</option>
              <option value="debug">Debug Only</option>
            </select>

            <button
              onClick={handleClearLogs}
              className={styles.actionButton}
              data-testid={`${testId}-clear`}
            >
              Clear
            </button>

            <button
              onClick={handleDownloadLogs}
              className={styles.actionButton}
              data-testid={`${testId}-download`}
            >
              Download
            </button>
          </div>

          <div className={styles.logContainer}>
            {filteredLogs.length === 0 ? (
              <div className={styles.emptyState}>
                No logs to display
              </div>
            ) : (
              filteredLogs.map((log, index) => (
                <div
                  key={index}
                  className={`${styles.logEntry} ${styles[log.level]}`}
                  data-testid={`${testId}-log-${index}`}
                >
                  <div className={styles.logHeader}>
                    <span className={styles.timestamp}>
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span className={styles.level}>{log.level.toUpperCase()}</span>
                    <span className={styles.component}>{log.component}</span>
                  </div>
                  <div className={styles.message}>{log.message}</div>
                  {log.data && (
                    <pre className={styles.data}>
                      {JSON.stringify(log.data, null, 2)}
                    </pre>
                  )}
                  {log.stack && (
                    <details className={styles.stackTrace}>
                      <summary>Stack Trace</summary>
                      <pre>{log.stack}</pre>
                    </details>
                  )}
                </div>
              ))
            )}
          </div>

          <div className={styles.footer}>
            <div className={styles.info}>
              Environment: {ENV.IS_DEVELOPMENT ? 'Development' : 'Test'}
            </div>
            <div className={styles.info}>
              Log Level: {ENV.LOG_LEVEL}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

DebugPanel.displayName = 'DebugPanel';
