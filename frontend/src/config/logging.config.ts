/**
 * Logging configuration for the frontend application
 * Controls log levels and output formatting
 */

export interface LoggingConfig {
  level: 'error' | 'warn' | 'info' | 'debug' | 'trace';
  enableConsole: boolean;
  enableRemote: boolean;
  remoteEndpoint?: string;
  includeTimestamp: boolean;
  includeStackTrace: boolean;
}

// Default logging configuration
export const defaultLoggingConfig: LoggingConfig = {
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  enableConsole: true,
  enableRemote: process.env.NODE_ENV === 'production',
  remoteEndpoint: process.env.VITE_LOG_ENDPOINT || '/api/v1/logs',
  includeTimestamp: true,
  includeStackTrace: process.env.NODE_ENV !== 'production'
};

// Log level priorities
export const LOG_LEVELS = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
  trace: 4
} as const;

// Get logging config from environment or localStorage
export function getLoggingConfig(): LoggingConfig {
  try {
    // Check localStorage for user preferences
    const stored = localStorage.getItem('logging-config');
    if (stored) {
      return { ...defaultLoggingConfig, ...JSON.parse(stored) };
    }
  } catch (error) {
    // Ignore errors and use defaults
  }

  return defaultLoggingConfig;
}

// Save logging config to localStorage
export function saveLoggingConfig(config: Partial<LoggingConfig>): void {
  try {
    const current = getLoggingConfig();
    const updated = { ...current, ...config };
    localStorage.setItem('logging-config', JSON.stringify(updated));
  } catch (error) {
    // Ignore errors
  }
}
