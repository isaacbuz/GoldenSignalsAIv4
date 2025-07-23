/**
 * Centralized logging service for the frontend application
 * Browser-compatible logger that replaces console.log with structured logging
 */

// Define log levels
export enum LogLevel {
  ERROR = 0,
  WARN = 1,
  INFO = 2,
  DEBUG = 3,
  TRACE = 4
}

// Logger configuration
interface LoggerConfig {
  level: LogLevel;
  enableTimestamp: boolean;
  enableColors: boolean;
}

class Logger {
  private config: LoggerConfig;
  private levelNames = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE'];
  private levelColors = {
    ERROR: '#ff0000',
    WARN: '#ff9800',
    INFO: '#2196f3',
    DEBUG: '#4caf50',
    TRACE: '#9e9e9e'
  };

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = {
      level: process.env.NODE_ENV === 'production' ? LogLevel.INFO : LogLevel.DEBUG,
      enableTimestamp: true,
      enableColors: process.env.NODE_ENV !== 'production',
      ...config
    };

    // In development, intercept console methods
    if (process.env.NODE_ENV !== 'production') {
      this.interceptConsole();
    }
  }

  private interceptConsole() {
    // Store original console methods
    const originalConsole = {
      log: console.log.bind(console),
      error: console.error.bind(console),
      warn: console.warn.bind(console),
      info: console.info.bind(console),
      debug: console.debug.bind(console)
    };

    // Expose original console for debugging
    (window as any).__originalConsole = originalConsole;

    // Override console methods
    console.log = (...args: any[]) => this.info(...args);
    console.error = (...args: any[]) => this.error(...args);
    console.warn = (...args: any[]) => this.warn(...args);
    console.info = (...args: any[]) => this.info(...args);
    console.debug = (...args: any[]) => this.debug(...args);
  }

  private shouldLog(level: LogLevel): boolean {
    return level <= this.config.level;
  }

  private formatMessage(level: LogLevel, message: string, meta?: any): string {
    const levelName = this.levelNames[level];
    const timestamp = this.config.enableTimestamp
      ? `[${new Date().toLocaleTimeString()}] `
      : '';
    const metaString = meta ? ` ${JSON.stringify(meta)}` : '';
    return `${timestamp}${levelName}: ${message}${metaString}`;
  }

  private log(level: LogLevel, message: string, meta?: any) {
    if (!this.shouldLog(level)) return;

    const formattedMessage = this.formatMessage(level, message, meta);
    const levelName = this.levelNames[level];

    if (this.config.enableColors && (window as any).__originalConsole) {
      const color = this.levelColors[levelName as keyof typeof this.levelColors];
      const style = `color: ${color}; font-weight: ${level <= LogLevel.WARN ? 'bold' : 'normal'}`;
      (window as any).__originalConsole.log(`%c${formattedMessage}`, style);
    } else {
      // Fallback to regular console
      const consoleMethod = level === LogLevel.ERROR ? 'error' :
                          level === LogLevel.WARN ? 'warn' :
                          'log';
      (console as any)[consoleMethod](formattedMessage);
    }

    // Send errors to tracking service in production
    if (level === LogLevel.ERROR && process.env.NODE_ENV === 'production' && (window as any).Sentry) {
      (window as any).Sentry.captureMessage(message, {
        level: 'error',
        extra: meta
      });
    }
  }

  error(message: string | Error, meta?: any) {
    if (message instanceof Error) {
      this.log(LogLevel.ERROR, message.message, { ...meta, stack: message.stack });
    } else {
      this.log(LogLevel.ERROR, message, meta);
    }
  }

  warn(message: string, meta?: any) {
    this.log(LogLevel.WARN, message, meta);
  }

  info(message: string, meta?: any) {
    this.log(LogLevel.INFO, message, meta);
  }

  debug(message: string, meta?: any) {
    this.log(LogLevel.DEBUG, message, meta);
  }

  trace(message: string, meta?: any) {
    this.log(LogLevel.TRACE, message, meta);
  }
}

// Create the default logger instance
const logger = new Logger();

// Performance monitoring
export const logPerformance = (operation: string, startTime: number) => {
  const duration = performance.now() - startTime;
  logger.debug(`Performance: ${operation} took ${duration.toFixed(2)}ms`);
};

// Error tracking with context
export const logError = (error: Error, context?: Record<string, any>) => {
  logger.error(error, context);
};

// API call logging
export const logApiCall = (method: string, url: string, data?: any, response?: any) => {
  logger.debug(`API ${method} ${url}`, {
    request: data,
    response: response?.status,
    duration: response?.duration
  });
};

// WebSocket event logging
export const logWebSocketEvent = (event: string, data?: any) => {
  logger.debug(`WebSocket ${event}`, data);
};

// Trading action logging
export const logTradingAction = (action: string, details: Record<string, any>) => {
  logger.info(`Trading Action: ${action}`, details);
};

// Chart event logging
export const logChartEvent = (event: string, details?: Record<string, any>) => {
  logger.debug(`Chart Event: ${event}`, details);
};

// Create logger instances for different modules
export const createModuleLogger = (moduleName: string) => {
  return {
    error: (message: string, meta?: any) => logger.error(`[${moduleName}] ${message}`, meta),
    warn: (message: string, meta?: any) => logger.warn(`[${moduleName}] ${message}`, meta),
    info: (message: string, meta?: any) => logger.info(`[${moduleName}] ${message}`, meta),
    debug: (message: string, meta?: any) => logger.debug(`[${moduleName}] ${message}`, meta),
    trace: (message: string, meta?: any) => logger.trace(`[${moduleName}] ${message}`, meta)
  };
};

// Export the main logger
export default logger;
