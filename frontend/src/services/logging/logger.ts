/**
 * Frontend Logging Service
 * Captures all frontend errors and logs them to files in dev/test environments
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
    timestamp: string;
    level: LogLevel;
    component?: string;
    message: string;
    data?: any;
    stack?: string;
    userAgent: string;
    url: string;
}

interface LoggerConfig {
    enableFileLogging: boolean;
    logLevel: LogLevel;
    bufferSize: number;
    flushInterval: number;
}

class FrontendLogger {
    private buffer: LogEntry[] = [];
    private config: LoggerConfig;
    private flushTimer?: NodeJS.Timeout;
    private componentStack: string[] = [];

    constructor(config?: Partial<LoggerConfig>) {
        this.config = {
            enableFileLogging: process.env.NODE_ENV !== 'production',
            logLevel: (process.env.VITE_LOG_LEVEL as LogLevel) || 'info',
            bufferSize: Number(process.env.VITE_LOG_BUFFER_SIZE) || 100,
            flushInterval: 5000, // 5 seconds
            ...config
        };

        this.setupErrorHandlers();
        this.startPeriodicFlush();
    }

    private setupErrorHandlers() {
        // Capture uncaught errors
        window.addEventListener('error', (event) => {
            this.error('Uncaught error', {
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno,
                error: event.error
            });
        });

        // Capture unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.error('Unhandled promise rejection', {
                reason: event.reason,
                promise: event.promise
            });
        });

        // Store logs for E2E tests
        if (window.location.search.includes('e2e=true') || process.env.NODE_ENV === 'test') {
            (window as any).__frontendLogs = this.buffer;
        }
    }

    private startPeriodicFlush() {
        if (this.config.enableFileLogging) {
            this.flushTimer = setInterval(() => {
                this.flushToFile();
            }, this.config.flushInterval);
        }
    }

    private shouldLog(level: LogLevel): boolean {
        const levels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
        const currentLevelIndex = levels.indexOf(this.config.logLevel);
        const messageLevelIndex = levels.indexOf(level);
        return messageLevelIndex >= currentLevelIndex;
    }

    private createLogEntry(level: LogLevel, message: string, data?: any): LogEntry {
        const error = new Error();
        return {
            timestamp: new Date().toISOString(),
            level,
            message,
            data,
            stack: error.stack,
            userAgent: navigator.userAgent,
            url: window.location.href,
            component: this.getCurrentComponent()
        };
    }

    private getCurrentComponent(): string {
        return this.componentStack[this.componentStack.length - 1] || 'Unknown';
    }

    public pushComponent(component: string) {
        this.componentStack.push(component);
    }

    public popComponent() {
        this.componentStack.pop();
    }

    public debug(message: string, data?: any) {
        this.log('debug', message, data);
    }

    public info(message: string, data?: any) {
        this.log('info', message, data);
    }

    public warn(message: string, data?: any) {
        this.log('warn', message, data);
    }

    public error(message: string, data?: any) {
        this.log('error', message, data);
    }

    private log(level: LogLevel, message: string, data?: any) {
        if (!this.shouldLog(level)) return;

        const entry = this.createLogEntry(level, message, data);

        // Add to buffer
        this.buffer.push(entry);

        // Trim buffer if needed
        if (this.buffer.length > this.config.bufferSize) {
            this.buffer = this.buffer.slice(-this.config.bufferSize);
        }

        // Console output in development
        if (process.env.NODE_ENV === 'development') {
            const consoleMethod = level === 'debug' ? 'log' : level;
            console[consoleMethod](`[${entry.component}]`, message, data || '');
        }

        // Immediate flush for errors
        if (level === 'error' && this.config.enableFileLogging) {
            this.flushToFile();
        }
    }

    private async flushToFile() {
        if (!this.config.enableFileLogging || this.buffer.length === 0) return;

        try {
            const response = await fetch('/api/logs/frontend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    level: 'info',
                    message: 'Frontend logs batch',
                    data: {
                        logs: this.buffer,
                        sessionId: this.getSessionId(),
                        timestamp: new Date().toISOString()
                    }
                })
            });

            if (response.ok) {
                // Clear buffer after successful flush
                this.buffer = [];

                // Keep reference for E2E tests
                if ((window as any).__frontendLogs) {
                    (window as any).__frontendLogs = this.buffer;
                }
            }
        } catch (error) {
            // Don't log flush errors to avoid infinite loop
            console.error('Failed to flush logs:', error);
        }
    }

    private getSessionId(): string {
        let sessionId = sessionStorage.getItem('logger_session_id');
        if (!sessionId) {
            sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            sessionStorage.setItem('logger_session_id', sessionId);
        }
        return sessionId;
    }

    public async downloadLogs() {
        const logs = {
            session: this.getSessionId(),
            entries: this.buffer,
            timestamp: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `frontend-logs-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    public clearLogs() {
        this.buffer = [];
        if ((window as any).__frontendLogs) {
            (window as any).__frontendLogs = [];
        }
    }

    public getRecentErrors(count: number = 10): LogEntry[] {
        return this.buffer
            .filter(entry => entry.level === 'error')
            .slice(-count);
    }

    public destroy() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
        }
        this.flushToFile();
    }
}

// Create singleton instance
export const logger = new FrontendLogger();

// Export for testing
export { FrontendLogger };

// React component logging hook
import { useEffect } from 'react';

export function useComponentLogger(componentName: string) {
    useEffect(() => {
        logger.pushComponent(componentName);
        logger.debug(`${componentName} mounted`);

        return () => {
            logger.debug(`${componentName} unmounted`);
            logger.popComponent();
        };
    }, [componentName]);

    return logger;
} 