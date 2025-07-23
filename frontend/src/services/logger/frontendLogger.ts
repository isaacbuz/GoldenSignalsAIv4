import { apiClient } from '../api/apiClient';
import logger from './';


export interface LogEntry {
    level: 'debug' | 'info' | 'warn' | 'error';
    message: string;
    data?: any;
    timestamp: Date;
    source: string;
    userId?: string;
    sessionId?: string;
}

class FrontendLogger {
    private logs: LogEntry[] = [];
    private maxLogs = 1000;
    private batchSize = 10;
    private flushInterval = 30000; // 30 seconds
    private sessionId: string;

    constructor() {
        this.sessionId = this.generateSessionId();
        this.startPeriodicFlush();

        // Capture unhandled errors
        window.addEventListener('error', (event) => {
            this.error('Unhandled Error', {
                message: event.error?.message || event.message,
                stack: event.error?.stack,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno,
            });
        });
    }

    private generateSessionId(): string {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    private startPeriodicFlush() {
        setInterval(() => {
            this.flush();
        }, this.flushInterval);
    }

    debug(message: string, data?: any) {
        this.log('debug', message, data);
    }

    info(message: string, data?: any) {
        this.log('info', message, data);
    }

    warn(message: string, data?: any) {
        this.log('warn', message, data);
    }

    error(message: string, data?: any) {
        this.log('error', message, data);
    }

    logError(message: string, data?: any) {
        this.error(message, data);
    }

    private log(level: LogEntry['level'], message: string, data?: any) {
        const entry: LogEntry = {
            level,
            message,
            data,
            timestamp: new Date(),
            source: 'frontend',
            sessionId: this.sessionId,
        };

        this.logs.push(entry);

        // Keep only recent logs
        if (this.logs.length > this.maxLogs) {
            this.logs = this.logs.slice(-this.maxLogs);
        }

        // Immediately flush critical errors
        if (level === 'error') {
            this.flush();
        }

        // Console output in development
        if (process.env.NODE_ENV === 'development') {
            console[level](message, data);
        }
    }

    private async flush() {
        if (this.logs.length === 0) return;

        const logsToSend = this.logs.splice(0, this.batchSize);

        try {
            // Use a proper request method - create the logs endpoint
            const response = await fetch('/api/logs/frontend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ logs: logsToSend }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            // If sending fails, put logs back
            this.logs.unshift(...logsToSend);
            logger.warn('Failed to send logs to backend:', error);
        }
    }

    getLogs(filter?: { level?: LogEntry['level']; since?: Date }): LogEntry[] {
        let filtered = [...this.logs];

        if (filter?.level) {
            filtered = filtered.filter(log => log.level === filter.level);
        }

        if (filter?.since) {
            filtered = filtered.filter(log => log.timestamp >= filter.since!);
        }

        return filtered;
    }

    clearLogs() {
        this.logs = [];
    }

    exportLogs(): string {
        return JSON.stringify(this.logs, null, 2);
    }
}

export const frontendLogger = new FrontendLogger();
