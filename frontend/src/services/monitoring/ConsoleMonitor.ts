/**
 * Console Monitor Service
 * Captures and logs console errors, warnings, and other messages
 */

import React from 'react';
import { logger } from '../logging/logger';

export interface LogEntry {
    timestamp: number;
    level: 'log' | 'warn' | 'error' | 'info' | 'debug';
    message: string;
    data?: any;
    stack?: string;
    url?: string;
    line?: number;
    column?: number;
}

interface LogFilter {
    level?: 'log' | 'warn' | 'error' | 'info' | 'debug';
    since?: number;
    limit?: number;
    search?: string;
}

class ConsoleMonitor {
    private logs: LogEntry[] = [];
    private maxLogs = 1000;
    private originalConsole: any = {};
    private isMonitoring = false;

    constructor() {
        // Check if we're in development mode
        const isDev = typeof window !== 'undefined' &&
            (window.location.hostname === 'localhost' ||
                window.location.hostname === '127.0.0.1' ||
                window.location.hostname.includes('dev'));

        if (isDev) {
            this.startMonitoring();
        }
    }

    startMonitoring() {
        if (this.isMonitoring) return;

        // Store original console methods
        this.originalConsole = {
            log: console.log,
            warn: console.warn,
            error: console.error,
            info: console.info,
            debug: console.debug
        };

        // Override console methods
        console.log = this.createLogWrapper('log');
        console.warn = this.createLogWrapper('warn');
        console.error = this.createLogWrapper('error');
        console.info = this.createLogWrapper('info');
        console.debug = this.createLogWrapper('debug');

        // Monitor unhandled errors
        window.addEventListener('error', this.handleError.bind(this));
        window.addEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));

        this.isMonitoring = true;
    }

    stopMonitoring() {
        if (!this.isMonitoring) return;

        // Restore original console methods
        Object.assign(console, this.originalConsole);

        // Remove error listeners
        window.removeEventListener('error', this.handleError.bind(this));
        window.removeEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));

        this.isMonitoring = false;
    }

    private createLogWrapper(level: LogEntry['level']) {
        return (...args: any[]) => {
            // Call original console method
            this.originalConsole[level].apply(console, args);

            // Store log entry
            const entry: LogEntry = {
                timestamp: Date.now(),
                level,
                message: args.map(arg =>
                    typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
                ).join(' '),
                data: args.length > 1 ? args.slice(1) : undefined
            };

            this.addLogEntry(entry);
        };
    }

    private handleError(event: ErrorEvent) {
        const entry: LogEntry = {
            timestamp: Date.now(),
            level: 'error',
            message: event.message,
            stack: event.error?.stack,
            url: event.filename,
            line: event.lineno,
            column: event.colno
        };

        this.addLogEntry(entry);
    }

    private handleUnhandledRejection(event: PromiseRejectionEvent) {
        const entry: LogEntry = {
            timestamp: Date.now(),
            level: 'error',
            message: `Unhandled Promise Rejection: ${event.reason}`,
            data: event.reason
        };

        this.addLogEntry(entry);
    }

    private addLogEntry(entry: LogEntry) {
        // Prevent infinite loops by not logging console monitor errors
        if (entry.message.includes('Console Monitor Error') ||
            entry.message.includes('Failed to send logs to backend') ||
            entry.message.includes('Failed to flush logs') ||
            entry.message.includes('WebSocket') ||
            entry.message.includes('Network timeout') ||
            entry.message.includes('CORS')) {
            return;
        }

        this.logs.push(entry);

        // Keep only the most recent logs
        if (this.logs.length > this.maxLogs) {
            this.logs = this.logs.slice(-this.maxLogs);
        }

        // Send critical errors to backend (but not console monitor errors)
        if (entry.level === 'error' && 
            !entry.message.includes('WebSocket') &&
            !entry.message.includes('Network timeout') &&
            !entry.message.includes('CORS') &&
            !entry.message.includes('Failed to fetch')) {
            this.sendToBackend([entry]);
        }
    }

    private async sendToBackend(logs: LogEntry[]) {
        try {
            // Use the correct backend API endpoint with a timeout
            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Timeout')), 2000)
            );

            const response = await Promise.race([
                fetch('/api/logs/frontend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        level: 'error',
                        message: 'Console Monitor Error',
                        data: {
                            logs,
                            timestamp: Date.now(),
                            userAgent: navigator.userAgent,
                            url: window.location.href
                        }
                    })
                }),
                timeoutPromise
            ]) as Response;

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            // Silently fail to prevent infinite loops
            // Only log in development mode
            if (process.env.NODE_ENV === 'development') {
                this.originalConsole.warn('Failed to send logs to backend:', error);
            }
        }
    }

    getLogs(filter: LogFilter = {}): LogEntry[] {
        let filtered = [...this.logs];

        if (filter.level) {
            filtered = filtered.filter(entry => entry.level === filter.level);
        }

        if (filter.since) {
            filtered = filtered.filter(entry => entry.timestamp >= filter.since!);
        }

        if (filter.search) {
            const search = filter.search.toLowerCase();
            filtered = filtered.filter(entry =>
                entry.message.toLowerCase().includes(search)
            );
        }

        if (filter.limit) {
            filtered = filtered.slice(-filter.limit);
        }

        return filtered;
    }

    getStats() {
        const stats = {
            total: this.logs.length,
            byLevel: {} as Record<string, number>,
            recentErrors: 0
        };

        const oneHourAgo = Date.now() - (60 * 60 * 1000);

        this.logs.forEach(entry => {
            stats.byLevel[entry.level] = (stats.byLevel[entry.level] || 0) + 1;

            if (entry.level === 'error' && entry.timestamp > oneHourAgo) {
                stats.recentErrors++;
            }
        });

        return stats;
    }

    exportLogs(): string {
        return JSON.stringify(this.logs, null, 2);
    }

    clearLogs() {
        this.logs = [];
    }
}

export const consoleMonitor = new ConsoleMonitor();

// React hook for using console monitor
export function useConsoleMonitor() {
    const [logs, setLogs] = React.useState<LogEntry[]>([]);
    const [stats, setStats] = React.useState(consoleMonitor.getStats());

    React.useEffect(() => {
        // Update logs and stats periodically
        const interval = setInterval(() => {
            setLogs(consoleMonitor.getLogs());
            setStats(consoleMonitor.getStats());
        }, 1000);

        // Get initial logs
        setLogs(consoleMonitor.getLogs());

        return () => clearInterval(interval);
    }, []);

    return {
        logs,
        stats,
        clearLogs: () => consoleMonitor.clearLogs(),
        getLogs: (filter?: Parameters<typeof consoleMonitor.getLogs>[0]) =>
            consoleMonitor.getLogs(filter)
    };
} 