import React from 'react';
import logger from '../services/logger';


// Performance monitoring utilities
export class PerformanceMonitor {
    private static instance: PerformanceMonitor;
    private longTaskThreshold = 50; // milliseconds
    private performanceObserver: PerformanceObserver | null = null;
    private isMonitoring = false;

    private constructor() { }

    public static getInstance(): PerformanceMonitor {
        if (!PerformanceMonitor.instance) {
            PerformanceMonitor.instance = new PerformanceMonitor();
        }
        return PerformanceMonitor.instance;
    }

    public startMonitoring(): void {
        if (this.isMonitoring || typeof window === 'undefined') return;

        try {
            // Monitor long tasks
            if ('PerformanceObserver' in window) {
                this.performanceObserver = new PerformanceObserver((list) => {
                    const entries = list.getEntries();
                    entries.forEach((entry) => {
                        if (entry.duration > this.longTaskThreshold) {
                            if (process.env.NODE_ENV === 'development') {
                                logger.warn(`Long task detected: ${entry.duration.toFixed(2)}ms`, {
                                    name: entry.name,
                                    startTime: entry.startTime,
                                    duration: entry.duration
                                });
                            }
                        }
                    });
                });

                this.performanceObserver.observe({ entryTypes: ['longtask'] });
                this.isMonitoring = true;
            }
        } catch (error) {
            logger.error('Failed to start performance monitoring:', error);
        }
    }

    public stopMonitoring(): void {
        if (this.performanceObserver) {
            this.performanceObserver.disconnect();
            this.performanceObserver = null;
            this.isMonitoring = false;
        }
    }

    public measureFunction<T>(fn: () => T, name: string): T {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        const duration = end - start;

        if (duration > this.longTaskThreshold && process.env.NODE_ENV === 'development') {
            logger.warn(`Function "${name}" took ${duration.toFixed(2)}ms`);
        }

        return result;
    }

    public async measureAsyncFunction<T>(fn: () => Promise<T>, name: string): Promise<T> {
        const start = performance.now();
        const result = await fn();
        const end = performance.now();
        const duration = end - start;

        if (duration > this.longTaskThreshold && process.env.NODE_ENV === 'development') {
            logger.warn(`Async function "${name}" took ${duration.toFixed(2)}ms`);
        }

        return result;
    }

    public setLongTaskThreshold(threshold: number): void {
        this.longTaskThreshold = threshold;
    }
}

// Utility functions for performance optimization
export const debounce = <T extends (...args: any[]) => any>(
    func: T,
    wait: number,
    immediate = false
): ((...args: Parameters<T>) => void) => {
    let timeout: NodeJS.Timeout | null = null;

    return (...args: Parameters<T>) => {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };

        const callNow = immediate && !timeout;

        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(later, wait);

        if (callNow) func(...args);
    };
};

export const throttle = <T extends (...args: any[]) => any>(
    func: T,
    limit: number
): ((...args: Parameters<T>) => void) => {
    let inThrottle: boolean;

    return (...args: Parameters<T>) => {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};

// React hook for performance monitoring
export const usePerformanceMonitor = (enabled = true) => {
    React.useEffect(() => {
        if (!enabled) return;

        const monitor = PerformanceMonitor.getInstance();
        monitor.startMonitoring();

        return () => {
            monitor.stopMonitoring();
        };
    }, [enabled]);
};

// Utility to break up long tasks
export const yieldToMain = (): Promise<void> => {
    return new Promise(resolve => {
        setTimeout(resolve, 0);
    });
};

// Scheduler for breaking up work
export class TaskScheduler {
    private tasks: Array<() => void> = [];
    private isRunning = false;

    public schedule(task: () => void): void {
        this.tasks.push(task);
        if (!this.isRunning) {
            this.run();
        }
    }

    private async run(): Promise<void> {
        this.isRunning = true;

        while (this.tasks.length > 0) {
            const task = this.tasks.shift();
            if (task) {
                const start = performance.now();
                task();
                const duration = performance.now() - start;

                // If task took more than 5ms, yield to main thread
                if (duration > 5) {
                    await yieldToMain();
                }
            }
        }

        this.isRunning = false;
    }
}

export const taskScheduler = new TaskScheduler();
