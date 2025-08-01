/**
 * Performance Monitoring Service
 * Tracks component render times, API calls, and other performance metrics
 */

import { logger } from '@/services/logging/logger';
import { ENV } from '@/config/environment';

export interface PerformanceMetric {
    name: string;
    count: number;
    totalDuration: number;
    avgDuration: number;
    maxDuration: number;
    minDuration: number;
    errors: number;
    lastMeasured: string;
}

export interface APIMetric extends PerformanceMetric {
    endpoint: string;
    method: string;
    statusCodes: Record<number, number>;
    errorRate: number;
}

class PerformanceMonitor {
    private metrics: Map<string, PerformanceMetric> = new Map();
    private apiMetrics: Map<string, APIMetric> = new Map();
    private navigationStart: number;
    private observers: PerformanceObserver[] = [];

    constructor() {
        this.navigationStart = performance.now();

        if (ENV.ENABLE_PERFORMANCE_MONITORING) {
            this.initializeObservers();
            this.trackWebVitals();
        }
    }

    private initializeObservers() {
        // Track long tasks
        if ('PerformanceObserver' in window && PerformanceObserver.supportedEntryTypes.includes('longtask')) {
            const longTaskObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    logger.warn('Long task detected', {
                        duration: entry.duration,
                        startTime: entry.startTime,
                        name: entry.name
                    });
                }
            });

            longTaskObserver.observe({ entryTypes: ['longtask'] });
            this.observers.push(longTaskObserver);
        }

        // Track resource timing
        if ('PerformanceObserver' in window && PerformanceObserver.supportedEntryTypes.includes('resource')) {
            const resourceObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    const resourceEntry = entry as PerformanceResourceTiming;
                    if (resourceEntry.initiatorType === 'fetch' || resourceEntry.initiatorType === 'xmlhttprequest') {
                        this.trackResourceTiming(resourceEntry);
                    }
                }
            });

            resourceObserver.observe({ entryTypes: ['resource'] });
            this.observers.push(resourceObserver);
        }
    }

    private trackResourceTiming(entry: PerformanceResourceTiming) {
        const duration = entry.responseEnd - entry.fetchStart;
        const url = new URL(entry.name);
        const endpoint = url.pathname;

        if (duration > 1000) {
            logger.warn(`Slow API call: ${endpoint}`, {
                duration,
                method: entry.initiatorType,
                size: entry.transferSize
            });
        }
    }

    private trackWebVitals() {
        // Track First Contentful Paint (FCP)
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const fcp = entries[entries.length - 1];
            logger.info('First Contentful Paint', {
                value: fcp.startTime,
                rating: fcp.startTime < 1800 ? 'good' : fcp.startTime < 3000 ? 'needs improvement' : 'poor'
            });
        }).observe({ entryTypes: ['paint'] });

        // Track Largest Contentful Paint (LCP)
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lcp = entries[entries.length - 1];
            logger.info('Largest Contentful Paint', {
                value: lcp.startTime,
                element: (lcp as any).element?.tagName,
                rating: lcp.startTime < 2500 ? 'good' : lcp.startTime < 4000 ? 'needs improvement' : 'poor'
            });
        }).observe({ entryTypes: ['largest-contentful-paint'] });
    }

    measureComponent(componentName: string): MethodDecorator {
        return (target: any, propertyKey: string | symbol, descriptor: PropertyDescriptor) => {
            const originalMethod = descriptor.value;

            descriptor.value = function (...args: any[]) {
                const start = performance.now();
                const result = originalMethod.apply(this, args);
                const duration = performance.now() - start;

                this.updateMetric(componentName, duration);

                if (duration > ENV.SLOW_RENDER_THRESHOLD) {
                    logger.warn(`Slow render detected in ${componentName}`, {
                        method: String(propertyKey),
                        duration,
                        threshold: ENV.SLOW_RENDER_THRESHOLD
                    });
                }

                return result;
            };

            return descriptor;
        };
    }

    startMeasure(name: string): () => void {
        const start = performance.now();

        return () => {
            const duration = performance.now() - start;
            this.updateMetric(name, duration);

            if (duration > 100) {
                logger.debug(`Performance measure: ${name}`, { duration });
            }
        };
    }

    private updateMetric(name: string, duration: number, hasError: boolean = false) {
        const existing = this.metrics.get(name) || {
            name,
            count: 0,
            totalDuration: 0,
            avgDuration: 0,
            maxDuration: 0,
            minDuration: Infinity,
            errors: 0,
            lastMeasured: new Date().toISOString()
        };

        existing.count++;
        existing.totalDuration += duration;
        existing.avgDuration = existing.totalDuration / existing.count;
        existing.maxDuration = Math.max(existing.maxDuration, duration);
        existing.minDuration = Math.min(existing.minDuration, duration);
        if (hasError) existing.errors++;
        existing.lastMeasured = new Date().toISOString();

        this.metrics.set(name, existing);
    }

    async trackAPICall(
        endpoint: string,
        method: string,
        duration: number,
        status: number,
        error?: Error
    ) {
        const key = `${method} ${endpoint}`;
        const existing = this.apiMetrics.get(key) || {
            name: key,
            endpoint,
            method,
            count: 0,
            totalDuration: 0,
            avgDuration: 0,
            maxDuration: 0,
            minDuration: Infinity,
            errors: 0,
            statusCodes: {},
            errorRate: 0,
            lastMeasured: new Date().toISOString()
        };

        existing.count++;
        existing.totalDuration += duration;
        existing.avgDuration = existing.totalDuration / existing.count;
        existing.maxDuration = Math.max(existing.maxDuration, duration);
        existing.minDuration = Math.min(existing.minDuration, duration);

        // Track status codes
        existing.statusCodes[status] = (existing.statusCodes[status] || 0) + 1;

        if (error || status >= 400) {
            existing.errors++;
        }

        existing.errorRate = existing.errors / existing.count;
        existing.lastMeasured = new Date().toISOString();

        this.apiMetrics.set(key, existing);

        // Log slow API calls
        if (duration > 1000) {
            logger.warn(`Slow API call to ${endpoint}`, {
                method,
                duration,
                status
            });
        }
    }

    getMetrics(): PerformanceMetric[] {
        return Array.from(this.metrics.values());
    }

    getAPIMetrics(): APIMetric[] {
        return Array.from(this.apiMetrics.values());
    }

    getReport() {
        const metrics = this.getMetrics();
        const apiMetrics = this.getAPIMetrics();

        return {
            summary: {
                totalMetrics: metrics.length,
                totalAPICalls: apiMetrics.reduce((sum, m) => sum + m.count, 0),
                avgAPIResponseTime: apiMetrics.reduce((sum, m) => sum + m.avgDuration, 0) / apiMetrics.length || 0,
                errorRate: apiMetrics.reduce((sum, m) => sum + m.errorRate, 0) / apiMetrics.length || 0,
                uptime: performance.now() - this.navigationStart
            },
            metrics,
            apiMetrics,
            timestamp: new Date().toISOString()
        };
    }

    clearMetrics() {
        this.metrics.clear();
        this.apiMetrics.clear();
    }

    destroy() {
        this.observers.forEach(observer => observer.disconnect());
        this.observers = [];
    }
}

// Create singleton instance
export const performanceMonitor = new PerformanceMonitor();

// React hook for component performance tracking
import { useEffect } from 'react';

export function usePerformanceTracking(componentName: string) {
    useEffect(() => {
        const endMeasure = performanceMonitor.startMeasure(`${componentName}.mount`);

        return () => {
            endMeasure();
            performanceMonitor.startMeasure(`${componentName}.unmount`)();
        };
    }, [componentName]);

    return {
        trackRender: () => {
            const endMeasure = performanceMonitor.startMeasure(`${componentName}.render`);
            return endMeasure;
        },
        trackAction: (actionName: string) => {
            return performanceMonitor.startMeasure(`${componentName}.${actionName}`);
        }
    };
}

// Axios interceptor helper for API tracking
export function createAPIInterceptor(axiosInstance: any) {
    // Request interceptor
    axiosInstance.interceptors.request.use((config: any) => {
        config.metadata = { startTime: performance.now() };
        return config;
    });

    // Response interceptor
    axiosInstance.interceptors.response.use(
        (response: any) => {
            const duration = performance.now() - response.config.metadata.startTime;
            performanceMonitor.trackAPICall(
                response.config.url,
                response.config.method.toUpperCase(),
                duration,
                response.status
            );
            return response;
        },
        (error: any) => {
            if (error.config && error.config.metadata) {
                const duration = performance.now() - error.config.metadata.startTime;
                performanceMonitor.trackAPICall(
                    error.config.url,
                    error.config.method.toUpperCase(),
                    duration,
                    error.response?.status || 0,
                    error
                );
            }
            return Promise.reject(error);
        }
    );
}
