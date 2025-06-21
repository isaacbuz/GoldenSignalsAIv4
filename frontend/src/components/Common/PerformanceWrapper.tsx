import React, { memo, useCallback, useMemo } from 'react';

/**
 * Performance optimization wrapper for expensive components
 * Implements React.memo with custom comparison function
 */

interface PerformanceWrapperProps {
    children: React.ReactNode;
    dependencies?: any[];
    compareFunction?: (prevProps: any, nextProps: any) => boolean;
}

// Default shallow comparison function
const defaultCompare = (prevProps: any, nextProps: any): boolean => {
    const prevKeys = Object.keys(prevProps);
    const nextKeys = Object.keys(nextProps);

    if (prevKeys.length !== nextKeys.length) {
        return false;
    }

    for (const key of prevKeys) {
        if (prevProps[key] !== nextProps[key]) {
            return false;
        }
    }

    return true;
};

export const PerformanceWrapper = memo<PerformanceWrapperProps>(
    ({ children }) => {
        return <>{children}</>;
    },
    (prevProps, nextProps) => {
        if (prevProps.compareFunction) {
            return prevProps.compareFunction(prevProps, nextProps);
        }
        return defaultCompare(prevProps, nextProps);
    }
);

PerformanceWrapper.displayName = 'PerformanceWrapper';

/**
 * Hook for expensive calculations with memoization
 */
export const useExpensiveCalculation = <T,>(
    calculateFn: () => T,
    dependencies: React.DependencyList
): T => {
    return useMemo(calculateFn, dependencies);
};

/**
 * Hook for optimized callbacks
 */
export const useOptimizedCallback = <T extends (...args: any[]) => any>(
    callback: T,
    dependencies: React.DependencyList
): T => {
    return useCallback(callback, dependencies);
};

/**
 * Virtual list component for large data sets
 */
interface VirtualListProps<T> {
    items: T[];
    height: number;
    itemHeight: number;
    renderItem: (item: T, index: number) => React.ReactNode;
    overscan?: number;
}

export function VirtualList<T>({
    items,
    height,
    itemHeight,
    renderItem,
    overscan = 3
}: VirtualListProps<T>) {
    const [scrollTop, setScrollTop] = React.useState(0);

    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const endIndex = Math.min(
        items.length - 1,
        Math.ceil((scrollTop + height) / itemHeight) + overscan
    );

    const visibleItems = items.slice(startIndex, endIndex + 1);
    const totalHeight = items.length * itemHeight;
    const offsetY = startIndex * itemHeight;

    const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
        setScrollTop(e.currentTarget.scrollTop);
    }, []);

    return (
        <div
            style={{ height, overflow: 'auto' }}
            onScroll={handleScroll}
        >
            <div style={{ height: totalHeight, position: 'relative' }}>
                <div
                    style={{
                        transform: `translateY(${offsetY}px)`,
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                    }}
                >
                    {visibleItems.map((item, index) => (
                        <div key={startIndex + index} style={{ height: itemHeight }}>
                            {renderItem(item, startIndex + index)}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

/**
 * Debounce hook for input optimization
 */
export const useDebounce = <T,>(value: T, delay: number): T => {
    const [debouncedValue, setDebouncedValue] = React.useState<T>(value);

    React.useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => {
            clearTimeout(handler);
        };
    }, [value, delay]);

    return debouncedValue;
};

/**
 * Lazy load wrapper for code splitting
 */
interface LazyLoadWrapperProps {
    fallback?: React.ReactNode;
    children: React.ReactNode;
}

export const LazyLoadWrapper: React.FC<LazyLoadWrapperProps> = ({
    fallback = <div>Loading...</div>,
    children
}) => {
    return (
        <React.Suspense fallback={fallback}>
            {children}
        </React.Suspense>
    );
}; 