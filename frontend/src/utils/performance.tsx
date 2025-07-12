import React, { useState, useEffect, useRef, useCallback } from 'react';
import { lazy, Suspense, ComponentType } from 'react';
import { useIntersectionObserver } from '../hooks/useIntersectionObserver';

// Lazy load components with loading fallback
export function lazyLoadComponent<T extends ComponentType<any>>(
  importFunc: () => Promise<{ default: T }>,
  fallback?: React.ReactNode
) {
  const LazyComponent = lazy(importFunc);

  return (props: React.ComponentProps<T>) => (
    <Suspense fallback={fallback || <div>Loading...</div>}>
      <LazyComponent {...props} />
    </Suspense>
  );
}

// Image optimization hook
export function useOptimizedImage(src: string, options?: {
  sizes?: string;
  quality?: number;
  format?: 'webp' | 'jpg' | 'png';
}) {
  const { sizes = '100vw', quality = 85, format = 'webp' } = options || {};

  // Generate srcset for responsive images
  const generateSrcSet = () => {
    const widths = [320, 640, 768, 1024, 1280, 1920];
    return widths
      .map(w => `${src}?w=${w}&q=${quality}&fm=${format} ${w}w`)
      .join(', ');
  };

  return {
    src: `${src}?q=${quality}&fm=${format}`,
    srcSet: generateSrcSet(),
    sizes,
  };
}

// Virtualized list component
export { FixedSizeList as VirtualList } from 'react-window';

// Memoization utilities
export { memo, useMemo, useCallback } from 'react';

// Debounce hook
export const useDebounce = <T,>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

// Throttle hook
export const useThrottle = <T,>(value: T, limit: number): T => {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastRan = useRef(Date.now());

  useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (Date.now() - lastRan.current));

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
};

// Request idle callback wrapper
export function scheduleIdleTask(callback: () => void) {
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(callback);
  } else {
    setTimeout(callback, 1);
  }
}

// Performance monitoring
export const PerformanceMonitor: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  useEffect(() => {
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        console.log('Performance entry:', entry);
      });
    });

    observer.observe({ entryTypes: ['measure', 'navigation'] });

    return () => {
      observer.disconnect();
    };
  }, []);

  return <>{children}</>;
};

// Web Worker for heavy computations
export class ComputeWorker {
  private worker: Worker;

  constructor(workerScript: string) {
    this.worker = new Worker(workerScript);
  }

  async compute<T, R>(data: T): Promise<R> {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => resolve(e.data);
      this.worker.onerror = reject;
      this.worker.postMessage(data);
    });
  }

  terminate() {
    this.worker.terminate();
  }
}

// Create intersection observer hook since it's missing
export const useIntersectionObserver = (
  options: IntersectionObserverInit = {}
) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null);
  const elementRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
        setEntry(entry);
      },
      options
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [options]);

  return { isIntersecting, entry, elementRef };
};

// Lazy loading component
export const LazyComponent: React.FC<{
  children: React.ReactNode;
  fallback?: React.ReactNode;
}> = ({ children, fallback = <div>Loading...</div> }) => {
  const { isIntersecting, elementRef } = useIntersectionObserver({
    threshold: 0.1,
  });

  return (
    <div ref={elementRef as React.RefObject<HTMLDivElement>}>
      {isIntersecting ? children : fallback}
    </div>
  );
};

// Performance measurement decorator
export const measurePerformance = (componentName: string) => {
  return <P extends object>(Component: React.ComponentType<P>) => {
    const MeasuredComponent: React.FC<P> = (props) => {
      useEffect(() => {
        performance.mark(`${componentName}-start`);

        return () => {
          performance.mark(`${componentName}-end`);
          performance.measure(
            `${componentName}-render`,
            `${componentName}-start`,
            `${componentName}-end`
          );
        };
      }, []);

      return <Component {...props} />;
    };

    MeasuredComponent.displayName = `Measured(${componentName})`;
    return MeasuredComponent;
  };
};
