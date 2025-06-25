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
export function useDebounce<T>(value: T, delay: number): T {
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
}

// Request idle callback wrapper
export function scheduleIdleTask(callback: () => void) {
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(callback);
  } else {
    setTimeout(callback, 1);
  }
}

// Performance monitoring
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Map<string, number[]> = new Map();
  
  static getInstance() {
    if (!this.instance) {
      this.instance = new PerformanceMonitor();
    }
    return this.instance;
  }
  
  measure(name: string, fn: () => void) {
    const start = performance.now();
    fn();
    const duration = performance.now() - start;
    
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);
    
    // Log slow operations
    if (duration > 16) { // Slower than 60fps
      console.warn(`Slow operation: ${name} took ${duration.toFixed(2)}ms`);
    }
  }
  
  getAverageTime(name: string): number {
    const times = this.metrics.get(name) || [];
    if (times.length === 0) return 0;
    return times.reduce((a, b) => a + b, 0) / times.length;
  }
  
  report() {
    const report: Record<string, any> = {};
    this.metrics.forEach((times, name) => {
      report[name] = {
        average: this.getAverageTime(name),
        count: times.length,
        total: times.reduce((a, b) => a + b, 0),
      };
    });
    return report;
  }
}

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
