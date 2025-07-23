import logger from '../services/logger';

/**
 * Cache Manager for GoldenSignalsAI
 * Provides utilities to manage and clear various caches
 */

export interface CacheStatus {
  localStorage: {
    size: number;
    items: string[];
  };
  sessionStorage: {
    size: number;
    items: string[];
  };
  serviceworker: {
    active: boolean;
    caches: string[];
  };
}

export class CacheManager {
  /**
   * Get current cache status
   */
  static async getStatus(): Promise<CacheStatus> {
    const status: CacheStatus = {
      localStorage: {
        size: 0,
        items: [],
      },
      sessionStorage: {
        size: 0,
        items: [],
      },
      serviceworker: {
        active: false,
        caches: [],
      },
    };

    // Check localStorage
    try {
      status.localStorage.items = Object.keys(localStorage);
      status.localStorage.size = new Blob(
        status.localStorage.items.map(k => localStorage.getItem(k) || '')
      ).size;
    } catch (e) {
      logger.error('Error checking localStorage:', e);
    }

    // Check sessionStorage
    try {
      status.sessionStorage.items = Object.keys(sessionStorage);
      status.sessionStorage.size = new Blob(
        status.sessionStorage.items.map(k => sessionStorage.getItem(k) || '')
      ).size;
    } catch (e) {
      logger.error('Error checking sessionStorage:', e);
    }

    // Check service worker caches
    if ('caches' in window) {
      try {
        const cacheNames = await caches.keys();
        status.serviceworker.caches = cacheNames;
        status.serviceworker.active = 'serviceWorker' in navigator;
      } catch (e) {
        logger.error('Error checking caches:', e);
      }
    }

    return status;
  }

  /**
   * Clear all caches
   */
  static async clearAll(): Promise<void> {
    logger.info('🧹 Clearing all caches...');

    // Clear localStorage
    try {
      const keys = Object.keys(localStorage);
      logger.info(`Clearing ${keys.length} localStorage items`);
      localStorage.clear();
    } catch (e) {
      logger.error('Error clearing localStorage:', e);
    }

    // Clear sessionStorage
    try {
      const keys = Object.keys(sessionStorage);
      logger.info(`Clearing ${keys.length} sessionStorage items`);
      sessionStorage.clear();
    } catch (e) {
      logger.error('Error clearing sessionStorage:', e);
    }

    // Clear service worker caches
    if ('caches' in window) {
      try {
        const cacheNames = await caches.keys();
        logger.info(`Clearing ${cacheNames.length} service worker caches`);
        await Promise.all(cacheNames.map(name => caches.delete(name)));
      } catch (e) {
        logger.error('Error clearing caches:', e);
      }
    }

    // Clear IndexedDB
    if ('indexedDB' in window) {
      try {
        const databases = await (indexedDB as any).databases?.() || [];
        logger.info(`Clearing ${databases.length} IndexedDB databases`);
        for (const db of databases) {
          if (db.name) {
            indexedDB.deleteDatabase(db.name);
          }
        }
      } catch (e) {
        logger.error('Error clearing IndexedDB:', e);
      }
    }

    logger.info('✅ All caches cleared');
  }

  /**
   * Clear chart-specific caches
   */
  static clearChartCache(): void {
    const chartKeys = Object.keys(localStorage).filter(key =>
      key.includes('chart') ||
      key.includes('layout') ||
      key.includes('lastAnalysis') ||
      key.includes('symbol')
    );

    logger.info(`🧹 Clearing ${chartKeys.length} chart-related cache items`);

    chartKeys.forEach(key => {
      localStorage.removeItem(key);
    });
  }

  /**
   * Clear API response caches
   */
  static async clearAPICache(): Promise<void> {
    // Clear backend cache via API
    try {
      const response = await fetch('http://localhost:8000/api/v1/cache/clear', {
        method: 'POST',
      });

      if (response.ok) {
        logger.info('✅ Backend cache cleared');
      }
    } catch (e) {
      logger.error('Error clearing backend cache:', e);
    }

    // Clear frontend API cache
    const apiKeys = Object.keys(localStorage).filter(key =>
      key.includes('api_cache') ||
      key.includes('market_data') ||
      key.includes('signals')
    );

    apiKeys.forEach(key => {
      localStorage.removeItem(key);
    });

    logger.info(`✅ Cleared ${apiKeys.length} API cache items`);
  }
}

// Auto-clear old cache items on load
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    // Clear items older than 24 hours
    const now = Date.now();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours

    Object.keys(localStorage).forEach(key => {
      try {
        const item = localStorage.getItem(key);
        if (item && item.includes('timestamp')) {
          const data = JSON.parse(item);
          if (data.timestamp && now - data.timestamp > maxAge) {
            localStorage.removeItem(key);
            logger.info(`🧹 Removed old cache item: ${key}`);
          }
        }
      } catch (e) {
        // Ignore parse errors
      }
    });
  });
}
