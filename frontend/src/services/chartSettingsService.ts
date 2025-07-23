import logger from './logger';

/**
 * Chart Settings Service
 *
 * Manages persistence of chart configurations, indicators, and user preferences
 * across sessions using localStorage with fallback to sessionStorage.
 */

export interface ChartSettings {
  symbol?: string;
  timeframe?: string;
  indicators?: string[];
  theme?: 'dark' | 'light' | 'auto';
  showVolume?: boolean;
  showGrid?: boolean;
  showCrosshair?: boolean;
  drawingTools?: any[];
  savedLayouts?: SavedLayout[];
  favoriteSymbols?: string[];
  defaultView?: 'standard' | 'multi-timeframe' | 'comparison';
  multiTimeframeLayout?: 'single' | 'dual-horizontal' | 'dual-vertical' | 'quad';
  comparisonMode?: 'absolute' | 'percentage' | 'normalized';
  comparisonSymbols?: string[];
}

export interface SavedLayout {
  id: string;
  name: string;
  symbol: string;
  settings: ChartSettings;
  createdAt: string;
  updatedAt: string;
}

class ChartSettingsService {
  private readonly SETTINGS_KEY = 'goldensignals_chart_settings';
  private readonly LAYOUTS_KEY = 'goldensignals_saved_layouts';
  private readonly MAX_LAYOUTS = 20;

  /**
   * Get current chart settings
   */
  getSettings(): ChartSettings {
    try {
      const stored = localStorage.getItem(this.SETTINGS_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      logger.error('Error loading chart settings:', error);
    }

    return this.getDefaultSettings();
  }

  /**
   * Save chart settings
   */
  saveSettings(settings: Partial<ChartSettings>): void {
    try {
      const current = this.getSettings();
      const updated = { ...current, ...settings };
      localStorage.setItem(this.SETTINGS_KEY, JSON.stringify(updated));
    } catch (error) {
      logger.error('Error saving chart settings:', error);
      // Fallback to sessionStorage
      try {
        sessionStorage.setItem(this.SETTINGS_KEY, JSON.stringify(settings));
      } catch (sessionError) {
        logger.error('Error saving to sessionStorage:', sessionError);
      }
    }
  }

  /**
   * Get all saved layouts
   */
  getSavedLayouts(): SavedLayout[] {
    try {
      const stored = localStorage.getItem(this.LAYOUTS_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      logger.error('Error loading saved layouts:', error);
    }

    return [];
  }

  /**
   * Save a chart layout
   */
  saveLayout(name: string, symbol: string, settings: ChartSettings): SavedLayout {
    const layouts = this.getSavedLayouts();

    const newLayout: SavedLayout = {
      id: `layout_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      symbol,
      settings,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // Add new layout and limit to MAX_LAYOUTS
    const updatedLayouts = [newLayout, ...layouts].slice(0, this.MAX_LAYOUTS);

    try {
      localStorage.setItem(this.LAYOUTS_KEY, JSON.stringify(updatedLayouts));
    } catch (error) {
      logger.error('Error saving layout:', error);
    }

    return newLayout;
  }

  /**
   * Load a saved layout
   */
  loadLayout(layoutId: string): SavedLayout | null {
    const layouts = this.getSavedLayouts();
    return layouts.find(layout => layout.id === layoutId) || null;
  }

  /**
   * Delete a saved layout
   */
  deleteLayout(layoutId: string): void {
    const layouts = this.getSavedLayouts();
    const filtered = layouts.filter(layout => layout.id !== layoutId);

    try {
      localStorage.setItem(this.LAYOUTS_KEY, JSON.stringify(filtered));
    } catch (error) {
      logger.error('Error deleting layout:', error);
    }
  }

  /**
   * Update a saved layout
   */
  updateLayout(layoutId: string, updates: Partial<SavedLayout>): void {
    const layouts = this.getSavedLayouts();
    const updated = layouts.map(layout => {
      if (layout.id === layoutId) {
        return {
          ...layout,
          ...updates,
          updatedAt: new Date().toISOString(),
        };
      }
      return layout;
    });

    try {
      localStorage.setItem(this.LAYOUTS_KEY, JSON.stringify(updated));
    } catch (error) {
      logger.error('Error updating layout:', error);
    }
  }

  /**
   * Add a favorite symbol
   */
  addFavoriteSymbol(symbol: string): void {
    const settings = this.getSettings();
    const favorites = settings.favoriteSymbols || [];

    if (!favorites.includes(symbol)) {
      this.saveSettings({
        ...settings,
        favoriteSymbols: [...favorites, symbol].slice(0, 50), // Limit to 50 favorites
      });
    }
  }

  /**
   * Remove a favorite symbol
   */
  removeFavoriteSymbol(symbol: string): void {
    const settings = this.getSettings();
    const favorites = settings.favoriteSymbols || [];

    this.saveSettings({
      ...settings,
      favoriteSymbols: favorites.filter(s => s !== symbol),
    });
  }

  /**
   * Clear all settings
   */
  clearAllSettings(): void {
    try {
      localStorage.removeItem(this.SETTINGS_KEY);
      localStorage.removeItem(this.LAYOUTS_KEY);
    } catch (error) {
      logger.error('Error clearing settings:', error);
    }
  }

  /**
   * Export settings as JSON
   */
  exportSettings(): string {
    const settings = this.getSettings();
    const layouts = this.getSavedLayouts();

    return JSON.stringify({
      settings,
      layouts,
      exportedAt: new Date().toISOString(),
      version: '1.0',
    }, null, 2);
  }

  /**
   * Import settings from JSON
   */
  importSettings(jsonString: string): boolean {
    try {
      const data = JSON.parse(jsonString);

      if (data.settings) {
        this.saveSettings(data.settings);
      }

      if (data.layouts) {
        localStorage.setItem(this.LAYOUTS_KEY, JSON.stringify(data.layouts));
      }

      return true;
    } catch (error) {
      logger.error('Error importing settings:', error);
      return false;
    }
  }

  /**
   * Get default settings
   */
  private getDefaultSettings(): ChartSettings {
    return {
      timeframe: '5m',
      indicators: ['volume', 'prediction', 'signals'],
      theme: 'dark',
      showVolume: true,
      showGrid: true,
      showCrosshair: true,
      drawingTools: [],
      savedLayouts: [],
      favoriteSymbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ'],
      defaultView: 'standard',
      multiTimeframeLayout: 'single',
      comparisonMode: 'percentage',
      comparisonSymbols: [],
    };
  }
}

// Export singleton instance
export const chartSettingsService = new ChartSettingsService();
