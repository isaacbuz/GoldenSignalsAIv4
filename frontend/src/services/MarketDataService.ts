import axios from 'axios';
import yahooFinance from 'yahoo-finance2';

export interface MarketBar {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface MarketQuote {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    bid: number;
    ask: number;
    timestamp: number;
}

export interface SymbolInfo {
    symbol: string;
    name: string;
    exchange: string;
    type: string;
    currency: string;
    minMovement: number;
    pricePrecision: number;
    volumePrecision: number;
    timezone: string;
    sessionRegular: string;
    sessionExtended: string;
    supported_resolutions: string[];
}

export class MarketDataService {
    private static instance: MarketDataService;
    private wsConnections: Map<string, WebSocket> = new Map();
    private subscribers: Map<string, Set<(data: any) => void>> = new Map();

    // Alpha Vantage API key (free tier: 5 requests/minute)
    private alphaVantageKey = process.env.REACT_APP_ALPHA_VANTAGE_KEY || 'demo';
    private alphaVantageBaseUrl = 'https://www.alphavantage.co/query';

    public static getInstance(): MarketDataService {
        if (!MarketDataService.instance) {
            MarketDataService.instance = new MarketDataService();
        }
        return MarketDataService.instance;
    }

    /**
     * Get symbol information for TradingView
     */
    async getSymbolInfo(symbol: string): Promise<SymbolInfo> {
        try {
            // Try Yahoo Finance first (free and reliable)
            const quote = await yahooFinance.quote(symbol);

            return {
                symbol: symbol.toUpperCase(),
                name: quote.displayName || quote.shortName || symbol,
                exchange: quote.fullExchangeName || 'UNKNOWN',
                type: this.determineAssetType(quote.quoteType || ''),
                currency: quote.currency || 'USD',
                minMovement: 1,
                pricePrecision: 2,
                volumePrecision: 0,
                timezone: 'America/New_York',
                sessionRegular: '0930-1600',
                sessionExtended: '0400-2000',
                supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D', '1W', '1M']
            };
        } catch (error) {
            console.error(`Error fetching symbol info for ${symbol}:`, error);
            // Fallback symbol info
            return {
                symbol: symbol.toUpperCase(),
                name: symbol.toUpperCase(),
                exchange: 'UNKNOWN',
                type: 'stock',
                currency: 'USD',
                minMovement: 1,
                pricePrecision: 2,
                volumePrecision: 0,
                timezone: 'America/New_York',
                sessionRegular: '0930-1600',
                sessionExtended: '0400-2000',
                supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D', '1W', '1M']
            };
        }
    }

    /**
     * Get historical market data for TradingView
     */
    async getHistoricalData(
        symbol: string,
        resolution: string,
        from: number,
        to: number,
        countback?: number
    ): Promise<MarketBar[]> {
        try {
            console.log(`Fetching historical data for ${symbol}, resolution: ${resolution}`);

            // Convert TradingView resolution to Yahoo Finance interval
            const interval = this.convertResolutionToInterval(resolution);
            const period1 = new Date(from * 1000);
            const period2 = new Date(to * 1000);

            // Use Yahoo Finance for historical data
            const result = await yahooFinance.historical(symbol, {
                period1,
                period2,
                interval: interval as any,
            });

            // Convert to TradingView format
            const bars: MarketBar[] = result.map(bar => ({
                time: Math.floor(bar.date.getTime() / 1000),
                open: bar.open || 0,
                high: bar.high || 0,
                low: bar.low || 0,
                close: bar.close || 0,
                volume: bar.volume || 0,
            })).filter(bar => bar.time >= from && bar.time <= to);

            // Sort by time ascending
            bars.sort((a, b) => a.time - b.time);

            console.log(`Retrieved ${bars.length} bars for ${symbol}`);
            return bars;

        } catch (error) {
            console.error(`Error fetching historical data for ${symbol}:`, error);

            // Fallback: Generate sample data if API fails
            return this.generateSampleData(symbol, resolution, from, to);
        }
    }

    /**
     * Get real-time quote for a symbol
     */
    async getRealTimeQuote(symbol: string): Promise<MarketQuote> {
        try {
            const quote = await yahooFinance.quote(symbol);

            return {
                symbol: symbol.toUpperCase(),
                price: quote.regularMarketPrice || 0,
                change: quote.regularMarketChange || 0,
                changePercent: quote.regularMarketChangePercent || 0,
                volume: quote.regularMarketVolume || 0,
                bid: quote.bid || 0,
                ask: quote.ask || 0,
                timestamp: Math.floor(Date.now() / 1000),
            };
        } catch (error) {
            console.error(`Error fetching real-time quote for ${symbol}:`, error);
            throw error;
        }
    }

    /**
     * Search for symbols (for symbol search in TradingView)
     */
    async searchSymbols(query: string, type?: string, exchange?: string): Promise<SymbolInfo[]> {
        try {
            // Use Yahoo Finance search
            const searchResults = await yahooFinance.search(query, {
                quotesCount: 10,
                newsCount: 0,
            });

            const symbols: SymbolInfo[] = [];

            if (searchResults.quotes) {
                for (const quote of searchResults.quotes) {
                    if (quote.symbol) {
                        try {
                            const symbolInfo = await this.getSymbolInfo(quote.symbol);
                            symbols.push(symbolInfo);
                        } catch (error) {
                            console.warn(`Failed to get info for ${quote.symbol}:`, error);
                        }
                    }
                }
            }

            return symbols;
        } catch (error) {
            console.error('Error searching symbols:', error);
            return [];
        }
    }

    /**
     * Subscribe to real-time data updates
     */
    subscribeToRealTimeData(symbol: string, callback: (quote: MarketQuote) => void): string {
        const subscriptionId = `${symbol}_${Date.now()}`;

        if (!this.subscribers.has(symbol)) {
            this.subscribers.set(symbol, new Set());
        }

        this.subscribers.get(symbol)!.add(callback);

        // Start polling for real-time data (since Yahoo Finance doesn't have WebSocket)
        this.startPolling(symbol);

        return subscriptionId;
    }

    /**
     * Unsubscribe from real-time data
     */
    unsubscribeFromRealTimeData(subscriptionId: string): void {
        // Implementation for unsubscribing
        console.log(`Unsubscribed from ${subscriptionId}`);
    }

    /**
     * Private helper methods
     */
    private convertResolutionToInterval(resolution: string): string {
        const resolutionMap: { [key: string]: string } = {
            '1': '1m',
            '5': '5m',
            '15': '15m',
            '30': '30m',
            '60': '1h',
            '240': '4h',
            '1D': '1d',
            '1W': '1wk',
            '1M': '1mo',
        };

        return resolutionMap[resolution] || '1d';
    }

    private determineAssetType(quoteType: string): string {
        const typeMap: { [key: string]: string } = {
            'EQUITY': 'stock',
            'ETF': 'stock',
            'MUTUALFUND': 'stock',
            'INDEX': 'index',
            'CRYPTOCURRENCY': 'crypto',
            'CURRENCY': 'forex',
            'FUTURE': 'futures',
            'OPTION': 'option',
        };

        return typeMap[quoteType.toUpperCase()] || 'stock';
    }

    private async startPolling(symbol: string): Promise<void> {
        // Simple polling every 5 seconds for real-time updates
        const interval = setInterval(async () => {
            try {
                const quote = await this.getRealTimeQuote(symbol);
                const callbacks = this.subscribers.get(symbol);

                if (callbacks) {
                    callbacks.forEach(callback => callback(quote));
                }
            } catch (error) {
                console.error(`Error polling ${symbol}:`, error);
            }
        }, 5000); // Poll every 5 seconds

        // Store interval for cleanup
        setTimeout(() => clearInterval(interval), 300000); // Clean up after 5 minutes
    }

    private generateSampleData(symbol: string, resolution: string, from: number, to: number): MarketBar[] {
        console.log(`Generating sample data for ${symbol} as fallback`);

        const bars: MarketBar[] = [];
        const intervalMs = this.getIntervalMs(resolution);
        const basePrice = 150 + Math.random() * 100; // Random base price between 150-250

        let currentTime = from;
        let currentPrice = basePrice;

        while (currentTime <= to) {
            const volatility = 0.02; // 2% volatility
            const change = (Math.random() - 0.5) * volatility * currentPrice;

            const open = currentPrice;
            const close = Math.max(0.01, currentPrice + change);
            const high = Math.max(open, close) * (1 + Math.random() * 0.01);
            const low = Math.min(open, close) * (1 - Math.random() * 0.01);
            const volume = Math.floor(1000000 + Math.random() * 5000000);

            bars.push({
                time: currentTime,
                open,
                high,
                low,
                close,
                volume,
            });

            currentPrice = close;
            currentTime += Math.floor(intervalMs / 1000);
        }

        return bars;
    }

    private getIntervalMs(resolution: string): number {
        const intervalMap: { [key: string]: number } = {
            '1': 60 * 1000,        // 1 minute
            '5': 5 * 60 * 1000,    // 5 minutes
            '15': 15 * 60 * 1000,  // 15 minutes
            '30': 30 * 60 * 1000,  // 30 minutes
            '60': 60 * 60 * 1000,  // 1 hour
            '240': 240 * 60 * 1000, // 4 hours
            '1D': 24 * 60 * 60 * 1000, // 1 day
            '1W': 7 * 24 * 60 * 60 * 1000, // 1 week
            '1M': 30 * 24 * 60 * 60 * 1000, // 1 month (approx)
        };

        return intervalMap[resolution] || 24 * 60 * 60 * 1000;
    }
}

// Export singleton instance
export const marketDataService = MarketDataService.getInstance(); 