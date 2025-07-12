import {
    IExternalDatafeed,
    TradingViewSymbolInfo,
    TradingViewBar,
    TradingViewSearchSymbol,
    TradingViewHistoryCallback,
    TradingViewErrorCallback,
    TradingViewResolveCallback,
    TradingViewSearchCallback,
    TradingViewSubscribeBarsCallback
} from '../../types/TradingViewTypes';
import { marketDataService, MarketBar, MarketQuote } from '../../services/MarketDataService';

export class TradingViewDatafeed implements IExternalDatafeed {
    private subscribers: Map<string, {
        symbolInfo: TradingViewSymbolInfo;
        resolution: string;
        onRealtimeCallback: TradingViewSubscribeBarsCallback;
        subscriberUID: string;
    }> = new Map();

    private lastBarsCache = new Map<string, TradingViewBar>();

    constructor() {
        console.log('TradingView Datafeed initialized with real market data');
    }

    /**
     * Called by TradingView to check if datafeed is ready
     */
    onReady(callback: (configurationData: any) => void): void {
        console.log('[onReady]: Method called');

        setTimeout(() => {
            callback({
                exchanges: [
                    { value: 'NASDAQ', name: 'NASDAQ', desc: 'NASDAQ Stock Market' },
                    { value: 'NYSE', name: 'NYSE', desc: 'New York Stock Exchange' },
                    { value: 'AMEX', name: 'AMEX', desc: 'American Stock Exchange' },
                ],
                symbols_types: [
                    { name: 'All types', value: '' },
                    { name: 'Stock', value: 'stock' },
                    { name: 'Index', value: 'index' },
                    { name: 'Forex', value: 'forex' },
                    { name: 'Bitcoin', value: 'bitcoin' },
                ],
                supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D', '1W', '1M'],
                supports_marks: true,
                supports_timescale_marks: true,
                supports_time: true,
                supports_search: true,
                supports_group_request: false,
            });
        }, 0);
    }

    /**
     * Search for symbols
     */
    searchSymbols(
        userInput: string,
        exchange: string,
        symbolType: string,
        onResultReadyCallback: TradingViewSearchCallback
    ): void {
        console.log('[searchSymbols]: Method called', { userInput, exchange, symbolType });

        marketDataService.searchSymbols(userInput, symbolType, exchange)
            .then(symbols => {
                const searchResults: TradingViewSearchSymbol[] = symbols.map(symbol => ({
                    symbol: symbol.symbol,
                    full_name: `${symbol.exchange}:${symbol.symbol}`,
                    description: symbol.name,
                    exchange: symbol.exchange,
                    ticker: symbol.symbol,
                    type: symbol.type,
                }));

                console.log('[searchSymbols]: Returning', searchResults.length, 'results');
                onResultReadyCallback(searchResults);
            })
            .catch(error => {
                console.error('[searchSymbols]: Error', error);
                onResultReadyCallback([]);
            });
    }

    /**
     * Resolve symbol information
     */
    resolveSymbol(
        symbolName: string,
        onSymbolResolvedCallback: TradingViewResolveCallback,
        onResolveErrorCallback: TradingViewErrorCallback,
        extension?: any
    ): void {
        console.log('[resolveSymbol]: Method called', symbolName);

        // Clean symbol name (remove exchange prefix if present)
        const cleanSymbol = symbolName.includes(':') ? symbolName.split(':')[1] : symbolName;

        marketDataService.getSymbolInfo(cleanSymbol)
            .then(symbolInfo => {
                const tradingViewSymbolInfo: TradingViewSymbolInfo = {
                    name: symbolInfo.symbol,
                    full_name: `${symbolInfo.exchange}:${symbolInfo.symbol}`,
                    description: symbolInfo.name,
                    exchange: symbolInfo.exchange,
                    listed_exchange: symbolInfo.exchange,
                    type: symbolInfo.type,
                    session: symbolInfo.sessionRegular,
                    timezone: symbolInfo.timezone,
                    ticker: symbolInfo.symbol,
                    minmov: symbolInfo.minMovement,
                    pricescale: Math.pow(10, symbolInfo.pricePrecision),
                    has_intraday: true,
                    has_no_volume: false,
                    has_weekly_and_monthly: true,
                    supported_resolutions: symbolInfo.supported_resolutions,
                    intraday_multipliers: ['1', '5', '15', '30', '60', '240'],
                    has_seconds: false,
                    has_daily: true,
                    has_empty_bars: true,
                    force_session_rebuild: false,
                    volume_precision: symbolInfo.volumePrecision,
                    data_status: 'streaming',
                    currency_code: symbolInfo.currency,
                };

                console.log('[resolveSymbol]: Symbol resolved', tradingViewSymbolInfo);
                onSymbolResolvedCallback(tradingViewSymbolInfo);
            })
            .catch(error => {
                console.error('[resolveSymbol]: Error resolving symbol', error);
                onResolveErrorCallback(`Cannot resolve symbol ${symbolName}`);
            });
    }

    /**
     * Get historical data
     */
    getBars(
        symbolInfo: TradingViewSymbolInfo,
        resolution: string,
        periodParams: {
            from: number;
            to: number;
            countBack: number;
            firstDataRequest: boolean;
        },
        onHistoryCallback: TradingViewHistoryCallback,
        onErrorCallback: TradingViewErrorCallback
    ): void {
        console.log('[getBars]: Method called', {
            symbol: symbolInfo.name,
            resolution,
            from: new Date(periodParams.from * 1000),
            to: new Date(periodParams.to * 1000),
            firstDataRequest: periodParams.firstDataRequest,
        });

        const { from, to } = periodParams;

        marketDataService.getHistoricalData(symbolInfo.name, resolution, from, to)
            .then((bars: MarketBar[]) => {
                console.log(`[getBars]: Retrieved ${bars.length} bars for ${symbolInfo.name}`);

                if (bars.length === 0) {
                    onHistoryCallback([], { noData: true });
                    return;
                }

                // Convert MarketBar to TradingViewBar
                const tradingViewBars: TradingViewBar[] = bars.map(bar => ({
                    time: bar.time * 1000, // TradingView expects milliseconds
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume,
                }));

                // Cache the last bar for real-time updates
                if (tradingViewBars.length > 0) {
                    const lastBar = tradingViewBars[tradingViewBars.length - 1];
                    const cacheKey = `${symbolInfo.name}_${resolution}`;
                    this.lastBarsCache.set(cacheKey, lastBar);
                }

                onHistoryCallback(tradingViewBars, { noData: false });
            })
            .catch(error => {
                console.error('[getBars]: Error fetching historical data', error);
                onErrorCallback(`Failed to fetch data: ${error.message}`);
            });
    }

    /**
     * Subscribe to real-time data
     */
    subscribeBars(
        symbolInfo: TradingViewSymbolInfo,
        resolution: string,
        onRealtimeCallback: TradingViewSubscribeBarsCallback,
        subscriberUID: string,
        onResetCacheNeededCallback: () => void
    ): void {
        console.log('[subscribeBars]: Method called', {
            symbol: symbolInfo.name,
            resolution,
            subscriberUID,
        });

        // Store subscription info
        this.subscribers.set(subscriberUID, {
            symbolInfo,
            resolution,
            onRealtimeCallback,
            subscriberUID,
        });

        // Subscribe to real-time data from our market data service
        const subscriptionId = marketDataService.subscribeToRealTimeData(
            symbolInfo.name,
            (quote: MarketQuote) => {
                console.log('[subscribeBars]: Real-time quote received', quote);

                // Convert quote to bar format
                const cacheKey = `${symbolInfo.name}_${resolution}`;
                const lastBar = this.lastBarsCache.get(cacheKey);

                if (lastBar) {
                    // Update the last bar with new price
                    const updatedBar: TradingViewBar = {
                        ...lastBar,
                        close: quote.price,
                        high: Math.max(lastBar.high, quote.price),
                        low: Math.min(lastBar.low, quote.price),
                        volume: quote.volume || lastBar.volume,
                        time: quote.timestamp * 1000, // Convert to milliseconds
                    };

                    // Update cache
                    this.lastBarsCache.set(cacheKey, updatedBar);

                    // Send update to TradingView
                    onRealtimeCallback(updatedBar);
                } else {
                    // Create new bar if no cached bar exists
                    const newBar: TradingViewBar = {
                        time: quote.timestamp * 1000,
                        open: quote.price,
                        high: quote.price,
                        low: quote.price,
                        close: quote.price,
                        volume: quote.volume || 0,
                    };

                    this.lastBarsCache.set(cacheKey, newBar);
                    onRealtimeCallback(newBar);
                }
            }
        );

        console.log(`[subscribeBars]: Subscribed to ${symbolInfo.name} with ID ${subscriptionId}`);
    }

    /**
     * Unsubscribe from real-time data
     */
    unsubscribeBars(subscriberUID: string): void {
        console.log('[unsubscribeBars]: Method called', subscriberUID);

        const subscription = this.subscribers.get(subscriberUID);
        if (subscription) {
            // Remove from our subscribers map
            this.subscribers.delete(subscriberUID);

            // Unsubscribe from market data service
            marketDataService.unsubscribeFromRealTimeData(subscriberUID);

            console.log(`[unsubscribeBars]: Unsubscribed ${subscriberUID}`);
        }
    }

    /**
     * Get quotes (optional - for multiple symbols)
     */
    getQuotes?(
        symbols: string[],
        onDataCallback: (data: any[]) => void,
        onErrorCallback: (error: string) => void
    ): void {
        console.log('[getQuotes]: Method called', symbols);

        const quotePromises = symbols.map(symbol =>
            marketDataService.getRealTimeQuote(symbol)
                .then(quote => ({
                    s: 'ok',
                    n: symbol,
                    v: {
                        ch: quote.change,
                        chp: quote.changePercent,
                        short_name: symbol,
                        exchange: 'NASDAQ', // Default exchange
                        original_name: symbol,
                        description: symbol,
                        lp: quote.price,
                        ask: quote.ask,
                        bid: quote.bid,
                        open_price: quote.price, // Approximation
                        high_price: quote.price, // Approximation
                        low_price: quote.price,  // Approximation
                        prev_close_price: quote.price - quote.change,
                        volume: quote.volume,
                    }
                }))
                .catch(error => ({
                    s: 'error',
                    n: symbol,
                    v: {},
                }))
        );

        Promise.allSettled(quotePromises)
            .then(results => {
                const quotes = results.map(result =>
                    result.status === 'fulfilled' ? result.value : { s: 'error', n: '', v: {} }
                );
                onDataCallback(quotes);
            })
            .catch(error => {
                console.error('[getQuotes]: Error fetching quotes', error);
                onErrorCallback(`Failed to fetch quotes: ${error.message}`);
            });
    }
}

// Export singleton instance
export const tradingViewDatafeed = new TradingViewDatafeed(); 