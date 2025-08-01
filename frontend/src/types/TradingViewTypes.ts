// TradingView Charting Library Types
export interface TradingViewBar {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

export interface TradingViewSymbolInfo {
    name: string;
    full_name: string;
    description: string;
    exchange: string;
    listed_exchange: string;
    type: string;
    session: string;
    timezone: string;
    ticker?: string;
    minmov: number;
    pricescale: number;
    minmove2?: number;
    fractional?: boolean;
    has_intraday: boolean;
    has_no_volume: boolean;
    has_weekly_and_monthly: boolean;
    supported_resolutions: string[];
    intraday_multipliers: string[];
    has_seconds: boolean;
    seconds_multipliers?: string[];
    has_daily: boolean;
    has_weekly_and_monthly: boolean;
    has_empty_bars: boolean;
    force_session_rebuild: boolean;
    has_no_volume: boolean;
    volume_precision: number;
    data_status: string;
    expired?: boolean;
    expiration_date?: number;
    sector?: string;
    industry?: string;
    currency_code: string;
    original_currency_code?: string;
}

export interface TradingViewQuote {
    ch: number;
    chp: number;
    short_name: string;
    exchange: string;
    original_name: string;
    description: string;
    lp: number;
    ask: number;
    bid: number;
    open_price: number;
    high_price: number;
    low_price: number;
    prev_close_price: number;
    volume: number;
}

export interface TradingViewSearchSymbol {
    symbol: string;
    full_name: string;
    description: string;
    exchange: string;
    ticker: string;
    type: string;
}

export interface TradingViewHistoryCallback {
    (bars: TradingViewBar[], meta: { noData: boolean }): void;
}

export interface TradingViewErrorCallback {
    (error: string): void;
}

export interface TradingViewOnReady {
    (): void;
}

export interface TradingViewResolveCallback {
    (symbolInfo: TradingViewSymbolInfo): void;
}

export interface TradingViewSearchCallback {
    (symbols: TradingViewSearchSymbol[]): void;
}

export interface TradingViewSubscribeBarsCallback {
    (bar: TradingViewBar): void;
}

// TradingView Datafeed Interface
export interface IExternalDatafeed {
    onReady(callback: (configurationData: any) => void): void;

    searchSymbols(
        userInput: string,
        exchange: string,
        symbolType: string,
        onResultReadyCallback: TradingViewSearchCallback
    ): void;

    resolveSymbol(
        symbolName: string,
        onSymbolResolvedCallback: TradingViewResolveCallback,
        onResolveErrorCallback: TradingViewErrorCallback,
        extension?: any
    ): void;

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
    ): void;

    subscribeBars(
        symbolInfo: TradingViewSymbolInfo,
        resolution: string,
        onRealtimeCallback: TradingViewSubscribeBarsCallback,
        subscriberUID: string,
        onResetCacheNeededCallback: () => void
    ): void;

    unsubscribeBars(subscriberUID: string): void;

    getQuotes?(
        symbols: string[],
        onDataCallback: (data: TradingViewQuote[]) => void,
        onErrorCallback: (error: string) => void
    ): void;

    subscribeQuotes?(
        symbols: string[],
        fastSymbols: string[],
        onRealtimeCallback: (data: TradingViewQuote[]) => void,
        listenerGUID: string
    ): void;

    unsubscribeQuotes?(listenerGUID: string): void;
}

// TradingView Widget Configuration
export interface TradingViewWidgetOptions {
    symbol: string;
    datafeed: IExternalDatafeed;
    interval: string;
    container_id: string;
    library_path: string;
    locale: string;
    disabled_features?: string[];
    enabled_features?: string[];
    charts_storage_url?: string;
    charts_storage_api_version?: string;
    client_id?: string;
    user_id?: string;
    fullscreen?: boolean;
    autosize?: boolean;
    theme?: 'light' | 'dark';
    toolbar_bg?: string;
    studies_overrides?: any;
    overrides?: any;
    time_frames?: any[];
    debug?: boolean;
    snapshot_url?: string;
    custom_css_url?: string;
    loading_screen?: { backgroundColor: string; foregroundColor: string };
    favorites?: {
        intervals: string[];
        chartTypes: string[];
    };
    save_image?: boolean;
    numeric_formatting?: {
        decimal_sign: string;
    };
    rounding?: number;
}

// AI Signal Integration Types
export interface AISignal {
    id: string;
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    timestamp: number;
    price: number;
    reasoning: string;
    targets: number[];
    stopLoss: number;
    timeframe: string;
    pattern?: string;
    indicators?: {
        rsi?: number;
        macd?: number;
        ema21?: number;
        ema50?: number;
        volume?: number;
    };
}

export interface AIPattern {
    id: string;
    symbol: string;
    type: string;
    confidence: number;
    startTime: number;
    endTime: number;
    points: Array<{ time: number; price: number }>;
    description: string;
    expectedOutcome: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    probability: number;
}

// TradingView Chart API Types
export interface IChartingLibraryWidget {
    onChartReady(callback: () => void): void;
    activeChart(): IChartApi;
    remove(): void;
    save(callback: (state: any) => void): void;
    load(state: any, callback?: () => void): void;
    setSymbol(symbol: string, interval: string, callback?: () => void): void;
    takeScreenshot(): Promise<string>;
}

export interface IChartApi {
    createStudy(name: string, forceOverlay?: boolean, lock?: boolean, inputs?: any): Promise<any>;
    removeAllStudies(): void;
    createShape(shape: any): any;
    removeAllShapes(): void;
    createMultipointShape(shape: any): any;
    getVisibleRange(): { from: number; to: number };
    setVisibleRange(range: { from: number; to: number }): void;
    scrollPosition(): number;
    defaultScrollPosition(): number;
    executeActionById(actionId: string): void;
    getCheckableActionState(actionId: string): boolean;
    setChartType(type: number): void;
    exportData(options?: any): Promise<any>;
}

// Chart Shape Types for AI Signals
export interface ChartShape {
    id?: string;
    time: number;
    channel?: string;
    text?: string;
    color?: string;
    size?: 'auto' | 'tiny' | 'small' | 'normal' | 'large' | 'huge';
    shape?: 'arrow_up' | 'arrow_down' | 'flag' | 'vertical_line' | 'horizontal_line';
    location?: 'absolute' | 'relative';
}

export interface ChartMarker {
    time: number;
    position: 'aboveBar' | 'belowBar' | 'inBar';
    color: string;
    shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown';
    text?: string;
    size?: 'auto' | 'tiny' | 'small' | 'normal' | 'large' | 'huge';
    id?: string;
}

// Export type for window TradingView
declare global {
    interface Window {
        TradingView?: {
            widget: new (options: TradingViewWidgetOptions) => IChartingLibraryWidget;
            version: () => string;
        };
    }
}
