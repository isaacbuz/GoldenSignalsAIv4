/**
 * TypeScript interfaces for GoldenSignalsAI
 *
 * Defines the structure of options signals and related data types
 */

export interface PreciseOptionsSignal {
    id: string;
    signal_id?: string;
    symbol: string;
    signal_type: 'BUY_CALL' | 'BUY_PUT';
    type?: string;
    strike_price: number;
    expiration_date: string;
    confidence: number;
    timestamp: string;
    reasoning?: string;

    // Additional properties used by components
    entry_price?: number;
    stop_loss?: number;
    take_profit?: number;
    risk_reward_ratio?: number;
    max_loss?: number;
    setup_name?: string;
    entry_window?: {
        start_time: string;
        end_time: string;
    };
    targets?: Array<{
        price: number;
        probability: number;
    }>;
}

export interface PerformanceMetrics {
    winRate: number;
    avgReturn: number;
    totalSignals: number;
    successfulSignals: number;
    avgHoldTime: string;
    bestPerformer: {
        symbol: string;
        return: number;
    };
    worstPerformer: {
        symbol: string;
        return: number;
    };
}

export interface RiskMetrics {
    activePositions: number;
    totalExposure: number;
    maxDrawdown: number;
    sharpeRatio: number;
    currentRisk: number;
    riskLimit: number;
    utilizationPct: number;
}

export interface MarketConditions {
    vix: number;
    marketTrend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    sectorRotation: string[];
    unusualActivity: Array<{
        symbol: string;
        type: string;
        description: string;
    }>;
}

export interface SignalFilters {
    signalType?: 'all' | 'BUY_CALL' | 'BUY_PUT';
    minConfidence?: number;
    timeframe?: 'all' | string;
}

export interface AIInsight {
    price: number;
    type: 'SUPPORT' | 'RESISTANCE' | 'ENTRY' | 'TARGET' | 'STOP';
    confidence: number;
    label: string;
    reasoning?: string;
}

export interface SignalAlert {
    id: string;
    signalId: string;
    type: 'ENTRY' | 'EXIT' | 'STOP_LOSS' | 'TARGET_HIT' | 'TIME_BASED';
    price: number;
    message: string;
    timestamp: string;
    acknowledged: boolean;
}

export interface TradingSession {
    name: string;
    startTime: string;
    endTime: string;
    isActive: boolean;
}

export interface OptionsGreeks {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
    iv: number; // Implied Volatility
}

export interface SignalPerformance {
    signalId: string;
    entryPrice: number;
    exitPrice?: number;
    currentPrice: number;
    pnl: number;
    pnlPercent: number;
    status: 'ACTIVE' | 'CLOSED' | 'STOPPED_OUT';
    entryTime: string;
    exitTime?: string;
    holdTime: string;
}

export interface SignalDetails {
    entry_price: number;
    target_price: number;
    stop_loss: number;
    reason: string;
}

export interface Signal {
    signal_id: string;
    symbol: string;
    signal_type: string;
    confidence_score: number;
    timestamp: string;
    timeframe: string;
    signal_details: SignalDetails;
}
