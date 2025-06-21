/**
 * TypeScript interfaces for GoldenSignalsAI
 * 
 * Defines the structure of options signals and related data types
 */

export interface PreciseOptionsSignal {
    // Identification
    id: string;
    symbol: string;
    signal_id: string;
    generated_at: string;
    timestamp: string;

    // Trade Direction
    type: 'CALL' | 'PUT';
    signal_type: 'BUY_CALL' | 'BUY_PUT';
    confidence: number; // 0-100
    priority: 'HIGH' | 'MEDIUM' | 'LOW';

    // Precise Timing
    entry_window: {
        date: string;
        start_time: string;
        end_time: string;
    };
    hold_duration: string;
    expiration_warning: string;

    // Options Contract
    strike_price: number;
    expiration_date: string;
    contract_type: 'Weekly' | 'Monthly';
    max_premium: number;

    // Entry Levels
    current_price: number;
    entry_price: number;
    entry_trigger: number;
    entry_zone: [number, number];

    // Risk Management
    stop_loss: number;
    stop_loss_pct: number;
    position_size: number;
    max_risk_dollars: number;
    max_loss: number;

    // Profit Targets
    take_profit: number;
    targets: Array<{
        price: number;
        exit_pct: number;
    }>;
    risk_reward_ratio: number;

    // Exit Conditions
    exit_rules: string[];
    time_based_exits: {
        intraday: string;
        multi_day: string;
        expiration: string;
    };

    // Technical Justification
    setup_name: string;
    key_indicators: Record<string, string | number>;
    chart_patterns: string[];

    // Action Items
    alerts_to_set: string[];
    pre_entry_checklist: string[];
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
    priority?: 'ALL' | 'HIGH' | 'MEDIUM' | 'LOW';
    type?: 'ALL' | 'CALL' | 'PUT';
    signalType?: 'all' | 'CALL' | 'PUT';
    minConfidence?: number;
    maxRisk?: number;
    symbols?: string[];
    timeframe?: 'all' | 'INTRADAY' | 'SWING' | 'POSITION';
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