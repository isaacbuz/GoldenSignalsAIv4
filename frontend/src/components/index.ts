/**
 * Unified Components Index
 * 
 * Centralized exports for all consolidated components after cleanup and reorganization.
 * This replaces the need to import from multiple files and provides a single source of truth.
 */

// === AI COMPONENTS ===
// Main AI components with Golden Eye theme preserved
export { FloatingAIProphetWidget } from './AI/FloatingAIProphetWidget';
export { EnhancedAIProphetChat } from './AI/EnhancedAIProphetChat';
export { UnifiedAIChat } from './AI/UnifiedAIChat';
export { AIInsightsPanel } from './AI/AIInsightsPanel';
export { AIBrainDashboard } from './AI/AIBrainDashboard';
export { FloatingAIChatButton } from './AI/FloatingAIChatButton';
export { AISignalCard } from './AI/AISignalCard';
export { AIExplanationPanel } from './AI/AIExplanationPanel';
export { AIPredictionChart } from './AI/AIPredictionChart';
export { TransformerPredictionChart } from './AI/TransformerPredictionChart';
export { VoiceSignalAssistant } from './AI/VoiceSignalAssistant';

// === DASHBOARD COMPONENTS ===
// Consolidated dashboard components
export { UnifiedDashboard } from './Dashboard/UnifiedDashboard';
export { ProfessionalTradingDashboard } from './Dashboard/ProfessionalTradingDashboard';
export { LiveDataStatus } from './Dashboard/LiveDataStatus';
export { RiskMonitor } from './Dashboard/RiskMonitor';
export { QuickStats } from './Dashboard/QuickStats';
export { MainTradingChart } from './Dashboard/MainTradingChart';
export { SentimentGauge } from './Dashboard/SentimentGauge';
export { SymbolCard } from './Dashboard/SymbolCard';

// === CHART COMPONENTS ===
// Consolidated chart components
export { UnifiedChart } from './Chart/UnifiedChart';
export { MiniChart } from './Chart/MiniChart';
export { ChartShowcase } from './Chart/ChartShowcase';
export { AdvancedSignalChart } from './Chart/AdvancedSignalChart';
export { MassiveOptionsChart } from './Chart/MassiveOptionsChart';
export { TradingViewDatafeed } from './Chart/TradingViewDatafeed';

// === SIGNAL COMPONENTS ===
// All signal-related components
export { default as SignalList } from './Signals/SignalList';
export { default as SignalCard } from './Signals/SignalCard';
export { default as EnhancedSignalCard } from './Signals/EnhancedSignalCard';
export { SignalDetailsModal } from './Signals/SignalDetailsModal';
export { SignalConfidence } from './Signals/SignalConfidence';
export { SignalPriorityQueue } from './Signals/SignalPriorityQueue';
export { MarketScreener } from './Signals/MarketScreener';
export { SignalStream } from './Signals/SignalStream';

// === COMMON COMPONENTS ===
// Shared utility components
export { UnifiedSearchBar } from './Common/UnifiedSearchBar';
export { SimpleSearchBar } from './Common/SimpleSearchBar';
export { ProfessionalSearchBar } from './Common/ProfessionalSearchBar';
export { LoadingStatus } from './Common/LoadingStatus';
export { LoadingStates } from './Common/LoadingStates';
export { LoadingSkeletons } from './Common/LoadingSkeletons';
export { LoadingScreen } from './Common/LoadingScreen';
export { MetricCard } from './Common/MetricCard';
export { MiniChart as CommonMiniChart } from './Common/MiniChart';
export { NotificationCenter } from './Common/NotificationCenter';
export { WebSocketStatus } from './Common/WebSocketStatus';
export { Breadcrumbs } from './Common/Breadcrumbs';
export { PerformanceWrapper } from './Common/PerformanceWrapper';
export { ErrorBoundary } from './Common/ErrorBoundary';

// === LAYOUT COMPONENTS ===
// Navigation and layout components
export { TopBar } from './Layout/TopBar';
export { ModernTopBar } from './Layout/ModernTopBar';

// === AGENT COMPONENTS ===
// AI Agent related components
export { default as AgentConsensusFlow } from './Agents/AgentConsensusFlow';

// === CORE COMPONENTS ===
// Core design system components
export { Button } from './Core/Button';

// === UTILITY COMPONENTS ===
// Basic utility components
export { Card } from './Card';
export { Table } from './Table';
export { Modal } from './Modal';
export { Tooltip } from './Tooltip';

// === SPECIALIZED COMPONENTS ===
// Market and trading specific components
export { MarketInsights } from './MarketInsights/MarketInsights';
export { default as MarketNews } from './News/MarketNews';
export { default as PerformanceMetrics } from './Performance/PerformanceMetrics';
export { BacktestingInterface } from './Backtesting/BacktestingInterface';

// === GOLDEN EYE AI COMPONENTS ===
// Preserved Golden Eye AI Prophet components
export { AISignalProphet } from './GoldenEyeAI/AISignalProphet';

// === TYPE EXPORTS ===
// Export commonly used types
export type { DashboardMode, DashboardLayout } from './Dashboard/UnifiedDashboard';
export type { AIChatMode, AIModel } from './AI/UnifiedAIChat';
export type { SearchMode, SearchVariant } from './Common/UnifiedSearchBar';
export type { ChartType, ChartMode, ChartTheme } from './Chart/UnifiedChart';
export type { SignalData } from './Signals/SignalCard';
export type { EnhancedSignalData } from './Signals/SignalList';

/**
 * CONSOLIDATION SUMMARY:
 * 
 * ✅ REMOVED:
 * - AITradingLab/ (placeholder components)
 * - Charts/ (merged with Chart/)
 * - HybridDashboard/ (merged with Dashboard/)
 * - Professional/ (moved to Common/)
 * - Main/ (moved to Signals/)
 * - AISignalProphet.disabled/ (moved to GoldenEyeAI/)
 * - Alert.disabled/ (removed)
 * - Empty directories
 * 
 * ✅ PRESERVED:
 * - Golden Eye AI Prophet theme and components
 * - All functional components
 * - Design system components
 * - Chart components (consolidated)
 * - Dashboard components (consolidated)
 * 
 * ✅ ORGANIZED:
 * - AI components in AI/
 * - Dashboard components in Dashboard/
 * - Chart components in Chart/
 * - Signal components in Signals/
 * - Common utilities in Common/
 * - Golden Eye components in GoldenEyeAI/
 * 
 * Migration Guide:
 * Use the unified components for better maintainability:
 * - Single source of truth for all components
 * - Reduced bundle size through consolidation
 * - Better maintainability with organized structure
 * - Consistent API across similar components
 * - Enhanced features through consolidation
 */
export { SignalCard } from './SignalCard/SignalCard';
export { RealTimeFeed } from './RealTimeFeed/RealTimeFeed';
export { FloatingOrbAssistant } from './FloatingOrbAssistant/FloatingOrbAssistant';
export { AdvancedChart } from './AdvancedChart/AdvancedChart';
export { CentralChart } from './CentralChart/CentralChart';
export { FloatingOrb } from './FloatingOrb/FloatingOrb';
export { OptionsChainTable } from './OptionsChainTable/OptionsChainTable';
export { PredictionTimeline } from './PredictionTimeline/PredictionTimeline';
export { SignalPanel } from './SignalPanel/SignalPanel';
export { TradeSearch } from './TradeSearch/TradeSearch';
export { ProphetOrb } from './ProphetOrb/ProphetOrb';
export { OptionsPanel } from './OptionsPanel/OptionsPanel';
export { AnalysisLegend } from './AnalysisLegend/AnalysisLegend';