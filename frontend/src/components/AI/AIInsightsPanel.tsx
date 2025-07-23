/**
 * Enhanced AI Insights Panel
 * Real-time market intelligence with advanced categorization and visual improvements
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    Box,
    Card,
    CardContent,
    CardHeader,
    Typography,
    Chip,
    Avatar,
    Stack,
    IconButton,
    Tooltip,
    Badge,
    LinearProgress,
    Collapse,
    Button,
    Menu,
    MenuItem,
    Divider,
    Alert,
    useTheme,
    alpha,
    Skeleton,
    Grid,
    Paper,
    Fade,
    Zoom,
} from '@mui/material';
import {
    Psychology,
    TrendingUp,
    TrendingDown,
    Warning,
    Lightbulb,
    Timeline,
    Refresh,
    ExpandMore,
    ExpandLess,
    FilterList,
    Star,
    StarBorder,
    Visibility,
    VisibilityOff,
    Analytics,
    Speed,
    AutoAwesome,
    BrightnessMedium,
    ErrorOutline,
    CheckCircle,
    Info,
    PriorityHigh,
    Schedule,
    TrendingFlat,
    ShowChart,
    Assessment,
    Notifications,
    NotificationsActive,
    Circle,
    FiberManualRecord,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api/apiClient';
import { ChartSignal } from '../Chart/UnifiedChart';
import logger from '../../services/logger';


// Enhanced AI Insight interface
export interface EnhancedAIInsight {
    id: string;
    symbol: string;
    type: 'Bullish' | 'Bearish' | 'Warning' | 'Opportunity' | 'Neutral' | 'Technical' | 'Fundamental';
    priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
    confidence: number;
    signal: string;
    reasoning: string;
    summary: string;
    timestamp: string;

    // Enhanced fields
    category: 'market' | 'technical' | 'fundamental' | 'sentiment' | 'risk' | 'opportunity';
    impact: 'high' | 'medium' | 'low';
    timeframe: 'immediate' | 'short' | 'medium' | 'long';
    tags: string[];
    relatedSignals: string[];
    accuracy?: number;
    backtestData?: any;

    // Real-time indicators
    isLive?: boolean;
    isNew?: boolean;
    isExpiring?: boolean;
    expiryTime?: string;

    // Visual enhancements
    color?: string;
    icon?: string;
    animation?: boolean;
}

export interface AIInsightsPanelProps {
    symbol?: string;
    signals?: ChartSignal[];
    maxInsights?: number;
    enableLiveUpdates?: boolean;
    showFilters?: boolean;
    showPriorityBadges?: boolean;
    autoRefresh?: boolean;
    refreshInterval?: number;
    height?: number;
    className?: string;
    onInsightClick?: (insight: EnhancedAIInsight) => void;
}

type InsightFilter = 'all' | 'critical' | 'high' | 'bullish' | 'bearish' | 'warning' | 'opportunity' | 'live';
type SortBy = 'priority' | 'confidence' | 'timestamp' | 'impact';

const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({
    symbol,
    signals = [],
    maxInsights = 10,
    enableLiveUpdates = true,
    showFilters = true,
    showPriorityBadges = true,
    autoRefresh = true,
    refreshInterval = 30000,
    height = 400,
    className,
    onInsightClick,
}) => {
    const theme = useTheme();
    const [expanded, setExpanded] = useState<string | false>(false);
    const [filter, setFilter] = useState<InsightFilter>('all');
    const [sortBy, setSortBy] = useState<SortBy>('priority');
    const [showExpired, setShowExpired] = useState(false);
    const [favoriteInsights, setFavoriteInsights] = useState<Set<string>>(new Set());
    const [hiddenInsights, setHiddenInsights] = useState<Set<string>>(new Set());
    const [filterMenuAnchor, setFilterMenuAnchor] = useState<null | HTMLElement>(null);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [liveInsights, setLiveInsights] = useState<EnhancedAIInsight[]>([]);

    // Query for AI insights with error handling
    const {
        data: apiInsights,
        isLoading,
        error,
        refetch
    } = useQuery({
        queryKey: ['ai-insights', symbol],
        queryFn: async () => {
            try {
                const response = await apiClient.getAIInsights(symbol);

                // Handle different response formats
                let insights: any[] = [];

                if (Array.isArray(response)) {
                    insights = response;
                } else if (response && typeof response === 'object') {
                    // Handle object response format like {symbol: 'SPY', insights: [...], last_updated: '...'}
                    if (Array.isArray((response as any).insights)) {
                        insights = (response as any).insights;
                    } else if (Array.isArray((response as any).data)) {
                        insights = (response as any).data;
                    } else {
                        logger.warn('API returned non-array insights:', response);
                        return [];
                    }
                } else {
                    logger.warn('API returned unexpected response format:', response);
                    return [];
                }

                return insights.map(insight => enhanceInsight(insight));
            } catch (error) {
                logger.error('Failed to fetch AI insights:', error);
                return [];
            }
        },
        refetchInterval: autoRefresh ? refreshInterval : false,
        staleTime: 15000, // 15 seconds
        retry: 3,
        retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
    });

    // Enhance raw insight data
    const enhanceInsight = useCallback((rawInsight: any): EnhancedAIInsight => {
        const now = Date.now();
        const insightTime = new Date(rawInsight.timestamp).getTime();
        const ageMinutes = (now - insightTime) / (1000 * 60);

        // Determine category from type and content
        const category = determineCategory(rawInsight.type, rawInsight.reasoning);

        // Calculate impact based on confidence and priority
        const impact = rawInsight.confidence >= 80 && rawInsight.priority === 'CRITICAL' ? 'high' :
            rawInsight.confidence >= 60 && ['CRITICAL', 'HIGH'].includes(rawInsight.priority) ? 'medium' : 'low';

        // Determine timeframe from reasoning content
        const timeframe = determineTimeframe(rawInsight.reasoning);

        // Generate tags from reasoning and signal content
        const tags = generateTags(rawInsight.reasoning, rawInsight.signal);

        return {
            id: rawInsight.id || `insight_${Date.now()}_${Math.random()}`,
            symbol: rawInsight.symbol,
            type: rawInsight.type || 'Neutral',
            priority: rawInsight.priority || 'MEDIUM',
            confidence: rawInsight.confidence || 50,
            signal: rawInsight.signal || 'No signal',
            reasoning: rawInsight.reasoning || 'No reasoning provided',
            summary: rawInsight.summary || rawInsight.signal,
            timestamp: rawInsight.timestamp || new Date().toISOString(),

            category,
            impact,
            timeframe,
            tags,
            relatedSignals: findRelatedSignals(rawInsight.symbol, signals),

            isLive: ageMinutes < 5,
            isNew: ageMinutes < 1,
            isExpiring: rawInsight.expiryTime && new Date(rawInsight.expiryTime).getTime() - now < 300000, // 5 minutes
            expiryTime: rawInsight.expiryTime,

            color: getInsightColor(rawInsight.type, rawInsight.priority),
            animation: ageMinutes < 1,
        };
    }, [signals]);

    // Helper functions
    const determineCategory = (type: string, reasoning: string): EnhancedAIInsight['category'] => {
        if (!reasoning || typeof reasoning !== 'string') {
            return 'market'; // Default category when reasoning is invalid
        }

        const lowerReasoning = reasoning.toLowerCase();
        if (lowerReasoning.includes('technical') || lowerReasoning.includes('chart') || lowerReasoning.includes('indicator')) return 'technical';
        if (lowerReasoning.includes('earnings') || lowerReasoning.includes('fundamental') || lowerReasoning.includes('valuation')) return 'fundamental';
        if (lowerReasoning.includes('sentiment') || lowerReasoning.includes('social') || lowerReasoning.includes('news')) return 'sentiment';
        if (lowerReasoning.includes('risk') || lowerReasoning.includes('volatility') || lowerReasoning.includes('drawdown')) return 'risk';
        if (lowerReasoning.includes('opportunity') || lowerReasoning.includes('breakout') || lowerReasoning.includes('momentum')) return 'opportunity';
        return 'market';
    };

    const determineTimeframe = (reasoning: string): EnhancedAIInsight['timeframe'] => {
        if (!reasoning || typeof reasoning !== 'string') {
            return 'short'; // Default timeframe when reasoning is invalid
        }

        const lowerReasoning = reasoning.toLowerCase();
        if (lowerReasoning.includes('immediate') || lowerReasoning.includes('now') || lowerReasoning.includes('urgent')) return 'immediate';
        if (lowerReasoning.includes('short') || lowerReasoning.includes('intraday') || lowerReasoning.includes('today')) return 'short';
        if (lowerReasoning.includes('medium') || lowerReasoning.includes('week') || lowerReasoning.includes('monthly')) return 'medium';
        if (lowerReasoning.includes('long') || lowerReasoning.includes('quarter') || lowerReasoning.includes('yearly')) return 'long';
        return 'short';
    };

    const generateTags = (reasoning: string, signal: string): string[] => {
        const safeReasoning = reasoning || '';
        const safeSignal = signal || '';
        const text = `${safeReasoning} ${safeSignal}`.toLowerCase();
        const tags: string[] = [];

        // Technical tags
        if (text.includes('breakout')) tags.push('breakout');
        if (text.includes('support') || text.includes('resistance')) tags.push('levels');
        if (text.includes('momentum')) tags.push('momentum');
        if (text.includes('volume')) tags.push('volume');
        if (text.includes('volatility')) tags.push('volatility');

        // Fundamental tags
        if (text.includes('earnings')) tags.push('earnings');
        if (text.includes('revenue')) tags.push('revenue');
        if (text.includes('guidance')) tags.push('guidance');

        // Market tags
        if (text.includes('bullish')) tags.push('bullish');
        if (text.includes('bearish')) tags.push('bearish');
        if (text.includes('neutral')) tags.push('neutral');

        return tags.slice(0, 3); // Limit to 3 tags
    };

    const findRelatedSignals = (insightSymbol: string, signals: ChartSignal[]): string[] => {
        return signals
            .filter(signal => signal.symbol === insightSymbol)
            .map(signal => signal.id)
            .slice(0, 3);
    };

    const getInsightColor = (type: string, priority: string): string => {
        const colors = {
            'Bullish': theme.palette.success.main,
            'Bearish': theme.palette.error.main,
            'Warning': theme.palette.warning.main,
            'Opportunity': theme.palette.info.main,
            'Neutral': theme.palette.grey[500],
            'Technical': theme.palette.primary.main,
            'Fundamental': theme.palette.secondary.main,
        };

        const baseColor = colors[type as keyof typeof colors] || theme.palette.grey[500];
        const intensity = priority === 'CRITICAL' ? 1 : priority === 'HIGH' ? 0.8 : priority === 'MEDIUM' ? 0.6 : 0.4;

        return alpha(baseColor, intensity);
    };

    // Process and filter insights
    const processedInsights = useMemo(() => {
        // Ensure apiInsights is always an array
        const safeApiInsights = Array.isArray(apiInsights) ? apiInsights : [];
        const allInsights = [...liveInsights, ...safeApiInsights];

        // Remove duplicates
        const uniqueInsights = allInsights.filter((insight, index, self) =>
            index === self.findIndex(i => i.id === insight.id)
        );

        // Apply filters
        let filtered = uniqueInsights.filter(insight => {
            if (hiddenInsights.has(insight.id)) return false;
            if (!showExpired && insight.isExpiring) return false;

            switch (filter) {
                case 'critical':
                    return insight.priority === 'CRITICAL';
                case 'high':
                    return ['CRITICAL', 'HIGH'].includes(insight.priority);
                case 'bullish':
                    return insight.type === 'Bullish';
                case 'bearish':
                    return insight.type === 'Bearish';
                case 'warning':
                    return insight.type === 'Warning';
                case 'opportunity':
                    return insight.type === 'Opportunity';
                case 'live':
                    return insight.isLive;
                default:
                    return true;
            }
        });

        // Sort insights
        filtered.sort((a, b) => {
            switch (sortBy) {
                case 'priority':
                    const priorityOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
                    return priorityOrder[b.priority] - priorityOrder[a.priority];
                case 'confidence':
                    return b.confidence - a.confidence;
                case 'timestamp':
                    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
                case 'impact':
                    const impactOrder = { 'high': 3, 'medium': 2, 'low': 1 };
                    return impactOrder[b.impact] - impactOrder[a.impact];
                default:
                    return 0;
            }
        });

        return filtered.slice(0, maxInsights);
    }, [liveInsights, apiInsights, filter, sortBy, hiddenInsights, showExpired, maxInsights]);

    // Handle refresh
    const handleRefresh = useCallback(async () => {
        setIsRefreshing(true);
        await refetch();
        setTimeout(() => setIsRefreshing(false), 1000);
    }, [refetch]);

    // Toggle favorite
    const toggleFavorite = useCallback((insightId: string) => {
        setFavoriteInsights(prev => {
            const newSet = new Set(prev);
            if (newSet.has(insightId)) {
                newSet.delete(insightId);
            } else {
                newSet.add(insightId);
            }
            return newSet;
        });
    }, []);

    // Hide insight
    const hideInsight = useCallback((insightId: string) => {
        setHiddenInsights(prev => new Set([...prev, insightId]));
    }, []);

    // Get priority icon
    const getPriorityIcon = useCallback((priority: string) => {
        switch (priority) {
            case 'CRITICAL':
                return <PriorityHigh color="error" />;
            case 'HIGH':
                return <Warning color="warning" />;
            case 'MEDIUM':
                return <Info color="info" />;
            case 'LOW':
                return <Circle color="disabled" />;
            default:
                return <Info color="info" />;
        }
    }, []);

    // Get type icon
    const getTypeIcon = useCallback((type: string) => {
        switch (type) {
            case 'Bullish':
                return <TrendingUp color="success" />;
            case 'Bearish':
                return <TrendingDown color="error" />;
            case 'Warning':
                return <Warning color="warning" />;
            case 'Opportunity':
                return <Lightbulb color="info" />;
            case 'Technical':
                return <ShowChart color="primary" />;
            case 'Fundamental':
                return <Assessment color="secondary" />;
            default:
                return <TrendingFlat color="disabled" />;
        }
    }, []);

    // Render insight card
    const renderInsightCard = useCallback((insight: EnhancedAIInsight) => {
        const isExpanded = expanded === insight.id;
        const isFavorite = favoriteInsights.has(insight.id);

        return (
            <Zoom in={true} key={insight.id}>
                <Card
                    sx={{
                        mb: 1,
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        borderLeft: `4px solid ${insight.color}`,
                        backgroundColor: insight.isNew ? alpha(theme.palette.primary.main, 0.05) : 'background.paper',
                        '&:hover': {
                            backgroundColor: alpha(theme.palette.primary.main, 0.08),
                            transform: 'translateX(2px)',
                        },
                        ...(insight.animation && {
                            animation: 'pulse 2s infinite',
                        }),
                    }}
                    onClick={() => {
                        setExpanded(isExpanded ? false : insight.id);
                        onInsightClick?.(insight);
                    }}
                >
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                        {/* Header */}
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {getTypeIcon(insight.type)}
                                <Typography variant="subtitle2" fontWeight="bold">
                                    {insight.symbol}
                                </Typography>
                                <Chip
                                    label={insight.type}
                                    size="small"
                                    sx={{
                                        backgroundColor: insight.color,
                                        color: 'white',
                                        fontWeight: 'bold'
                                    }}
                                />
                                {insight.isLive && (
                                    <Badge
                                        badgeContent={<FiberManualRecord sx={{ fontSize: 6 }} />}
                                        color="success"
                                    >
                                        <Chip label="LIVE" size="small" color="success" />
                                    </Badge>
                                )}
                                {insight.isNew && (
                                    <Chip label="NEW" size="small" color="info" />
                                )}
                            </Box>

                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {showPriorityBadges && getPriorityIcon(insight.priority)}
                                <Typography variant="body2" fontWeight="bold" color="primary">
                                    {insight.confidence}%
                                </Typography>
                                <IconButton
                                    size="small"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        toggleFavorite(insight.id);
                                    }}
                                >
                                    {isFavorite ? <Star color="warning" /> : <StarBorder />}
                                </IconButton>
                            </Box>
                        </Box>

                        {/* Summary */}
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            {insight.summary}
                        </Typography>

                        {/* Tags */}
                        <Box sx={{ display: 'flex', gap: 0.5, mb: 1 }}>
                            {insight.tags.map(tag => (
                                <Chip
                                    key={tag}
                                    label={tag}
                                    size="small"
                                    variant="outlined"
                                    sx={{ fontSize: 10 }}
                                />
                            ))}
                        </Box>

                        {/* Metadata */}
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Typography variant="caption" color="text.secondary">
                                {insight.category} • {insight.timeframe} • {new Date(insight.timestamp).toLocaleTimeString()}
                            </Typography>
                            <IconButton size="small">
                                {isExpanded ? <ExpandLess /> : <ExpandMore />}
                            </IconButton>
                        </Box>

                        {/* Expanded Content */}
                        <Collapse in={isExpanded}>
                            <Divider sx={{ my: 1 }} />
                            <Typography variant="body2" sx={{ mb: 1 }}>
                                <strong>Reasoning:</strong> {insight.reasoning}
                            </Typography>
                            <Typography variant="body2" sx={{ mb: 1 }}>
                                <strong>Signal:</strong> {insight.signal}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                                <Chip
                                    label={`Impact: ${insight.impact}`}
                                    size="small"
                                    color={insight.impact === 'high' ? 'error' : insight.impact === 'medium' ? 'warning' : 'default'}
                                />
                                <Chip
                                    label={`Priority: ${insight.priority}`}
                                    size="small"
                                    color={insight.priority === 'CRITICAL' ? 'error' : insight.priority === 'HIGH' ? 'warning' : 'default'}
                                />
                                <Chip
                                    label={`Timeframe: ${insight.timeframe}`}
                                    size="small"
                                    variant="outlined"
                                />
                            </Box>
                        </Collapse>
                    </CardContent>
                </Card>
            </Zoom>
        );
    }, [expanded, favoriteInsights, theme, showPriorityBadges, toggleFavorite, onInsightClick, getTypeIcon, getPriorityIcon]);

    // Render loading skeleton
    const renderLoadingSkeleton = () => (
        <Box sx={{ p: 1 }}>
            {Array.from({ length: 3 }).map((_, index) => (
                <Card key={index} sx={{ mb: 1 }}>
                    <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Skeleton variant="circular" width={24} height={24} />
                            <Skeleton variant="text" width="30%" />
                            <Skeleton variant="rectangular" width={60} height={20} />
                        </Box>
                        <Skeleton variant="text" width="80%" />
                        <Skeleton variant="text" width="60%" />
                    </CardContent>
                </Card>
            ))}
        </Box>
    );

    return (
        <Box className={className} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <CardHeader
                avatar={
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                        <Psychology />
                    </Avatar>
                }
                title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h6">AI Insights</Typography>
                        <AutoAwesome sx={{ color: 'primary.main' }} />
                        {processedInsights.some(i => i.isLive) && (
                            <Badge
                                badgeContent={processedInsights.filter(i => i.isLive).length}
                                color="success"
                            >
                                <NotificationsActive color="success" />
                            </Badge>
                        )}
                    </Box>
                }
                subheader={`${processedInsights.length} insights • ${symbol || 'All symbols'}`}
                action={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {showFilters && (
                            <IconButton
                                size="small"
                                onClick={(e) => setFilterMenuAnchor(e.currentTarget)}
                            >
                                <FilterList />
                            </IconButton>
                        )}
                        <IconButton
                            size="small"
                            onClick={handleRefresh}
                            disabled={isRefreshing}
                        >
                            <Refresh sx={{
                                animation: isRefreshing ? 'spin 1s linear infinite' : 'none'
                            }} />
                        </IconButton>
                    </Box>
                }
            />

            {/* Filter Menu */}
            <Menu
                anchorEl={filterMenuAnchor}
                open={Boolean(filterMenuAnchor)}
                onClose={() => setFilterMenuAnchor(null)}
            >
                <MenuItem onClick={() => { setFilter('all'); setFilterMenuAnchor(null); }}>
                    All Insights
                </MenuItem>
                <MenuItem onClick={() => { setFilter('critical'); setFilterMenuAnchor(null); }}>
                    Critical Only
                </MenuItem>
                <MenuItem onClick={() => { setFilter('high'); setFilterMenuAnchor(null); }}>
                    High Priority
                </MenuItem>
                <MenuItem onClick={() => { setFilter('live'); setFilterMenuAnchor(null); }}>
                    Live Only
                </MenuItem>
                <Divider />
                <MenuItem onClick={() => { setFilter('bullish'); setFilterMenuAnchor(null); }}>
                    Bullish
                </MenuItem>
                <MenuItem onClick={() => { setFilter('bearish'); setFilterMenuAnchor(null); }}>
                    Bearish
                </MenuItem>
                <MenuItem onClick={() => { setFilter('opportunity'); setFilterMenuAnchor(null); }}>
                    Opportunities
                </MenuItem>
                <MenuItem onClick={() => { setFilter('warning'); setFilterMenuAnchor(null); }}>
                    Warnings
                </MenuItem>
            </Menu>

            {/* Content */}
            <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
                {isLoading && renderLoadingSkeleton()}

                {error && (
                    <Alert severity="error" sx={{ m: 1 }}>
                        Failed to load AI insights. Please try again.
                    </Alert>
                )}

                {!isLoading && !error && processedInsights.length === 0 && (
                    <Alert severity="info" sx={{ m: 1 }}>
                        No AI insights available for the selected filters.
                    </Alert>
                )}

                {processedInsights.map(insight => renderInsightCard(insight))}
            </Box>
        </Box>
    );
};

export default AIInsightsPanel;
export { AIInsightsPanel };
