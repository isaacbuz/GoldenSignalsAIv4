/**
 * Enhanced SignalList Component
 * Optimized for high-frequency real-time signal updates
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Avatar,
  Tooltip,
  Badge,
  Divider,
  LinearProgress,
  Stack,
  useTheme,
  alpha,
  Collapse,
  Button,
  Menu,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  TextField,
  InputAdornment,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  Psychology,
  Speed,
  FilterList,
  Search,
  MoreVert,
  Refresh,
  Star,
  StarBorder,
  Visibility,
  VisibilityOff,
  AccessTime,
  ShowChart,
  Analytics,
  TrendingFlat,
  Circle,
  RadioButtonChecked,
  FiberManualRecord,
} from '@mui/icons-material';
import { FixedSizeList as VirtualList } from 'react-window';
import { SignalData } from './SignalCard';

// Enhanced signal interface with performance metrics
export interface EnhancedSignalData extends SignalData {
  // Performance metrics
  riskRewardRatio?: number;
  technicalScore?: number;
  sentimentScore?: number;
  consensusScore?: number;
  volatility?: number;
  liquidity?: number;

  // Real-time indicators
  isLive?: boolean;
  isNew?: boolean;
  isExpiring?: boolean;
  timeToExpiry?: number;

  // Enhanced metadata
  marketCap?: number;
  sector?: string;
  industry?: string;
  tags?: string[];

  // Performance tracking
  accuracy?: number;
  historicalPerformance?: number;
  backtestResults?: any;
}

export interface SignalListProps {
  signals: EnhancedSignalData[];
  onSignalSelect?: (signal: EnhancedSignalData) => void;
  selectedSignal?: EnhancedSignalData | null;
  maxItems?: number;
  enableVirtualization?: boolean;
  enableFiltering?: boolean;
  enableSorting?: boolean;
  showPerformanceMetrics?: boolean;
  showLiveIndicator?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  height?: number;
  onRefresh?: () => void;
  className?: string;
}

type SortField = 'confidence' | 'timestamp' | 'symbol' | 'performance' | 'risk';
type SortDirection = 'asc' | 'desc';
type FilterType = 'all' | 'BUY' | 'SELL' | 'HOLD' | 'live' | 'strong' | 'moderate' | 'weak';

const SignalList: React.FC<SignalListProps> = ({
  signals,
  onSignalSelect,
  selectedSignal,
  maxItems = 50,
  enableVirtualization = true,
  enableFiltering = true,
  enableSorting = true,
  showPerformanceMetrics = true,
  showLiveIndicator = true,
  autoRefresh = false,
  refreshInterval = 30000,
  height = 400,
  onRefresh,
  className,
}) => {
  const theme = useTheme();
  const [sortField, setSortField] = useState<SortField>('confidence');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [filterType, setFilterType] = useState<FilterType>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [watchlist, setWatchlist] = useState<Set<string>>(new Set());
  const [hiddenSignals, setHiddenSignals] = useState<Set<string>>(new Set());
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Performance optimization refs
  const listRef = useRef<any>(null);
  const lastUpdateRef = useRef<number>(0);
  const animationFrameRef = useRef<number>();

  // Auto-refresh logic
  useEffect(() => {
    if (autoRefresh && onRefresh) {
      const interval = setInterval(() => {
        setIsRefreshing(true);
        onRefresh();
        setTimeout(() => setIsRefreshing(false), 1000);
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, onRefresh, refreshInterval]);

  // Memoized filtered and sorted signals
  const processedSignals = useMemo(() => {
    let filtered = signals.filter(signal => {
      // Apply filters
      if (filterType !== 'all') {
        if (filterType === 'live' && !signal.isLive) return false;
        if (filterType === 'strong' && signal.strength !== 'STRONG') return false;
        if (filterType === 'moderate' && signal.strength !== 'MODERATE') return false;
        if (filterType === 'weak' && signal.strength !== 'WEAK') return false;
        if (['BUY', 'SELL', 'HOLD'].includes(filterType) && signal.action !== filterType) return false;
      }

      // Apply search
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        if (!signal.symbol.toLowerCase().includes(query) &&
          !signal.reasoning?.toLowerCase().includes(query) &&
          !signal.source?.toLowerCase().includes(query)) {
          return false;
        }
      }

      // Apply hidden signals filter
      if (hiddenSignals.has(signal.id)) return false;

      return true;
    });

    // Sort signals
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;

      switch (sortField) {
        case 'confidence':
          aValue = a.confidence;
          bValue = b.confidence;
          break;
        case 'timestamp':
          aValue = new Date(a.timestamp).getTime();
          bValue = new Date(b.timestamp).getTime();
          break;
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'performance':
          aValue = a.historicalPerformance || 0;
          bValue = b.historicalPerformance || 0;
          break;
        case 'risk':
          aValue = a.riskRewardRatio || 0;
          bValue = b.riskRewardRatio || 0;
          break;
        default:
          aValue = a.confidence;
          bValue = b.confidence;
      }

      const comparison = aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return filtered.slice(0, maxItems);
  }, [signals, filterType, searchQuery, sortField, sortDirection, hiddenSignals, maxItems]);

  // Debounced search to improve performance
  const debouncedSearchQuery = useMemo(() => {
    const timeoutId = setTimeout(() => {
      return searchQuery;
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  // Handle signal selection with performance optimization
  const handleSignalSelect = useCallback((signal: EnhancedSignalData) => {
    // Debounce rapid selections
    const now = Date.now();
    if (now - lastUpdateRef.current < 100) return;
    lastUpdateRef.current = now;

    // Use animation frame for smooth UI updates
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    animationFrameRef.current = requestAnimationFrame(() => {
      onSignalSelect?.(signal);
    });
  }, [onSignalSelect]);

  // Watchlist management
  const toggleWatchlist = useCallback((symbol: string) => {
    setWatchlist(prev => {
      const newSet = new Set(prev);
      if (newSet.has(symbol)) {
        newSet.delete(symbol);
      } else {
        newSet.add(symbol);
      }
      return newSet;
    });
  }, []);

  // Hide/show signals
  const toggleSignalVisibility = useCallback((signalId: string) => {
    setHiddenSignals(prev => {
      const newSet = new Set(prev);
      if (newSet.has(signalId)) {
        newSet.delete(signalId);
      } else {
        newSet.add(signalId);
      }
      return newSet;
    });
  }, []);

  // Get signal color based on action and strength
  const getSignalColor = useCallback((signal: EnhancedSignalData) => {
    const baseColors = {
      BUY: theme.palette.success.main,
      SELL: theme.palette.error.main,
      HOLD: theme.palette.warning.main,
    };

    const intensity = signal.strength === 'STRONG' ? 1 :
      signal.strength === 'MODERATE' ? 0.7 : 0.5;

    return alpha(baseColors[signal.action] || theme.palette.info.main, intensity);
  }, [theme]);

  // Get signal icon
  const getSignalIcon = useCallback((signal: EnhancedSignalData) => {
    const iconProps = {
      sx: {
        color: getSignalColor(signal),
        fontSize: 20
      }
    };

    switch (signal.action) {
      case 'BUY':
        return <TrendingUp {...iconProps} />;
      case 'SELL':
        return <TrendingDown {...iconProps} />;
      case 'HOLD':
        return <TrendingFlat {...iconProps} />;
      default:
        return <Timeline {...iconProps} />;
    }
  }, [getSignalColor]);

  // Memoized render function to prevent unnecessary re-renders
  const MemoizedSignalItem = React.memo(({ signal, index, style }: { signal: EnhancedSignalData; index: number; style: any }) => {
    const isSelected = selectedSignal?.id === signal.id;
    const isInWatchlist = watchlist.has(signal.symbol);

    return (
      <div style={style}>
        <Card
          sx={{
            m: 0.5,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: isSelected ? alpha(theme.palette.primary.main, 0.1) : 'background.paper',
            borderLeft: `4px solid ${getSignalColor(signal)}`,
            '&:hover': {
              backgroundColor: alpha(theme.palette.primary.main, 0.05),
              transform: 'translateX(2px)',
            },
          }}
          onClick={() => handleSignalSelect(signal)}
        >
          <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              {/* Signal Info */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
                <Avatar sx={{
                  width: 32,
                  height: 32,
                  bgcolor: getSignalColor(signal),
                  fontSize: 14,
                  fontWeight: 'bold'
                }}>
                  {signal.symbol.slice(0, 2)}
                </Avatar>

                <Box sx={{ flex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {signal.symbol}
                    </Typography>
                    {getSignalIcon(signal)}
                    {signal.isLive && showLiveIndicator && (
                      <Badge
                        badgeContent={<FiberManualRecord sx={{ fontSize: 8 }} />}
                        color="success"
                        sx={{ '& .MuiBadge-badge': { right: -6, top: 4 } }}
                      >
                        <Chip label="LIVE" size="small" color="success" />
                      </Badge>
                    )}
                    {signal.isNew && (
                      <Chip label="NEW" size="small" color="info" />
                    )}
                  </Box>

                  <Typography variant="caption" color="text.secondary">
                    {signal.source} • {new Date(signal.timestamp).toLocaleTimeString()}
                  </Typography>
                </Box>
              </Box>

              {/* Confidence & Performance */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="body2" fontWeight="bold" color="primary">
                    {signal.confidence.toFixed(1)}%
                  </Typography>
                  {showPerformanceMetrics && signal.historicalPerformance && (
                    <Typography variant="caption" color="text.secondary">
                      {signal.historicalPerformance > 0 ? '+' : ''}{signal.historicalPerformance.toFixed(1)}%
                    </Typography>
                  )}
                </Box>

                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuAnchor(e.currentTarget);
                  }}
                >
                  <MoreVert fontSize="small" />
                </IconButton>
              </Box>
            </Box>

            {/* Signal Details */}
            {signal.reasoning && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{
                  mt: 1,
                  display: 'block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}
              >
                {signal.reasoning}
              </Typography>
            )}

            {/* Performance Metrics */}
            {showPerformanceMetrics && (
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                {signal.riskRewardRatio && (
                  <Chip
                    label={`R/R: ${signal.riskRewardRatio.toFixed(2)}`}
                    size="small"
                    variant="outlined"
                  />
                )}
                {signal.technicalScore && (
                  <Chip
                    label={`TA: ${signal.technicalScore.toFixed(0)}`}
                    size="small"
                    variant="outlined"
                  />
                )}
                {signal.sentimentScore && (
                  <Chip
                    label={`Sentiment: ${signal.sentimentScore.toFixed(0)}`}
                    size="small"
                    variant="outlined"
                  />
                )}
              </Box>
            )}
          </CardContent>
        </Card>
      </div>
    );
  });

  // Render individual signal item
  const renderSignalItem = useCallback(({ index, style }: { index: number; style: any }) => {
    const signal = processedSignals[index];
    return <MemoizedSignalItem signal={signal} index={index} style={style} />;
  }, [processedSignals, selectedSignal, watchlist, theme, handleSignalSelect, getSignalColor, getSignalIcon, showLiveIndicator, showPerformanceMetrics]);

  // Render loading skeleton
  const renderLoadingSkeleton = () => (
    <Box sx={{ p: 1 }}>
      {Array.from({ length: 5 }).map((_, index) => (
        <Card key={index} sx={{ m: 0.5 }}>
          <CardContent sx={{ p: 1.5 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Skeleton variant="circular" width={32} height={32} />
              <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="60%" />
                <Skeleton variant="text" width="40%" />
              </Box>
              <Skeleton variant="text" width="20%" />
            </Box>
          </CardContent>
        </Card>
      ))}
    </Box>
  );

  // Render filter controls
  const renderFilterControls = () => (
    <Collapse in={showFilters}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <TextField
            size="small"
            placeholder="Search signals..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search fontSize="small" />
                </InputAdornment>
              ),
            }}
            sx={{ minWidth: 200 }}
          />

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Filter</InputLabel>
            <Select
              value={filterType}
              label="Filter"
              onChange={(e) => setFilterType(e.target.value as FilterType)}
            >
              <MenuItem value="all">All Signals</MenuItem>
              <MenuItem value="BUY">Buy Signals</MenuItem>
              <MenuItem value="SELL">Sell Signals</MenuItem>
              <MenuItem value="HOLD">Hold Signals</MenuItem>
              <MenuItem value="live">Live Only</MenuItem>
              <MenuItem value="strong">Strong Only</MenuItem>
              <MenuItem value="moderate">Moderate Only</MenuItem>
              <MenuItem value="weak">Weak Only</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortField}
              label="Sort By"
              onChange={(e) => setSortField(e.target.value as SortField)}
            >
              <MenuItem value="confidence">Confidence</MenuItem>
              <MenuItem value="timestamp">Time</MenuItem>
              <MenuItem value="symbol">Symbol</MenuItem>
              <MenuItem value="performance">Performance</MenuItem>
              <MenuItem value="risk">Risk/Reward</MenuItem>
            </Select>
          </FormControl>

          <Button
            variant="outlined"
            size="small"
            onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
          >
            {sortDirection === 'asc' ? '↑' : '↓'}
          </Button>
        </Stack>
      </Box>
    </Collapse>
  );

  return (
    <Box className={className} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        p: 2,
        borderBottom: 1,
        borderColor: 'divider'
      }}>
        <Typography variant="h6">
          Signals ({processedSignals.length})
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {enableFiltering && (
            <IconButton
              size="small"
              onClick={() => setShowFilters(!showFilters)}
              color={showFilters ? 'primary' : 'default'}
            >
              <FilterList />
            </IconButton>
          )}

          {onRefresh && (
            <IconButton
              size="small"
              onClick={onRefresh}
              disabled={isRefreshing}
            >
              <Refresh sx={{
                animation: isRefreshing ? 'spin 1s linear infinite' : 'none'
              }} />
            </IconButton>
          )}
        </Box>
      </Box>

      {/* Filter Controls */}
      {enableFiltering && renderFilterControls()}

      {/* Loading State */}
      {signals.length === 0 && renderLoadingSkeleton()}

      {/* Signal List */}
      {signals.length > 0 && (
        <Box sx={{ flex: 1, overflow: 'hidden' }}>
          {enableVirtualization && processedSignals.length > 10 ? (
            <VirtualList
              ref={listRef}
              height={height}
              itemCount={processedSignals.length}
              itemSize={120}
              overscanCount={5}
            >
              {renderSignalItem}
            </VirtualList>
          ) : (
            <Box sx={{ height: '100%', overflow: 'auto' }}>
              {processedSignals.map((signal, index) =>
                <div key={signal.id}>
                  {renderSignalItem({ index, style: {} })}
                </div>
              )}
            </Box>
          )}
        </Box>
      )}

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={() => setMenuAnchor(null)}
      >
        <MenuItem onClick={() => setMenuAnchor(null)}>
          <Star sx={{ mr: 1 }} />
          Add to Watchlist
        </MenuItem>
        <MenuItem onClick={() => setMenuAnchor(null)}>
          <VisibilityOff sx={{ mr: 1 }} />
          Hide Signal
        </MenuItem>
        <MenuItem onClick={() => setMenuAnchor(null)}>
          <Analytics sx={{ mr: 1 }} />
          View Details
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default SignalList;
