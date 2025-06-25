import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Fade,
  TransitionGroup,
  Button,
  Menu,
  MenuItem,
  Divider,
} from '@mui/material';
import { Refresh, FilterList } from '@mui/icons-material';
import SignalCard, { SignalData } from './SignalCard';
import { useSignalUpdates } from '../../services/websocket/SignalWebSocketManager';

interface SignalListProps {
  signals?: SignalData[];
  loading?: boolean;
  variant?: 'compact' | 'detailed';
  maxItems?: number;
  onSignalClick?: (signal: SignalData) => void;
  onRefresh?: () => void;
  showFilters?: boolean;
  autoUpdate?: boolean;
}

const SignalList: React.FC<SignalListProps> = ({
  signals: initialSignals = [],
  loading = false,
  variant = 'compact',
  maxItems,
  onSignalClick,
  onRefresh,
  showFilters = true,
  autoUpdate = true,
}) => {
  const [signals, setSignals] = useState<SignalData[]>(initialSignals);
  const [filterMenuAnchor, setFilterMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedSignal, setSelectedSignal] = useState<SignalData | null>(null);
  const [actionMenuAnchor, setActionMenuAnchor] = useState<null | HTMLElement>(null);

  // WebSocket connection for real-time updates
  const { isConnected } = useSignalUpdates((newSignal) => {
    if (autoUpdate) {
      setSignals(prev => [newSignal, ...prev].slice(0, maxItems || 50));
    }
  }, 'signal-list');

  useEffect(() => {
    setSignals(initialSignals);
  }, [initialSignals]);

  const handleSignalAction = (action: string, signal: SignalData) => {
    if (action === 'menu') {
      setSelectedSignal(signal);
      setActionMenuAnchor(document.getElementById(`signal-${signal.id}`) as HTMLElement);
    }
  };

  const displaySignals = maxItems ? signals.slice(0, maxItems) : signals;

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress sx={{ color: '#FFD700' }} />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header Controls */}
      {(showFilters || onRefresh) && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {isConnected && (
              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', color: '#4CAF50' }}>
                • Live
              </Typography>
            )}
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {showFilters && (
              <Button
                size="small"
                startIcon={<FilterList />}
                onClick={(e) => setFilterMenuAnchor(e.currentTarget)}
              >
                Filter
              </Button>
            )}
            {onRefresh && (
              <Button size="small" startIcon={<Refresh />} onClick={onRefresh}>
                Refresh
              </Button>
            )}
          </Box>
        </Box>
      )}

      {/* Signal List */}
      <TransitionGroup>
        {displaySignals.map((signal) => (
          <Fade key={signal.id} timeout={300}>
            <Box id={`signal-${signal.id}`} sx={{ mb: 2 }}>
              <SignalCard
                signal={signal}
                variant={variant}
                highlight={displaySignals.indexOf(signal) === 0}
                onClick={onSignalClick}
                onAction={handleSignalAction}
              />
            </Box>
          </Fade>
        ))}
      </TransitionGroup>

      {/* Empty State */}
      {displaySignals.length === 0 && !loading && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body2" color="text.secondary">
            No signals available
          </Typography>
        </Box>
      )}

      {/* Filter Menu */}
      <Menu
        anchorEl={filterMenuAnchor}
        open={Boolean(filterMenuAnchor)}
        onClose={() => setFilterMenuAnchor(null)}
      >
        <MenuItem>Signal Type</MenuItem>
        <MenuItem>Confidence Range</MenuItem>
        <MenuItem>Time Period</MenuItem>
        <MenuItem>Agents</MenuItem>
      </Menu>

      {/* Action Menu */}
      <Menu
        anchorEl={actionMenuAnchor}
        open={Boolean(actionMenuAnchor)}
        onClose={() => setActionMenuAnchor(null)}
      >
        <MenuItem>View Details</MenuItem>
        <MenuItem>Copy Signal ID</MenuItem>
        <MenuItem>Set Alert</MenuItem>
        <Divider />
        <MenuItem>View Similar Signals</MenuItem>
      </Menu>
    </Box>
  );
};

export default SignalList;
