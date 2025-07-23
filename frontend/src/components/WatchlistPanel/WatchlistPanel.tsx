import React, { useState } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Avatar,
  Divider,
  Stack,
  alpha,
  useTheme,
  TextField,
  InputAdornment,
  Button,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Styled components
const WatchlistItem = styled(ListItem)(({ theme }) => ({
  borderRadius: theme.spacing(0.5),
  marginBottom: theme.spacing(0.25),
  padding: theme.spacing(0.4, 0.75),
  backgroundColor: alpha(theme.palette.background.default, 0.3),
  minHeight: 36,
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.08),
    cursor: 'pointer',
  },
  '& .MuiListItemText-primary': {
    fontSize: '0.8rem',
    fontWeight: 500,
  },
  '& .MuiListItemText-secondary': {
    fontSize: '0.7rem',
  },
}));

const PriceChange = styled(Box)<{ positive: boolean }>(({ theme, positive }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.25),
  color: positive ? theme.palette.success.main : theme.palette.error.main,
  fontSize: '0.75rem',
}));

const MiniChart = styled(Box)(({ theme }) => ({
  width: 60,
  height: 20,
  backgroundColor: alpha(theme.palette.primary.main, 0.1),
  borderRadius: theme.spacing(0.5),
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

interface WatchlistItem {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  isFavorite: boolean;
  signalStrength: 'Strong' | 'Moderate' | 'Weak' | 'None';
}

interface WatchlistPanelProps {
  onSymbolSelect: (symbol: string) => void;
}

const WatchlistPanel: React.FC<WatchlistPanelProps> = ({ onSymbolSelect }) => {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.25,
      change: 3.15,
      changePercent: 2.14,
      volume: 52000000,
      isFavorite: true,
      signalStrength: 'Strong',
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corp.',
      price: 280.50,
      change: -2.25,
      changePercent: -0.80,
      volume: 31000000,
      isFavorite: true,
      signalStrength: 'Moderate',
    },
    {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      price: 2450.00,
      change: 29.50,
      changePercent: 1.22,
      volume: 18000000,
      isFavorite: false,
      signalStrength: 'Weak',
    },
    {
      symbol: 'TSLA',
      name: 'Tesla Inc.',
      price: 245.67,
      change: 12.33,
      changePercent: 5.28,
      volume: 87000000,
      isFavorite: true,
      signalStrength: 'Strong',
    },
    {
      symbol: 'NVDA',
      name: 'NVIDIA Corp.',
      price: 425.80,
      change: -8.45,
      changePercent: -1.95,
      volume: 45000000,
      isFavorite: false,
      signalStrength: 'Moderate',
    },
    {
      symbol: 'AMD',
      name: 'Advanced Micro Devices',
      price: 102.15,
      change: 1.87,
      changePercent: 1.86,
      volume: 23000000,
      isFavorite: false,
      signalStrength: 'None',
    },
  ]);

  const getSignalColor = (strength: string) => {
    switch (strength) {
      case 'Strong':
        return 'success';
      case 'Moderate':
        return 'warning';
      case 'Weak':
        return 'info';
      default:
        return 'default';
    }
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    }
    if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const toggleFavorite = (symbol: string) => {
    setWatchlist(prev =>
      prev.map(item =>
        item.symbol === symbol
          ? { ...item, isFavorite: !item.isFavorite }
          : item
      )
    );
  };

  const removeFromWatchlist = (symbol: string) => {
    setWatchlist(prev => prev.filter(item => item.symbol !== symbol));
  };

  const filteredWatchlist = watchlist.filter(item =>
    item.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const favoriteItems = filteredWatchlist.filter(item => item.isFavorite);
  const otherItems = filteredWatchlist.filter(item => !item.isFavorite);

  return (
    <Box height="100%" display="flex" flexDirection="column">
      {/* Search Bar */}
      <TextField
        fullWidth
        size="small"
        placeholder="Search symbols..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
        sx={{ mb: 2 }}
      />

      {/* Watchlist Items */}
      <Box flex={1} sx={{ overflowY: 'auto' }}>
        {/* Favorites */}
        {favoriteItems.length > 0 && (
          <>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Favorites
            </Typography>
            <List dense>
              {favoriteItems.map((item) => (
                <WatchlistItem
                  key={item.symbol}
                  onClick={() => onSymbolSelect(item.symbol)}
                >
                  <Box display="flex" alignItems="center" width="100%">
                    <Box flex={1}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {item.symbol}
                        </Typography>
                        <Chip
                          label={item.signalStrength}
                          size="small"
                          color={getSignalColor(item.signalStrength)}
                          sx={{ height: 18, fontSize: '0.65rem' }}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {item.name}
                      </Typography>
                      <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                        <Typography variant="body2" fontWeight="bold">
                          ${item.price.toFixed(2)}
                        </Typography>
                        <PriceChange positive={item.change >= 0}>
                          {item.change >= 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
                          <Typography variant="caption">
                            {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)} ({item.changePercent.toFixed(2)}%)
                          </Typography>
                        </PriceChange>
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        Vol: {formatVolume(item.volume)}
                      </Typography>
                    </Box>

                    <Box display="flex" flexDirection="column" alignItems="flex-end" gap={0.5}>
                      <MiniChart>
                        <Typography variant="caption" color="primary">
                          📈
                        </Typography>
                      </MiniChart>
                      <Box display="flex" gap={0.5}>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFavorite(item.symbol);
                          }}
                        >
                          {item.isFavorite ? <StarIcon color="primary" /> : <StarBorderIcon />}
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromWatchlist(item.symbol);
                          }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Box>
                    </Box>
                  </Box>
                </WatchlistItem>
              ))}
            </List>
            {otherItems.length > 0 && <Divider sx={{ my: 2 }} />}
          </>
        )}

        {/* Other Items */}
        {otherItems.length > 0 && (
          <>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Watchlist
            </Typography>
            <List dense>
              {otherItems.map((item) => (
                <WatchlistItem
                  key={item.symbol}
                  onClick={() => onSymbolSelect(item.symbol)}
                >
                  <Box display="flex" alignItems="center" width="100%">
                    <Box flex={1}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {item.symbol}
                        </Typography>
                        <Chip
                          label={item.signalStrength}
                          size="small"
                          color={getSignalColor(item.signalStrength)}
                          sx={{ height: 18, fontSize: '0.65rem' }}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {item.name}
                      </Typography>
                      <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                        <Typography variant="body2" fontWeight="bold">
                          ${item.price.toFixed(2)}
                        </Typography>
                        <PriceChange positive={item.change >= 0}>
                          {item.change >= 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
                          <Typography variant="caption">
                            {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)} ({item.changePercent.toFixed(2)}%)
                          </Typography>
                        </PriceChange>
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        Vol: {formatVolume(item.volume)}
                      </Typography>
                    </Box>

                    <Box display="flex" flexDirection="column" alignItems="flex-end" gap={0.5}>
                      <MiniChart>
                        <Typography variant="caption" color="primary">
                          📊
                        </Typography>
                      </MiniChart>
                      <Box display="flex" gap={0.5}>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFavorite(item.symbol);
                          }}
                        >
                          {item.isFavorite ? <StarIcon color="primary" /> : <StarBorderIcon />}
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromWatchlist(item.symbol);
                          }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Box>
                    </Box>
                  </Box>
                </WatchlistItem>
              ))}
            </List>
          </>
        )}
      </Box>

      {/* Add Symbol Button */}
      <Box mt={2}>
        <Button
          fullWidth
          variant="outlined"
          startIcon={<AddIcon />}
          size="small"
        >
          Add Symbol
        </Button>
      </Box>
    </Box>
  );
};

export default WatchlistPanel;
