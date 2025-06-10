/**
 * Portfolio Page
 * 
 * Portfolio tracking with performance metrics, positions, and P&L analysis
 */

// React imports removed as not needed
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  useTheme,
  alpha,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as GainIcon,
  TrendingDown as LossIcon,
  AccountBalance as PortfolioIcon,
  PieChart as AllocationIcon,
} from '@mui/icons-material';

// Mock portfolio data
const mockPortfolio = {
  totalValue: 125430.75,
  dailyPnL: 2847.32,
  dailyPnLPercent: 2.32,
  totalPnL: 25430.75,
  totalPnLPercent: 25.43,
  cash: 15000.00,
  totalInvested: 100000.00,
  positions: [
    {
      symbol: 'AAPL',
      shares: 150,
      avgPrice: 175.20,
      currentPrice: 185.45,
      marketValue: 27817.50,
      unrealizedPnL: 1537.50,
      unrealizedPnLPercent: 5.85,
      allocation: 22.2,
    },
    {
      symbol: 'TSLA',
      shares: 75,
      avgPrice: 245.80,
      currentPrice: 267.90,
      marketValue: 20092.50,
      unrealizedPnL: 1657.50,
      unrealizedPnLPercent: 8.99,
      allocation: 16.0,
    },
    {
      symbol: 'NVDA',
      shares: 45,
      avgPrice: 420.50,
      currentPrice: 445.25,
      marketValue: 20036.25,
      unrealizedPnL: 1113.75,
      unrealizedPnLPercent: 5.88,
      allocation: 16.0,
    },
    {
      symbol: 'GOOGL',
      shares: 80,
      avgPrice: 135.75,
      currentPrice: 142.30,
      marketValue: 11384.00,
      unrealizedPnL: 524.00,
      unrealizedPnLPercent: 4.83,
      allocation: 9.1,
    },
    {
      symbol: 'AMZN',
      shares: 120,
      avgPrice: 125.60,
      currentPrice: 131.85,
      marketValue: 15822.00,
      unrealizedPnL: 750.00,
      unrealizedPnLPercent: 4.97,
      allocation: 12.6,
    },
  ],
  recentTrades: [
    {
      symbol: 'AAPL',
      side: 'BUY',
      shares: 25,
      price: 182.45,
      timestamp: '2024-01-15 14:23:00',
      pnl: null,
    },
    {
      symbol: 'TSLA',
      side: 'SELL',
      shares: 15,
      price: 265.20,
      timestamp: '2024-01-15 11:45:00',
      pnl: 287.50,
    },
    {
      symbol: 'NVDA',
      side: 'BUY',
      shares: 10,
      price: 438.75,
      timestamp: '2024-01-15 09:32:00',
      pnl: null,
    },
  ],
};

// Portfolio Summary Cards
function PortfolioSummary() {
  const theme = useTheme();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <PortfolioIcon sx={{ fontSize: 40, color: theme.palette.primary.main, mb: 2 }} />
            <Typography variant="h4" fontWeight={700} color="primary.main">
              {formatCurrency(mockPortfolio.totalValue)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Portfolio Value
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            {mockPortfolio.dailyPnL >= 0 ? (
              <GainIcon sx={{ fontSize: 40, color: theme.palette.success.main, mb: 2 }} />
            ) : (
              <LossIcon sx={{ fontSize: 40, color: theme.palette.error.main, mb: 2 }} />
            )}
            <Typography 
              variant="h4" 
              fontWeight={700} 
              color={mockPortfolio.dailyPnL >= 0 ? 'success.main' : 'error.main'}
            >
              {formatCurrency(mockPortfolio.dailyPnL)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Today's P&L ({formatPercentage(mockPortfolio.dailyPnLPercent)})
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            {mockPortfolio.totalPnL >= 0 ? (
              <GainIcon sx={{ fontSize: 40, color: theme.palette.success.main, mb: 2 }} />
            ) : (
              <LossIcon sx={{ fontSize: 40, color: theme.palette.error.main, mb: 2 }} />
            )}
            <Typography 
              variant="h4" 
              fontWeight={700} 
              color={mockPortfolio.totalPnL >= 0 ? 'success.main' : 'error.main'}
            >
              {formatCurrency(mockPortfolio.totalPnL)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total P&L ({formatPercentage(mockPortfolio.totalPnLPercent)})
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent sx={{ textAlign: 'center' }}>
            <AllocationIcon sx={{ fontSize: 40, color: theme.palette.info.main, mb: 2 }} />
            <Typography variant="h4" fontWeight={700} color="info.main">
              {formatCurrency(mockPortfolio.cash)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Available Cash
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

// Positions Table
function PositionsTable() {
  const theme = useTheme();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Current Positions
        </Typography>
        <TableContainer component={Paper} sx={{ borderRadius: 2 }}>
          <Table>
            <TableHead>
              <TableRow sx={{ backgroundColor: alpha(theme.palette.primary.main, 0.1) }}>
                <TableCell>Symbol</TableCell>
                <TableCell align="right">Shares</TableCell>
                <TableCell align="right">Avg Price</TableCell>
                <TableCell align="right">Current Price</TableCell>
                <TableCell align="right">Market Value</TableCell>
                <TableCell align="right">Unrealized P&L</TableCell>
                <TableCell align="right">Allocation</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {mockPortfolio.positions.map((position) => (
                <TableRow key={position.symbol}>
                  <TableCell>
                    <Typography variant="subtitle2" fontWeight={600}>
                      {position.symbol}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">{position.shares}</TableCell>
                  <TableCell align="right">{formatCurrency(position.avgPrice)}</TableCell>
                  <TableCell align="right">{formatCurrency(position.currentPrice)}</TableCell>
                  <TableCell align="right">{formatCurrency(position.marketValue)}</TableCell>
                  <TableCell align="right">
                    <Box>
                      <Typography 
                        variant="body2" 
                        fontWeight={600}
                        color={position.unrealizedPnL >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatCurrency(position.unrealizedPnL)}
                      </Typography>
                      <Typography 
                        variant="caption" 
                        color={position.unrealizedPnL >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatPercentage(position.unrealizedPnLPercent)}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                      <Typography variant="body2">{position.allocation}%</Typography>
                      <Box sx={{ width: 60 }}>
                        <LinearProgress
                          variant="determinate"
                          value={position.allocation}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                          }}
                        />
                      </Box>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
}

// Recent Trades
function RecentTrades() {
  const theme = useTheme();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Recent Trades
        </Typography>
        <Stack spacing={2}>
          {mockPortfolio.recentTrades.map((trade, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 2,
                borderRadius: 2,
                backgroundColor: alpha(
                  trade.side === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                  0.1
                ),
                border: `1px solid ${alpha(
                  trade.side === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                  0.2
                )}`,
              }}
            >
              <Box>
                <Typography variant="subtitle1" fontWeight={600}>
                  {trade.side} {trade.shares} {trade.symbol}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  @ {formatCurrency(trade.price)} • {new Date(trade.timestamp).toLocaleString()}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Chip
                  label={trade.side}
                  size="small"
                  sx={{
                    backgroundColor: trade.side === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                    color: 'white',
                    fontWeight: 600,
                  }}
                />
                {trade.pnl && (
                  <Typography 
                    variant="body2" 
                    fontWeight={600}
                    color={trade.pnl >= 0 ? 'success.main' : 'error.main'}
                    sx={{ mt: 0.5 }}
                  >
                    P&L: {formatCurrency(trade.pnl)}
                  </Typography>
                )}
              </Box>
            </Box>
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
}

// Main Portfolio Page Component
export default function PortfolioPage() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Portfolio
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Track your trading performance and positions
        </Typography>
      </Box>

      {/* Portfolio Summary */}
      <Box sx={{ mb: 4 }}>
        <PortfolioSummary />
      </Box>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Positions Table */}
        <Grid item xs={12}>
          <PositionsTable />
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12} md={6}>
          <RecentTrades />
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Stack spacing={2}>
                <Button variant="contained" fullWidth size="large">
                  View Detailed Analytics
                </Button>
                <Button variant="outlined" fullWidth size="large">
                  Export Portfolio Report
                </Button>
                <Button variant="outlined" fullWidth size="large">
                  Rebalance Portfolio
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
} 