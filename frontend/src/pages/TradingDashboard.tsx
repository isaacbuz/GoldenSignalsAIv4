import React, { useState } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  useTheme,
  alpha,
} from '@mui/material';
import { styled } from '@mui/material/styles';

// Import the new components
import { TradeSearch } from '../components/TradeSearch';
import { CentralChart } from '../components/CentralChart/CentralChart';
import MarketContext from '../components/MarketContext/MarketContext';

// Styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
  minHeight: '100vh',
  backgroundColor: theme.palette.background.default,
  paddingTop: theme.spacing(3),
  paddingBottom: theme.spacing(3),
}));

const PageHeader = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  padding: theme.spacing(2),
  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)} 0%, ${alpha(
    theme.palette.primary.main,
    0.05
  )} 100%)`,
  borderRadius: theme.spacing(2),
  border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
}));

const TradingDashboard: React.FC = () => {
  const theme = useTheme();
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1d');

  const handleAnalyze = async (newSymbol: string, newTimeframe: string) => {
    setSymbol(newSymbol);
    setTimeframe(newTimeframe);
    console.log('Analyzing:', newSymbol, newTimeframe);
    // Here you would trigger the actual analysis
  };

  const handleNewsClick = (news: any) => {
    console.log('News clicked:', news);
    // Handle news item click
  };

  return (
    <DashboardContainer>
      <Container maxWidth="xl">
        {/* Page Header */}
        <PageHeader>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            GoldenSignalsAI Trading Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Professional-grade trading signals with {'>'}90% accuracy powered by 30+ AI agents
          </Typography>
        </PageHeader>

        {/* Search Component */}
        <Box mb={3}>
          <TradeSearch
            onSubmit={handleAnalyze}
            onSymbolChange={setSymbol}
            onTimeframeChange={setTimeframe}
            defaultSymbol={symbol}
            defaultTimeframe={timeframe}
          />
        </Box>

        {/* Main Content Grid */}
        <Grid container spacing={3}>
          {/* Chart Section - Takes up 2/3 of the width */}
          <Grid item xs={12} lg={8}>
            <Paper elevation={0} sx={{ height: 600, p: 0 }}>
              <CentralChart
                symbol={symbol}
                timeframe={timeframe}
                onSymbolChange={setSymbol}
                onTimeframeChange={setTimeframe}
              />
            </Paper>
          </Grid>

          {/* Market Context Section - Takes up 1/3 of the width */}
          <Grid item xs={12} lg={4}>
            <Paper elevation={0} sx={{ height: 600, p: 2, overflow: 'hidden' }}>
              <MarketContext
                symbol={symbol}
                onNewsClick={handleNewsClick}
              />
            </Paper>
          </Grid>
        </Grid>

        {/* Additional Features Notice */}
        <Box mt={3} p={3} sx={{
          backgroundColor: alpha(theme.palette.info.main, 0.1),
          borderRadius: 2,
          border: `1px solid ${alpha(theme.palette.info.main, 0.3)}`,
        }}>
          <Typography variant="h6" gutterBottom>
            Additional Features in Development
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • AI Agent Consensus Matrix for {'>'}90% accuracy signals<br />
            • Real-time signal generation with entry/exit overlays<br />
            • Risk management dashboard with position sizing<br />
            • Backtesting interface for strategy validation<br />
            • Multi-timeframe analysis and correlation tracking
          </Typography>
        </Box>
      </Container>
    </DashboardContainer>
  );
};

export default TradingDashboard;