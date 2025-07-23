import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  useTheme,
  alpha,
  Container,
  IconButton,
  Chip,
  Stack,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Analytics as AnalyticsIcon,
  Groups as GroupsIcon,
} from '@mui/icons-material';

// Import existing components
import { TradeSearch } from '../components/TradeSearch';
import AITradingChart from '../components/AIChart/AITradingChart';

// New components we'll create
import SignalAnalysisPanel from '../components/SignalAnalysisPanel/SignalAnalysisPanel';
import WatchlistPanel from '../components/WatchlistPanel/WatchlistPanel';
import SignalSuggestions from '../components/SignalSuggestions/SignalSuggestions';
import AgentConsensusVisualizer from '../components/AgentConsensusVisualizer/AgentConsensusVisualizer';

// Import custom styles
import { customStyles } from '../theme/enhancedTheme';
import logger from '../services/logger';


// Styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
  minHeight: '100vh',
  backgroundColor: theme.palette.background.default,
  padding: theme.spacing(2, 3), // Increased left/right margins
  [theme.breakpoints.down('lg')]: {
    padding: theme.spacing(1.5, 2),
  },
}));

const CompactHeader = styled(Paper)(({ theme }) => ({
  ...customStyles.card,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: theme.spacing(1),
  padding: theme.spacing(0.75, 2),
  gap: theme.spacing(1),
  [theme.breakpoints.down('md')]: {
    padding: theme.spacing(0.5, 1.5),
  },
}));

const ContentWrapper = styled(Box)({
  maxWidth: '1800px', // Slightly reduced for better margins
  margin: '0 auto',
  width: '100%',
});

const MainContentArea = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(2),
  height: 'calc(100vh - 100px)', // Adjusted for header height and margins
  overflow: 'hidden',
  [theme.breakpoints.down('xl')]: {
    gap: theme.spacing(1.5),
  },
  [theme.breakpoints.down('lg')]: {
    flexDirection: 'column',
    height: 'auto',
    overflow: 'visible',
  },
}));

const ChartSection = styled(Box)(({ theme }) => ({
  flex: '1 1 auto',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  minWidth: 0,
  height: '100%',
  overflow: 'hidden',
  [theme.breakpoints.down('lg')]: {
    minHeight: '600px',
    height: 'auto',
  },
}));

const SignalAnalysisContainer = styled(Paper)(({ theme }) => ({
  ...customStyles.card,
  height: '220px', // Increased height to show all details
  overflow: 'auto', // Changed from hidden to auto for scrolling if needed
  padding: theme.spacing(1.5),
  '& .MuiTypography-root': {
    fontSize: '0.8rem',
  },
  '& .MuiTypography-h6': {
    fontSize: '0.875rem',
  },
  '& .MuiTypography-body2': {
    fontSize: '0.75rem',
  },
  '& .MuiTypography-caption': {
    fontSize: '0.7rem',
  },
  [theme.breakpoints.down('md')]: {
    height: '200px',
  },
}));



const Sidebar = styled(Box)(({ theme }) => ({
  width: 300, // Slightly increased for better content display
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  flexShrink: 0,
  height: '100%',
  overflow: 'auto',
  paddingBottom: theme.spacing(2),
  '&::-webkit-scrollbar': {
    width: '6px',
  },
  '&::-webkit-scrollbar-track': {
    background: 'transparent',
  },
  '&::-webkit-scrollbar-thumb': {
    background: alpha(theme.palette.divider, 0.2),
    borderRadius: '3px',
  },
  [theme.breakpoints.down('lg')]: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    overflow: 'visible',
  },
  [theme.breakpoints.down('md')]: {
    flexDirection: 'column',
  },
}));

const SidebarPanel = styled(Paper)(({ theme }) => ({
  ...customStyles.card,
  flex: 1,
  overflow: 'hidden',
  padding: theme.spacing(1.25),
  '& .MuiTypography-root': {
    fontSize: '0.8rem',
  },
  '& .MuiTypography-h6': {
    fontSize: '0.875rem',
  },
  '& .MuiTypography-subtitle1': {
    fontSize: '0.85rem',
  },
  '& .MuiTypography-body2': {
    fontSize: '0.75rem',
  },
  '& .MuiTypography-caption': {
    fontSize: '0.7rem',
  },
  [theme.breakpoints.down('lg')]: {
    flex: '1 1 calc(50% - 8px)',
    minHeight: 220,
  },
  [theme.breakpoints.down('md')]: {
    flex: '1 1 100%',
    minHeight: 180,
  },
}));

// Mock data for signals
const mockSignal = {
  symbol: 'AAPL',
  action: 'BUY',
  price: 150.25,
  confidence: 0.94,
  target: 165.00,
  stopLoss: 145.00,
  timestamp: new Date(),
  reasoning: [
    'Technical: RSI oversold bounce pattern detected',
    'Sentiment: Positive social media mentions increased 45%',
    'Volume: Above average by 150%, indicating strong interest',
    'Momentum: Breaking above key resistance level',
  ],
  agents: {
    total: 30,
    agreeing: 27,
    disagreeing: 3,
  },
};

const EnhancedTradingDashboard: React.FC = () => {
  const theme = useTheme();
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1d');
  const [currentSignal, setCurrentSignal] = useState(mockSignal);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);

  const handleAnalyze = async (newSymbol: string, newTimeframe: string) => {
    setIsAnalyzing(true);
    setSymbol(newSymbol);
    setTimeframe(newTimeframe);

    try {
      // Fetch live signal from backend
      const response = await fetch(`http://localhost:8000/api/v1/signals/generate/${newSymbol}`, {
        method: 'POST',
      });

      if (response.ok) {
        const signalData = await response.json();

        // Convert backend signal to UI format
        setCurrentSignal({
          symbol: signalData.symbol,
          action: signalData.action,
          price: signalData.price,
          confidence: signalData.confidence,
          target: signalData.price * 1.05, // 5% target
          stopLoss: signalData.price * 0.95, // 5% stop loss
          timestamp: new Date(signalData.timestamp),
          reasoning: signalData.reasoning.split('. ').filter(r => r.length > 0),
          agents: {
            total: 30,
            agreeing: Math.round(signalData.consensus_strength * 30),
            disagreeing: Math.round((1 - signalData.consensus_strength) * 30),
          },
        });
      }
    } catch (error) {
      logger.error('Failed to fetch signal:', error);
      // Keep the mock signal as fallback
      setCurrentSignal({
        ...mockSignal,
        symbol: newSymbol,
        timestamp: new Date(),
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Fetch chart data when symbol or timeframe changes
  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/market-data/${symbol}/history?period=30d&interval=1d`);
        if (response.ok) {
          const data = await response.json();
          setChartData(data.data || []);
        }
      } catch (error) {
        logger.error('Failed to fetch chart data:', error);
      }
    };

    fetchChartData();
  }, [symbol, timeframe]);

  return (
    <DashboardContainer>
      <ContentWrapper>
        {/* Compact Header */}
        <CompactHeader>
          <Box display="flex" alignItems="center" gap={1.5}>
            <AnalyticsIcon color="primary" sx={{ fontSize: 20 }} />
            <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
              GoldenSignalsAI
            </Typography>
            <Chip
              label="Pro"
              size="small"
              color="primary"
              sx={{ fontWeight: 'bold', height: 18, '& .MuiChip-label': { fontSize: '0.7rem', px: 0.75 } }}
            />
          </Box>

          <Box flex={1} />

          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              icon={<TrendingUpIcon sx={{ fontSize: '0.875rem' }} />}
              label="Live"
              color="success"
              size="small"
              sx={{ height: 20, '& .MuiChip-label': { fontSize: '0.7rem', px: 0.5 } }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
              Connected
            </Typography>
            <IconButton size="small">
              <GroupsIcon />
            </IconButton>
          </Box>
        </CompactHeader>

        {/* Main Content */}
        <MainContentArea>
          {/* Left: AI-Powered Chart - Center of Excellence */}
          <ChartSection>
            <Box sx={{ flex: 1, height: 'calc(100% - 240px)', minHeight: '400px', position: 'relative' }}> {/* Reduced chart height */}
              <AITradingChart
                height="100%"
                symbol={symbol}
                onSymbolAnalyze={(newSymbol, analysis) => {
                  setSymbol(newSymbol);
                  handleAnalyze(newSymbol, timeframe);
                  logger.info('AI Analysis:', analysis);
                }}
              />
            </Box>

            {/* Signal Analysis below chart */}
            <SignalAnalysisContainer elevation={0}>
              <SignalAnalysisPanel
                signal={currentSignal}
                isAnalyzing={isAnalyzing}
              />
            </SignalAnalysisContainer>
          </ChartSection>

          {/* Right Sidebar - Reduced width */}
          <Sidebar>
            {/* Agent Consensus - Now at the top, positioned under search */}
            <SidebarPanel elevation={0} sx={{ minHeight: 120, maxHeight: 150 }}>
              <Typography variant="subtitle1" sx={{ ...customStyles.sectionHeader, fontSize: '0.85rem', mb: 0.5 }}>
                Agent Consensus
              </Typography>
              <AgentConsensusVisualizer
                agents={currentSignal.agents}
                confidence={currentSignal.confidence}
              />
            </SidebarPanel>

            {/* Watchlist */}
            <SidebarPanel elevation={0} sx={{ flex: 1 }}>
              <Typography variant="subtitle1" sx={{ ...customStyles.sectionHeader, fontSize: '0.85rem', mb: 0.75 }}>
                Watchlist
              </Typography>
              <WatchlistPanel onSymbolSelect={setSymbol} />
            </SidebarPanel>

            {/* Signal Opportunities */}
            <SidebarPanel elevation={0} sx={{ flex: 1 }}>
              <Typography variant="subtitle1" sx={{ ...customStyles.sectionHeader, fontSize: '0.85rem', mb: 0.75 }}>
                Signal Opportunities
              </Typography>
              <SignalSuggestions onSignalSelect={handleAnalyze} />
            </SidebarPanel>
          </Sidebar>
        </MainContentArea>
      </ContentWrapper>
    </DashboardContainer>
  );
};

export default EnhancedTradingDashboard;
