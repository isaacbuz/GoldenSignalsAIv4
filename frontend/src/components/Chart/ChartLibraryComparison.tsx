/**
 * ChartLibraryComparison - Side-by-side comparison of Highcharts Stock vs LightningChart JS
 *
 * This component allows easy comparison of both charting libraries to help
 * decide which one to use for the production implementation.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Check as CheckIcon,
  Close as CloseIcon,
  Speed as SpeedIcon,
  Brush as FeaturesIcon,
  AttachMoney as PriceIcon,
  Code as APIIcon,
  TrendingUp as PerformanceIcon,
  Support as SupportIcon,
} from '@mui/icons-material';

// Import both chart implementations
import ProfessionalHighchartsChart from './ProfessionalHighchartsChart';
import ProfessionalLightningChart from './ProfessionalLightningChart';

// Styled Components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: '#0e0f14',
  display: 'flex',
  flexDirection: 'column',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
});

const Header = styled(Box)({
  backgroundColor: '#16181d',
  borderBottom: '1px solid #2a2d37',
  padding: '20px',
});

const ComparisonGrid = styled(Grid)({
  flex: 1,
  overflow: 'hidden',
});

const ChartSection = styled(Box)({
  height: '70%',
  position: 'relative',
  borderBottom: '1px solid #2a2d37',
});

const ComparisonSection = styled(Paper)({
  height: '30%',
  backgroundColor: '#16181d',
  overflow: 'auto',
  padding: '20px',
});

const FeatureRow = styled(ListItem)({
  borderBottom: '1px solid #2a2d37',
  '&:last-child': {
    borderBottom: 'none',
  },
});

const ScoreChip = styled(Chip)<{ score: number }>(({ score }) => ({
  backgroundColor: score >= 9 ? '#00c805' :
                 score >= 7 ? '#ff9800' :
                 score >= 5 ? '#ff6d00' : '#ff3b30',
  color: '#fff',
  fontWeight: 600,
}));

// Comparison data
const LIBRARY_FEATURES = {
  highcharts: {
    name: 'Highcharts Stock',
    pros: [
      'Mature library with extensive documentation',
      'Built-in technical indicators (20+)',
      'Stock-specific features (navigator, range selector)',
      'Drawing tools and annotations',
      'Excellent browser compatibility',
      'Large community and ecosystem',
      'Export to PNG/SVG/PDF',
    ],
    cons: [
      'Commercial license required ($995/year)',
      'Larger bundle size (~500KB)',
      'Can be slower with massive datasets',
      'Configuration can be complex',
    ],
    scores: {
      performance: 8,
      features: 10,
      ease: 8,
      price: 5,
      support: 9,
      overall: 8,
    },
  },
  lightning: {
    name: 'LightningChart JS Trader',
    pros: [
      'Exceptional performance (WebGL-based)',
      'Handles millions of data points smoothly',
      'Real-time optimized',
      'Modern API design',
      'GPU acceleration',
      'Minimal CPU usage',
      'Professional trading features',
    ],
    cons: [
      'Commercial license required ($1,395/year)',
      'Steeper learning curve',
      'Smaller community',
      'Less built-in indicators',
      'Newer library (less proven)',
    ],
    scores: {
      performance: 10,
      features: 8,
      ease: 6,
      price: 4,
      support: 7,
      overall: 8,
    },
  },
};

interface ChartLibraryComparisonProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const ChartLibraryComparison: React.FC<ChartLibraryComparisonProps> = ({
  symbol = 'TSLA',
  onSymbolChange
}) => {
  const [activeChart, setActiveChart] = useState<'both' | 'highcharts' | 'lightning'>('both');

  const handleChartChange = (
    event: React.MouseEvent<HTMLElement>,
    newChart: 'both' | 'highcharts' | 'lightning' | null,
  ) => {
    if (newChart !== null) {
      setActiveChart(newChart);
    }
  };

  const renderComparison = (library: 'highcharts' | 'lightning') => {
    const data = LIBRARY_FEATURES[library];

    return (
      <ComparisonSection elevation={0}>
        <Typography variant="h6" sx={{ color: '#e0e1e6', mb: 2, fontWeight: 600 }}>
          {data.name}
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="subtitle2" sx={{ color: '#00c805', mb: 1 }}>
              Pros
            </Typography>
            <List dense>
              {data.pros.map((pro, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemIcon sx={{ minWidth: 30 }}>
                    <CheckIcon sx={{ fontSize: 16, color: '#00c805' }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={pro}
                    primaryTypographyProps={{
                      fontSize: 13,
                      color: '#e0e1e6'
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </Grid>

          <Grid item xs={6}>
            <Typography variant="subtitle2" sx={{ color: '#ff3b30', mb: 1 }}>
              Cons
            </Typography>
            <List dense>
              {data.cons.map((con, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemIcon sx={{ minWidth: 30 }}>
                    <CloseIcon sx={{ fontSize: 16, color: '#ff3b30' }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={con}
                    primaryTypographyProps={{
                      fontSize: 13,
                      color: '#e0e1e6'
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </Grid>
        </Grid>

        <Divider sx={{ my: 2, borderColor: '#2a2d37' }} />

        <Typography variant="subtitle2" sx={{ color: '#8c8e96', mb: 1 }}>
          Scores
        </Typography>
        <Grid container spacing={1}>
          <Grid item xs={4}>
            <Box display="flex" alignItems="center" gap={1}>
              <PerformanceIcon sx={{ fontSize: 18, color: '#8c8e96' }} />
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>Performance:</Typography>
              <ScoreChip label={data.scores.performance} size="small" score={data.scores.performance} />
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box display="flex" alignItems="center" gap={1}>
              <FeaturesIcon sx={{ fontSize: 18, color: '#8c8e96' }} />
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>Features:</Typography>
              <ScoreChip label={data.scores.features} size="small" score={data.scores.features} />
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box display="flex" alignItems="center" gap={1}>
              <APIIcon sx={{ fontSize: 18, color: '#8c8e96' }} />
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>Ease of Use:</Typography>
              <ScoreChip label={data.scores.ease} size="small" score={data.scores.ease} />
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box display="flex" alignItems="center" gap={1}>
              <PriceIcon sx={{ fontSize: 18, color: '#8c8e96' }} />
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>Price:</Typography>
              <ScoreChip label={data.scores.price} size="small" score={data.scores.price} />
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box display="flex" alignItems="center" gap={1}>
              <SupportIcon sx={{ fontSize: 18, color: '#8c8e96' }} />
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>Support:</Typography>
              <ScoreChip label={data.scores.support} size="small" score={data.scores.support} />
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box display="flex" alignItems="center" gap={1}>
              <TrendingUp sx={{ fontSize: 18, color: '#FFD700' }} />
              <Typography variant="caption" sx={{ color: '#FFD700', fontWeight: 600 }}>Overall:</Typography>
              <ScoreChip label={data.scores.overall} size="small" score={data.scores.overall} />
            </Box>
          </Grid>
        </Grid>
      </ComparisonSection>
    );
  };

  return (
    <Container>
      <Header>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h5" sx={{ color: '#e0e1e6', fontWeight: 600 }}>
            Chart Library Comparison
          </Typography>

          <ToggleButtonGroup
            value={activeChart}
            exclusive
            onChange={handleChartChange}
            size="small"
            sx={{
              '& .MuiToggleButton-root': {
                color: '#8c8e96',
                borderColor: '#2a2d37',
                '&.Mui-selected': {
                  color: '#fff',
                  backgroundColor: '#2196f3',
                },
              },
            }}
          >
            <ToggleButton value="highcharts">
              Highcharts Only
            </ToggleButton>
            <ToggleButton value="both">
              Side by Side
            </ToggleButton>
            <ToggleButton value="lightning">
              LightningChart Only
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Header>

      <ComparisonGrid container>
        {(activeChart === 'highcharts' || activeChart === 'both') && (
          <Grid item xs={activeChart === 'both' ? 6 : 12}>
            <ChartSection>
              <ProfessionalHighchartsChart
                symbol={symbol}
                onSymbolChange={onSymbolChange}
              />
            </ChartSection>
            {renderComparison('highcharts')}
          </Grid>
        )}

        {activeChart === 'both' && (
          <Divider orientation="vertical" sx={{ borderColor: '#2a2d37' }} />
        )}

        {(activeChart === 'lightning' || activeChart === 'both') && (
          <Grid item xs={activeChart === 'both' ? 6 : 12}>
            <ChartSection>
              <ProfessionalLightningChart
                symbol={symbol}
                onSymbolChange={onSymbolChange}
              />
            </ChartSection>
            {renderComparison('lightning')}
          </Grid>
        )}
      </ComparisonGrid>
    </Container>
  );
};

export default ChartLibraryComparison;
