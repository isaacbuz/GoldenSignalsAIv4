import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  IconButton,
  ButtonGroup,
  Button,
  Typography,
  Paper,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Menu,
  MenuItem,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  GridView as GridViewIcon,
  ViewColumn as ViewColumnIcon,
  Compare as CompareIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Settings as SettingsIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import ProfessionalChart from './ProfessionalChart';

const MultiChartContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
}));

const ChartHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  backgroundColor: alpha(theme.palette.background.paper, 0.5),
}));

const ChartGrid = styled(Grid)(({ theme }) => ({
  flex: 1,
  overflow: 'hidden',
  '& .chart-panel': {
    height: '100%',
    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
    borderRadius: theme.spacing(0.5),
    overflow: 'hidden',
    position: 'relative',
  },
}));

const TimeframeChip = styled(Chip)(({ theme }) => ({
  margin: theme.spacing(0, 0.5),
  height: 24,
  fontSize: '0.75rem',
  fontWeight: 600,
}));

export interface MultiTimeframeChartProps {
  symbol: string;
  onSymbolChange?: (symbol: string) => void;
  height?: string | number;
}

type LayoutMode = 'single' | 'dual-horizontal' | 'dual-vertical' | 'quad';
type ComparisonMode = 'none' | 'overlay' | 'percentage';

interface ChartConfig {
  id: string;
  symbol: string;
  timeframe: string;
  indicators: string[];
}

const defaultTimeframes = ['5m', '15m', '1h', '4h', '1d'];

export const MultiTimeframeChart: React.FC<MultiTimeframeChartProps> = ({
  symbol,
  onSymbolChange,
  height = '100%',
}) => {
  const theme = useTheme();
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('single');
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('none');
  const [fullscreen, setFullscreen] = useState(false);
  const [selectedTimeframes, setSelectedTimeframes] = useState<string[]>(['5m']);
  const [comparisonSymbols, setComparisonSymbols] = useState<string[]>([]);
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);
  const [syncCrosshair, setSyncCrosshair] = useState(true);

  // Chart configurations based on layout
  const [chartConfigs, setChartConfigs] = useState<ChartConfig[]>([
    {
      id: 'chart-1',
      symbol,
      timeframe: '5m',
      indicators: ['prediction', 'signals', 'volume'],
    },
  ]);

  // Update chart configurations when layout changes
  useEffect(() => {
    const newConfigs: ChartConfig[] = [];

    switch (layoutMode) {
      case 'single':
        newConfigs.push({
          id: 'chart-1',
          symbol,
          timeframe: selectedTimeframes[0] || '5m',
          indicators: ['prediction', 'signals', 'volume'],
        });
        break;

      case 'dual-horizontal':
      case 'dual-vertical':
        newConfigs.push(
          {
            id: 'chart-1',
            symbol,
            timeframe: selectedTimeframes[0] || '5m',
            indicators: ['prediction', 'signals', 'volume'],
          },
          {
            id: 'chart-2',
            symbol,
            timeframe: selectedTimeframes[1] || '1h',
            indicators: ['prediction', 'signals', 'volume'],
          }
        );
        break;

      case 'quad':
        const timeframes = ['5m', '15m', '1h', '1d'];
        timeframes.forEach((tf, index) => {
          newConfigs.push({
            id: `chart-${index + 1}`,
            symbol,
            timeframe: selectedTimeframes[index] || tf,
            indicators: ['prediction', 'signals', 'volume'],
          });
        });
        break;
    }

    setChartConfigs(newConfigs);
  }, [layoutMode, symbol, selectedTimeframes]);

  const handleLayoutChange = (event: React.MouseEvent<HTMLElement>, newLayout: LayoutMode | null) => {
    if (newLayout !== null) {
      setLayoutMode(newLayout);
    }
  };

  const handleTimeframeChange = (chartId: string, newTimeframe: string) => {
    setChartConfigs(configs =>
      configs.map(config =>
        config.id === chartId ? { ...config, timeframe: newTimeframe } : config
      )
    );

    // Update selected timeframes
    const index = chartConfigs.findIndex(c => c.id === chartId);
    if (index !== -1) {
      const newTimeframes = [...selectedTimeframes];
      newTimeframes[index] = newTimeframe;
      setSelectedTimeframes(newTimeframes);
    }
  };

  const handleAddComparison = () => {
    // In a real implementation, this would open a symbol search dialog
    const newSymbol = prompt('Enter symbol to compare:');
    if (newSymbol) {
      setComparisonSymbols([...comparisonSymbols, newSymbol.toUpperCase()]);
      setComparisonMode('overlay');
    }
  };

  const handleRemoveComparison = (symbolToRemove: string) => {
    setComparisonSymbols(comparisonSymbols.filter(s => s !== symbolToRemove));
    if (comparisonSymbols.length === 1) {
      setComparisonMode('none');
    }
  };

  const renderChartGrid = () => {
    switch (layoutMode) {
      case 'single':
        return (
          <Grid container spacing={1} sx={{ height: '100%' }}>
            <Grid item xs={12} sx={{ height: '100%' }}>
              <Box className="chart-panel">
                <ProfessionalChart
                  symbol={chartConfigs[0].symbol}
                  timeframe={chartConfigs[0].timeframe}
                  initialIndicators={chartConfigs[0].indicators}
                  onTimeframeChange={(tf) => handleTimeframeChange(chartConfigs[0].id, tf)}
                  showWatermark={true}
                />
              </Box>
            </Grid>
          </Grid>
        );

      case 'dual-horizontal':
        return (
          <Grid container spacing={1} sx={{ height: '100%' }}>
            {chartConfigs.map((config, index) => (
              <Grid key={config.id} item xs={12} md={6} sx={{ height: '100%' }}>
                <Box className="chart-panel">
                  <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}>
                    <TimeframeChip
                      label={config.timeframe.toUpperCase()}
                      color="primary"
                      size="small"
                    />
                  </Box>
                  <ProfessionalChart
                    symbol={config.symbol}
                    timeframe={config.timeframe}
                    initialIndicators={config.indicators}
                    onTimeframeChange={(tf) => handleTimeframeChange(config.id, tf)}
                    showWatermark={true}
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        );

      case 'dual-vertical':
        return (
          <Grid container spacing={1} sx={{ height: '100%' }}>
            {chartConfigs.map((config, index) => (
              <Grid key={config.id} item xs={12} sx={{ height: '50%' }}>
                <Box className="chart-panel">
                  <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}>
                    <TimeframeChip
                      label={config.timeframe.toUpperCase()}
                      color="primary"
                      size="small"
                    />
                  </Box>
                  <ProfessionalChart
                    symbol={config.symbol}
                    timeframe={config.timeframe}
                    initialIndicators={config.indicators}
                    onTimeframeChange={(tf) => handleTimeframeChange(config.id, tf)}
                    showWatermark={true}
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        );

      case 'quad':
        return (
          <Grid container spacing={1} sx={{ height: '100%' }}>
            {chartConfigs.map((config, index) => (
              <Grid key={config.id} item xs={12} sm={6} sx={{ height: '50%' }}>
                <Box className="chart-panel">
                  <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}>
                    <TimeframeChip
                      label={config.timeframe.toUpperCase()}
                      color="primary"
                      size="small"
                    />
                  </Box>
                  <ProfessionalChart
                    symbol={config.symbol}
                    timeframe={config.timeframe}
                    initialIndicators={config.indicators}
                    onTimeframeChange={(tf) => handleTimeframeChange(config.id, tf)}
                    showWatermark={true}
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        );
    }
  };

  return (
    <MultiChartContainer sx={{ height }}>
      <ChartHeader>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h6" fontWeight="bold">
            Multi-Timeframe Analysis
          </Typography>

          <ToggleButtonGroup
            value={layoutMode}
            exclusive
            onChange={handleLayoutChange}
            size="small"
          >
            <ToggleButton value="single" aria-label="single view">
              <ViewColumnIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="dual-horizontal" aria-label="dual horizontal">
              <GridViewIcon fontSize="small" sx={{ transform: 'rotate(90deg)' }} />
            </ToggleButton>
            <ToggleButton value="dual-vertical" aria-label="dual vertical">
              <GridViewIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="quad" aria-label="quad view">
              <GridViewIcon fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>

          <Divider orientation="vertical" flexItem />

          {comparisonMode === 'none' ? (
            <Button
              startIcon={<CompareIcon />}
              size="small"
              onClick={handleAddComparison}
              variant="outlined"
            >
              Compare
            </Button>
          ) : (
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="body2" color="text.secondary">
                Comparing:
              </Typography>
              {comparisonSymbols.map(s => (
                <Chip
                  key={s}
                  label={s}
                  size="small"
                  onDelete={() => handleRemoveComparison(s)}
                />
              ))}
              <IconButton size="small" onClick={handleAddComparison}>
                <AddIcon fontSize="small" />
              </IconButton>
            </Box>
          )}
        </Box>

        <Box display="flex" alignItems="center" gap={1}>
          <IconButton
            size="small"
            onClick={() => setSettingsAnchor(event.currentTarget)}
          >
            <SettingsIcon fontSize="small" />
          </IconButton>

          <IconButton
            size="small"
            onClick={() => setFullscreen(!fullscreen)}
          >
            {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </IconButton>
        </Box>
      </ChartHeader>

      <ChartGrid container sx={{ flex: 1, p: 1 }}>
        {renderChartGrid()}
      </ChartGrid>

      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
      >
        <MenuItem onClick={() => setSyncCrosshair(!syncCrosshair)}>
          <Chip
            label={syncCrosshair ? 'ON' : 'OFF'}
            size="small"
            color={syncCrosshair ? 'success' : 'default'}
            sx={{ mr: 1 }}
          />
          Sync Crosshair
        </MenuItem>
        <MenuItem disabled>
          <Chip label="Soon" size="small" sx={{ mr: 1 }} />
          Sync Indicators
        </MenuItem>
        <MenuItem disabled>
          <Chip label="Soon" size="small" sx={{ mr: 1 }} />
          Sync Drawing Tools
        </MenuItem>
      </Menu>
    </MultiChartContainer>
  );
};

export default MultiTimeframeChart;
