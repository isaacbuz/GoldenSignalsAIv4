import React, { useState } from 'react';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Switch,
  Typography,
  Box,
  alpha,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Timeline as TimelineIcon,
  ShowChart as ShowChartIcon,
  BarChart as BarChartIcon,
  Analytics as AnalyticsIcon,
  Speed as SpeedIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  CandlestickChart as CandlestickChartIcon,
  GridOn as GridOnIcon,
  WaterDrop as WaterDropIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface ChartSettingsMenuProps {
  indicators: string[];
  onIndicatorToggle: (indicator: string) => void;
  chartType?: 'line' | 'candle';
  onChartTypeChange?: (type: 'line' | 'candle') => void;
  showGrid?: boolean;
  onGridToggle?: () => void;
  showWatermark?: boolean;
  onWatermarkToggle?: () => void;
}

const SettingsMenuItem = styled(MenuItem)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.08),
  },
}));

const IndicatorGroup = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  backgroundColor: alpha(theme.palette.background.default, 0.5),
}));

export const ChartSettingsMenu: React.FC<ChartSettingsMenuProps> = ({
  indicators,
  onIndicatorToggle,
  chartType = 'candle',
  onChartTypeChange,
  showGrid = true,
  onGridToggle,
  showWatermark = true,
  onWatermarkToggle,
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const indicatorConfig = [
    { id: 'sma', name: 'Moving Averages', icon: <TimelineIcon fontSize="small" /> },
    { id: 'volume', name: 'Volume', icon: <BarChartIcon fontSize="small" /> },
    { id: 'rsi', name: 'RSI', icon: <ShowChartIcon fontSize="small" /> },
    { id: 'macd', name: 'MACD', icon: <AnalyticsIcon fontSize="small" /> },
    { id: 'bollinger', name: 'Bollinger Bands', icon: <SpeedIcon fontSize="small" /> },
  ];

  return (
    <>
      <IconButton
        size="small"
        onClick={handleClick}
        sx={{
          '&:hover': {
            backgroundColor: alpha('#fff', 0.1),
          },
        }}
      >
        <SettingsIcon fontSize="small" />
      </IconButton>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        PaperProps={{
          elevation: 8,
          sx: {
            minWidth: 280,
            mt: 1,
            '& .MuiList-root': {
              py: 0.5,
            },
          },
        }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        {/* Chart Type Section */}
        <IndicatorGroup>
          <Typography variant="caption" color="text.secondary" fontWeight={600}>
            CHART TYPE
          </Typography>
        </IndicatorGroup>

        <SettingsMenuItem onClick={() => onChartTypeChange?.('candle')}>
          <ListItemIcon>
            <CandlestickChartIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Candlestick" />
          {chartType === 'candle' && <Typography variant="caption">✓</Typography>}
        </SettingsMenuItem>

        <SettingsMenuItem onClick={() => onChartTypeChange?.('line')}>
          <ListItemIcon>
            <ShowChartIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Line Chart" />
          {chartType === 'line' && <Typography variant="caption">✓</Typography>}
        </SettingsMenuItem>

        <Divider sx={{ my: 0.5 }} />

        {/* Indicators Section */}
        <IndicatorGroup>
          <Typography variant="caption" color="text.secondary" fontWeight={600}>
            INDICATORS
          </Typography>
        </IndicatorGroup>

        {indicatorConfig.map((indicator) => (
          <SettingsMenuItem
            key={indicator.id}
            onClick={() => onIndicatorToggle(indicator.id)}
          >
            <ListItemIcon>{indicator.icon}</ListItemIcon>
            <ListItemText primary={indicator.name} />
            <Switch
              edge="end"
              checked={indicators.includes(indicator.id)}
              size="small"
              onClick={(e) => e.stopPropagation()}
            />
          </SettingsMenuItem>
        ))}

        <Divider sx={{ my: 0.5 }} />

        {/* Display Options */}
        <IndicatorGroup>
          <Typography variant="caption" color="text.secondary" fontWeight={600}>
            DISPLAY
          </Typography>
        </IndicatorGroup>

        <SettingsMenuItem onClick={onGridToggle}>
          <ListItemIcon>
            <GridOnIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Grid Lines" />
          <Switch
            edge="end"
            checked={showGrid}
            size="small"
            onClick={(e) => e.stopPropagation()}
          />
        </SettingsMenuItem>

        <SettingsMenuItem onClick={onWatermarkToggle}>
          <ListItemIcon>
            <WaterDropIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Watermark" />
          <Switch
            edge="end"
            checked={showWatermark}
            size="small"
            onClick={(e) => e.stopPropagation()}
          />
        </SettingsMenuItem>
      </Menu>
    </>
  );
};

export default ChartSettingsMenu;
