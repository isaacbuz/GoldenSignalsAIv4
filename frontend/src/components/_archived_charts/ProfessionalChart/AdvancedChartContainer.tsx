import React, { useState, useEffect } from 'react';
import {
  Box,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  Divider,
  Typography,
  useTheme,
  alpha,
} from '@mui/material';
import {
  ShowChart as ShowChartIcon,
  GridView as GridViewIcon,
  Compare as CompareIcon,
  Settings as SettingsIcon,
  Save as SaveIcon,
  Share as ShareIcon,
  PhotoCamera as PhotoCameraIcon,
  ViewColumn as ViewColumnIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import ProfessionalChart from './ProfessionalChart';
import MultiTimeframeChart from './MultiTimeframeChart';
import ComparisonChart from './ComparisonChart';
import { chartSettingsService } from '../../services/chartSettingsService';
import StreamlinedGoldenChart from '../CustomChart/StreamlinedGoldenChart';

const Container = styled(Box)({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
});

const HeaderBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(1),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
}));

const ChartContent = styled(Box)(({ theme }) => ({
  flex: 1,
  overflow: 'hidden',
  position: 'relative',
}));

const StyledTabs = styled(Tabs)(({ theme }) => ({
  minHeight: 36,
  '& .MuiTab-root': {
    minHeight: 36,
    fontSize: '0.875rem',
    fontWeight: 600,
    minWidth: 120,
  },
}));

export interface AdvancedChartContainerProps {
  symbol: string;
  onSymbolChange?: (symbol: string) => void;
  height?: string | number;
}

type ChartView = 'standard' | 'multi-timeframe' | 'comparison';

export const AdvancedChartContainer: React.FC<AdvancedChartContainerProps> = ({
  symbol,
  onSymbolChange,
  height = '100%',
}) => {
  const theme = useTheme();
  const [view, setView] = useState<ChartView>('standard');
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);
  const [shareAnchor, setShareAnchor] = useState<null | HTMLElement>(null);
  const [layoutsAnchor, setLayoutsAnchor] = useState<null | HTMLElement>(null);
  const [savedLayouts, setSavedLayouts] = useState(chartSettingsService.getSavedLayouts());

  // Load settings on mount
  useEffect(() => {
    const settings = chartSettingsService.getSettings();
    if (settings.defaultView) {
      // Skip multi-timeframe if it was saved (since it's disabled now)
      if (settings.defaultView === 'multi-timeframe') {
        setView('standard');
      } else {
        setView(settings.defaultView);
      }
    }
  }, []);

  // Save settings when view changes
  useEffect(() => {
    chartSettingsService.saveSettings({
      defaultView: view,
    });
  }, [view]);

  const handleViewChange = (event: React.SyntheticEvent, newView: ChartView) => {
    setView(newView);
  };

  const handleSaveChart = () => {
    // Get current settings from service
    const currentSettings = chartSettingsService.getSettings();

    // Create layout name
    const layoutName = prompt('Enter a name for this chart layout:');
    if (!layoutName) return;

    // Save the layout
    const savedLayout = chartSettingsService.saveLayout(
      layoutName,
      symbol,
      {
        ...currentSettings,
        defaultView: view,
      }
    );

    // Refresh saved layouts
    setSavedLayouts(chartSettingsService.getSavedLayouts());

    alert(`Chart layout "${layoutName}" saved successfully!`);
  };

  const handleLoadLayout = (layoutId: string) => {
    const layout = chartSettingsService.loadLayout(layoutId);
    if (layout) {
      // Apply the layout settings
      if (layout.settings.defaultView) {
        setView(layout.settings.defaultView);
      }
      if (onSymbolChange && layout.symbol) {
        onSymbolChange(layout.symbol);
      }
      alert(`Loaded layout: ${layout.name}`);
    }
    setLayoutsAnchor(null);
  };

  const handleDeleteLayout = (layoutId: string) => {
    if (confirm('Are you sure you want to delete this layout?')) {
      chartSettingsService.deleteLayout(layoutId);
      setSavedLayouts(chartSettingsService.getSavedLayouts());
    }
  };

  const handleScreenshot = () => {
    // In a real implementation, this would capture the chart canvas
    alert('Screenshot feature coming soon!');
  };

  const handleShare = (method: string) => {
    const shareUrl = `${window.location.origin}/chart/${symbol}?view=${view}`;

    switch (method) {
      case 'copy':
        navigator.clipboard.writeText(shareUrl);
        alert('Link copied to clipboard!');
        break;
      case 'twitter':
        window.open(`https://twitter.com/intent/tweet?text=Check out this chart analysis for ${symbol}&url=${shareUrl}`, '_blank');
        break;
      case 'email':
        window.location.href = `mailto:?subject=Chart Analysis: ${symbol}&body=Check out this chart: ${shareUrl}`;
        break;
    }

    setShareAnchor(null);
  };

  const renderChart = () => {
    switch (view) {
      /* Multi-timeframe disabled for now
      case 'multi-timeframe':
        return (
          <MultiTimeframeChart
            symbol={symbol}
            onSymbolChange={onSymbolChange}
            height="100%"
          />
        );
      */

      case 'comparison':
        return (
          <ComparisonChart
            primarySymbol={symbol}
            height="100%"
            onSymbolChange={onSymbolChange}
          />
        );

      case 'standard':
      default:
        return (
          <StreamlinedGoldenChart
            symbol={symbol}
            onSymbolChange={onSymbolChange}
            showWatermark={true}
            height="100%"
          />
        );
    }
  };

  return (
    <Container sx={{ height }}>
      <HeaderBar>
        <StyledTabs value={view} onChange={handleViewChange}>
          <Tab
            icon={<ShowChartIcon fontSize="small" />}
            iconPosition="start"
            label="Standard"
            value="standard"
          />
          {/* Multi-Timeframe disabled for now - nice to have but not necessary
          <Tab
            icon={<GridViewIcon fontSize="small" />}
            iconPosition="start"
            label="Multi-Timeframe"
            value="multi-timeframe"
            disabled
          />
          */}
          <Tab
            icon={<CompareIcon fontSize="small" />}
            iconPosition="start"
            label="Comparison"
            value="comparison"
          />
        </StyledTabs>

        <Box display="flex" alignItems="center" gap={1}>
          <Tooltip title="Save/Load Chart Settings">
            <IconButton
              size="small"
              onClick={(e) => setLayoutsAnchor(e.currentTarget)}
            >
              <SaveIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title="Take Screenshot">
            <IconButton size="small" onClick={handleScreenshot}>
              <PhotoCameraIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title="Share">
            <IconButton
              size="small"
              onClick={(e) => setShareAnchor(e.currentTarget)}
            >
              <ShareIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Divider orientation="vertical" flexItem />

          <Tooltip title="Chart Settings">
            <IconButton
              size="small"
              onClick={(e) => setSettingsAnchor(e.currentTarget)}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </HeaderBar>

      <ChartContent>
        {renderChart()}
      </ChartContent>

      {/* Settings Menu */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
      >
        <MenuItem disabled>
          <Typography variant="body2">Chart Settings</Typography>
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => {
          chartSettingsService.saveSettings({ theme: 'dark' });
          setSettingsAnchor(null);
        }}>
          Dark Theme
        </MenuItem>
        <MenuItem onClick={() => {
          chartSettingsService.saveSettings({ theme: 'light' });
          setSettingsAnchor(null);
        }}>
          Light Theme
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => {
          chartSettingsService.saveSettings({ showVolume: true });
          setSettingsAnchor(null);
        }}>
          Show Volume
        </MenuItem>
        <MenuItem onClick={() => {
          chartSettingsService.saveSettings({ showGrid: true });
          setSettingsAnchor(null);
        }}>
          Show Grid
        </MenuItem>
      </Menu>

      {/* Share Menu */}
      <Menu
        anchorEl={shareAnchor}
        open={Boolean(shareAnchor)}
        onClose={() => setShareAnchor(null)}
      >
        <MenuItem onClick={() => handleShare('copy')}>
          Copy Link
        </MenuItem>
        <MenuItem onClick={() => handleShare('twitter')}>
          Share on Twitter
        </MenuItem>
        <MenuItem onClick={() => handleShare('email')}>
          Share via Email
        </MenuItem>
      </Menu>

      {/* Layouts Menu */}
      <Menu
        anchorEl={layoutsAnchor}
        open={Boolean(layoutsAnchor)}
        onClose={() => setLayoutsAnchor(null)}
      >
        <MenuItem onClick={handleSaveChart}>
          <SaveIcon fontSize="small" sx={{ mr: 1 }} />
          Save Current Layout
        </MenuItem>
        <Divider />
        {savedLayouts.length > 0 ? (
          <>
            <MenuItem disabled>
              <Typography variant="body2">Saved Layouts</Typography>
            </MenuItem>
            {savedLayouts.map((layout) => (
              <MenuItem
                key={layout.id}
                onClick={() => handleLoadLayout(layout.id)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="body2">{layout.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {layout.symbol} - {new Date(layout.updatedAt).toLocaleDateString()}
                    </Typography>
                  </Box>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteLayout(layout.id);
                    }}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>
              </MenuItem>
            ))}
          </>
        ) : (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              No saved layouts
            </Typography>
          </MenuItem>
        )}
      </Menu>
    </Container>
  );
};

export default AdvancedChartContainer;
