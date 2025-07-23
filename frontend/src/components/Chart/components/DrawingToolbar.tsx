/**
 * Drawing toolbar component
 * Provides tools for chart annotations and technical analysis drawings
 */

import React from 'react';
import { Box, ToggleButton, ToggleButtonGroup, Tooltip } from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Timeline as LineIcon,
  ShowChart as TrendLineIcon,
  HorizontalRule as HorizontalLineIcon,
  Rectangle as RectangleIcon,
  Circle as CircleIcon,
  Create as PencilIcon,
  TextFields as TextIcon,
  Functions as FibonacciIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { IChartApi } from 'lightweight-charts';

const ToolbarContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 80,
  left: 20,
  backgroundColor: '#1e222d',
  border: '1px solid #2a2e39',
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(1),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  zIndex: 100,
}));

interface DrawingToolbarProps {
  mode: string | null;
  onModeChange: (mode: string | null) => void;
  chart: IChartApi;
  onDrawingComplete?: (drawing: any) => void;
}

export const DrawingToolbar: React.FC<DrawingToolbarProps> = ({
  mode,
  onModeChange,
  chart,
  onDrawingComplete,
}) => {
  const handleModeChange = (_: React.MouseEvent<HTMLElement>, newMode: string | null) => {
    onModeChange(newMode);

    // Configure chart interaction based on mode
    if (newMode) {
      chart.applyOptions({
        handleScroll: { vertTouchDrag: false, horzTouchDrag: false },
        handleScale: { axisPressedMouseMove: false },
      });
    } else {
      chart.applyOptions({
        handleScroll: { vertTouchDrag: true, horzTouchDrag: true },
        handleScale: { axisPressedMouseMove: true },
      });
    }
  };

  const clearDrawings = () => {
    // Clear all drawings
    onModeChange(null);
    // TODO: Implement clear functionality
  };

  return (
    <ToolbarContainer>
      <ToggleButtonGroup
        orientation="vertical"
        value={mode}
        exclusive
        onChange={handleModeChange}
        size="small"
        sx={{
          '& .MuiToggleButton-root': {
            color: '#787b86',
            borderColor: '#2a2e39',
            padding: '8px',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.05)',
            },
            '&.Mui-selected': {
              backgroundColor: 'rgba(33, 150, 243, 0.15)',
              color: '#2196f3',
              '&:hover': {
                backgroundColor: 'rgba(33, 150, 243, 0.25)',
              },
            },
          },
        }}
      >
        <ToggleButton value="line">
          <Tooltip title="Line" placement="right">
            <LineIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="trend">
          <Tooltip title="Trend Line" placement="right">
            <TrendLineIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="horizontal">
          <Tooltip title="Horizontal Line" placement="right">
            <HorizontalLineIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="rectangle">
          <Tooltip title="Rectangle" placement="right">
            <RectangleIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="circle">
          <Tooltip title="Circle" placement="right">
            <CircleIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="fibonacci">
          <Tooltip title="Fibonacci Retracement" placement="right">
            <FibonacciIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="text">
          <Tooltip title="Text" placement="right">
            <TextIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="pencil">
          <Tooltip title="Free Draw" placement="right">
            <PencilIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
      </ToggleButtonGroup>

      <Tooltip title="Clear All Drawings" placement="right">
        <ToggleButton
          value="clear"
          size="small"
          onClick={clearDrawings}
          sx={{
            color: '#ef5350',
            borderColor: '#2a2e39',
            padding: '8px',
            '&:hover': {
              backgroundColor: 'rgba(239, 83, 80, 0.1)',
            },
          }}
        >
          <ClearIcon fontSize="small" />
        </ToggleButton>
      </Tooltip>
    </ToolbarContainer>
  );
};
