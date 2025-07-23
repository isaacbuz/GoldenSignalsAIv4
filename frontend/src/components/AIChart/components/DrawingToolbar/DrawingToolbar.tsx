/**
 * DrawingToolbar Component
 *
 * Provides UI for selecting and managing drawing tools on the chart.
 * Includes tools for trendlines, horizontal lines, fibonacci, and rectangles.
 */

import React from 'react';
import {
  Box,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
  Tooltip,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';
import {
  ShowChart as TrendlineIcon,
  HorizontalRule as HorizontalIcon,
  Timeline as FibonacciIcon,
  CropDin as RectangleIcon,
  Delete as DeleteIcon,
  PanTool as PanIcon,
} from '@mui/icons-material';
import { DrawingTool } from '../../hooks/useDrawingTools';

interface DrawingToolbarProps {
  selectedTool: DrawingTool;
  onToolChange: (tool: DrawingTool) => void;
  onClearAll: () => void;
  drawingCount?: number;
  position?: 'left' | 'top';
}

export const DrawingToolbar: React.FC<DrawingToolbarProps> = ({
  selectedTool,
  onToolChange,
  onClearAll,
  drawingCount = 0,
  position = 'left',
}) => {
  const theme = useTheme();

  const handleToolChange = (
    event: React.MouseEvent<HTMLElement>,
    newTool: DrawingTool | null
  ) => {
    if (newTool !== null) {
      onToolChange(newTool);
    }
  };

  const tools = [
    { value: 'none' as DrawingTool, icon: <PanIcon />, label: 'Select/Pan' },
    { value: 'trendline' as DrawingTool, icon: <TrendlineIcon />, label: 'Trendline' },
    { value: 'horizontal' as DrawingTool, icon: <HorizontalIcon />, label: 'Horizontal Line' },
    { value: 'fibonacci' as DrawingTool, icon: <FibonacciIcon />, label: 'Fibonacci Retracement' },
    { value: 'rectangle' as DrawingTool, icon: <RectangleIcon />, label: 'Rectangle' },
  ];

  const containerStyles = position === 'left'
    ? {
        position: 'absolute' as const,
        left: 16,
        top: '50%',
        transform: 'translateY(-50%)',
        flexDirection: 'column' as const,
        zIndex: 10,
      }
    : {
        position: 'absolute' as const,
        top: 80,
        left: '50%',
        transform: 'translateX(-50%)',
        flexDirection: 'row' as const,
        zIndex: 10,
      };

  return (
    <Box
      sx={{
        ...containerStyles,
        display: 'flex',
        gap: 1,
        backgroundColor: alpha(theme.palette.background.paper, 0.9),
        backdropFilter: 'blur(10px)',
        borderRadius: 1,
        p: 1,
        boxShadow: theme.shadows[3],
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}
    >
      <ToggleButtonGroup
        value={selectedTool}
        exclusive
        onChange={handleToolChange}
        orientation={position === 'left' ? 'vertical' : 'horizontal'}
        size="small"
        sx={{
          '& .MuiToggleButton-root': {
            p: 1,
            color: theme.palette.text.secondary,
            '&.Mui-selected': {
              backgroundColor: alpha(theme.palette.primary.main, 0.2),
              color: theme.palette.primary.main,
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.3),
              },
            },
          },
        }}
      >
        {tools.map((tool) => (
          <ToggleButton key={tool.value} value={tool.value}>
            <Tooltip title={tool.label} placement={position === 'left' ? 'right' : 'bottom'}>
              <Box>{tool.icon}</Box>
            </Tooltip>
          </ToggleButton>
        ))}
      </ToggleButtonGroup>

      {drawingCount > 0 && (
        <>
          <Divider orientation={position === 'left' ? 'horizontal' : 'vertical'} flexItem />
          <Tooltip title={`Clear all drawings (${drawingCount})`} placement={position === 'left' ? 'right' : 'bottom'}>
            <IconButton
              size="small"
              onClick={onClearAll}
              sx={{
                color: theme.palette.error.main,
                '&:hover': {
                  backgroundColor: alpha(theme.palette.error.main, 0.1),
                },
              }}
            >
              <DeleteIcon />
            </IconButton>
          </Tooltip>
        </>
      )}
    </Box>
  );
};
