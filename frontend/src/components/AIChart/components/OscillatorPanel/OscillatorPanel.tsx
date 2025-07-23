/**
 * OscillatorPanel Component
 *
 * Renders oscillator indicators (RSI, MACD, Stochastic, etc.) in separate panels
 * below the main chart. Each oscillator gets its own canvas with independent scaling.
 */

import React, { useRef, useEffect, useCallback } from 'react';
import { Box, IconButton, Tooltip, useTheme } from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { ChartDataPoint } from '../ChartCanvas/types';
import { useOscillatorDrawing } from '../../hooks/useOscillatorDrawing';

interface OscillatorPanelProps {
  data: ChartDataPoint[];
  indicators: string[];
  width: number;
  height?: number;
  onRemoveIndicator?: (indicator: string) => void;
}

const PANEL_HEIGHT = 100; // Height for each oscillator panel
const PANEL_MARGIN = 4;

export const OscillatorPanel: React.FC<OscillatorPanelProps> = ({
  data,
  indicators,
  width,
  height = PANEL_HEIGHT,
  onRemoveIndicator,
}) => {
  const theme = useTheme();
  const canvasRefs = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const { drawOscillatorPanel } = useOscillatorDrawing();

  /**
   * Set canvas ref for each indicator
   */
  const setCanvasRef = useCallback((indicator: string, canvas: HTMLCanvasElement | null) => {
    if (canvas) {
      canvasRefs.current.set(indicator, canvas);
    } else {
      canvasRefs.current.delete(indicator);
    }
  }, []);

  /**
   * Draw oscillator on its canvas
   */
  const drawOscillator = useCallback((indicator: string, canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || data.length === 0) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw oscillator
    drawOscillatorPanel({
      ctx,
      data,
      indicator,
      x: 0,
      y: 0,
      width: canvas.width,
      height: canvas.height,
      theme,
    });
  }, [data, theme, drawOscillatorPanel]);

  /**
   * Redraw all oscillators when data or theme changes
   */
  useEffect(() => {
    indicators.forEach(indicator => {
      const canvas = canvasRefs.current.get(indicator);
      if (canvas) {
        // Set canvas resolution
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;

        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }

        drawOscillator(indicator, canvas);
      }
    });
  }, [indicators, data, theme, drawOscillator]);

  // Filter only oscillator indicators
  const oscillatorIndicators = indicators.filter(ind =>
    ['rsi', 'macd', 'stochastic', 'atr', 'adx'].includes(ind)
  );

  if (oscillatorIndicators.length === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        width: '100%',
        backgroundColor: theme.palette.background.paper,
        borderTop: `1px solid ${theme.palette.divider}`,
      }}
    >
      {oscillatorIndicators.map((indicator, index) => (
        <Box
          key={indicator}
          sx={{
            position: 'relative',
            width: '100%',
            height,
            marginTop: index > 0 ? `${PANEL_MARGIN}px` : 0,
            borderTop: index > 0 ? `1px solid ${theme.palette.divider}` : 'none',
          }}
        >
          <canvas
            ref={(ref) => setCanvasRef(indicator, ref)}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
            }}
          />

          {/* Remove button */}
          {onRemoveIndicator && (
            <Tooltip title={`Remove ${indicator.toUpperCase()}`}>
              <IconButton
                size="small"
                onClick={() => onRemoveIndicator(indicator)}
                sx={{
                  position: 'absolute',
                  top: 4,
                  right: 4,
                  backgroundColor: theme.palette.background.paper,
                  '&:hover': {
                    backgroundColor: theme.palette.action.hover,
                  },
                }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      ))}
    </Box>
  );
};
