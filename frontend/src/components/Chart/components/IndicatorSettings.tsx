/**
 * Indicator settings component
 * Provides UI for configuring technical indicators
 */

import React from 'react';
import {
  Box,
  Typography,
  Switch,
  FormGroup,
  FormControlLabel,
  Divider,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const SettingsContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: '#1e222d',
  borderRadius: theme.shape.borderRadius,
  minWidth: 250,
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  color: '#d1d4dc',
  fontWeight: 600,
  marginBottom: theme.spacing(1),
}));

interface IndicatorConfig {
  ma20: boolean;
  ma50: boolean;
  ma200: boolean;
  bollinger: boolean;
  volume: boolean;
  rsi: boolean;
  macd: boolean;
  vwap?: boolean;
  atr?: boolean;
  stochastic?: boolean;
}

interface IndicatorSettingsProps {
  indicators: IndicatorConfig;
  onChange: (indicators: IndicatorConfig) => void;
  onClose?: () => void;
}

export const IndicatorSettings: React.FC<IndicatorSettingsProps> = ({
  indicators,
  onChange,
  onClose,
}) => {
  const handleIndicatorChange = (key: keyof IndicatorConfig, value: boolean) => {
    onChange({ ...indicators, [key]: value });
  };

  return (
    <SettingsContainer>
      <SectionTitle variant="h6">Technical Indicators</SectionTitle>

      <Accordion
        defaultExpanded
        sx={{
          backgroundColor: 'transparent',
          boxShadow: 'none',
          '&:before': { display: 'none' },
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: '#787b86' }} />}
          sx={{ px: 0, minHeight: 'auto' }}
        >
          <Typography variant="subtitle2" sx={{ color: '#d1d4dc' }}>
            Moving Averages
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0 }}>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.ma20}
                  onChange={(e) => handleIndicatorChange('ma20', e.target.checked)}
                />
              }
              label="MA 20"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.ma50}
                  onChange={(e) => handleIndicatorChange('ma50', e.target.checked)}
                />
              }
              label="MA 50"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.ma200}
                  onChange={(e) => handleIndicatorChange('ma200', e.target.checked)}
                />
              }
              label="MA 200"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
          </FormGroup>
        </AccordionDetails>
      </Accordion>

      <Divider sx={{ borderColor: '#2a2e39', my: 1 }} />

      <Accordion
        defaultExpanded
        sx={{
          backgroundColor: 'transparent',
          boxShadow: 'none',
          '&:before': { display: 'none' },
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: '#787b86' }} />}
          sx={{ px: 0, minHeight: 'auto' }}
        >
          <Typography variant="subtitle2" sx={{ color: '#d1d4dc' }}>
            Oscillators
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0 }}>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.rsi}
                  onChange={(e) => handleIndicatorChange('rsi', e.target.checked)}
                />
              }
              label="RSI (14)"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.macd}
                  onChange={(e) => handleIndicatorChange('macd', e.target.checked)}
                />
              }
              label="MACD (12, 26, 9)"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.stochastic || false}
                  onChange={(e) => handleIndicatorChange('stochastic', e.target.checked)}
                />
              }
              label="Stochastic (14, 3, 3)"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
          </FormGroup>
        </AccordionDetails>
      </Accordion>

      <Divider sx={{ borderColor: '#2a2e39', my: 1 }} />

      <Accordion
        sx={{
          backgroundColor: 'transparent',
          boxShadow: 'none',
          '&:before': { display: 'none' },
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: '#787b86' }} />}
          sx={{ px: 0, minHeight: 'auto' }}
        >
          <Typography variant="subtitle2" sx={{ color: '#d1d4dc' }}>
            Other Indicators
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0 }}>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.bollinger}
                  onChange={(e) => handleIndicatorChange('bollinger', e.target.checked)}
                />
              }
              label="Bollinger Bands"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.volume}
                  onChange={(e) => handleIndicatorChange('volume', e.target.checked)}
                />
              }
              label="Volume"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.vwap || false}
                  onChange={(e) => handleIndicatorChange('vwap', e.target.checked)}
                />
              }
              label="VWAP"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.atr || false}
                  onChange={(e) => handleIndicatorChange('atr', e.target.checked)}
                />
              }
              label="ATR (14)"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
          </FormGroup>
        </AccordionDetails>
      </Accordion>
    </SettingsContainer>
  );
};
