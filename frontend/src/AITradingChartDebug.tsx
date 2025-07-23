import React, { useState } from 'react';
import { Box, Typography, Button } from '@mui/material';
import { styled } from '@mui/material/styles';

// Test service imports one by one
// import logger from './services/logger';

const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  minHeight: '400px',
  backgroundColor: '#000000',
  display: 'flex',
  flexDirection: 'column',
}));

const Header = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(2),
  background: 'linear-gradient(180deg, rgba(0, 0, 0, 0.8) 0%, rgba(0, 0, 0, 0.6) 100%)',
  backdropFilter: 'blur(20px)',
  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  zIndex: 10,
}));

const AITradingChartDebug: React.FC = () => {
  const [step, setStep] = useState(0);

  const loadSteps = [
    { name: 'Basic Component', loaded: true },
    { name: 'Services', loaded: false },
    { name: 'Hooks', loaded: false },
    { name: 'Canvas', loaded: false },
  ];

  return (
    <ChartContainer>
      <Header>
        <Typography variant="h4" sx={{ color: '#FFD700' }}>
          AITradingChart Debug
        </Typography>
        <Button
          variant="contained"
          onClick={() => setStep(step + 1)}
          sx={{ backgroundColor: '#007AFF' }}
        >
          Load Next: {loadSteps[Math.min(step, loadSteps.length - 1)].name}
        </Button>
      </Header>

      <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h5" sx={{ mb: 3 }}>
            Component Loading Status
          </Typography>
          {loadSteps.map((item, index) => (
            <Typography
              key={item.name}
              sx={{
                color: index <= step ? '#00D964' : '#666',
                mb: 1
              }}
            >
              {index <= step ? '✓' : '○'} {item.name}
            </Typography>
          ))}

          {step === 0 && (
            <Typography sx={{ mt: 4, color: '#999' }}>
              Basic component loads successfully!
            </Typography>
          )}

          {step === 1 && (
            <Box sx={{ mt: 4 }}>
              <Typography sx={{ color: '#FFD700' }}>
                Testing service imports...
              </Typography>
            </Box>
          )}
        </Box>
      </Box>
    </ChartContainer>
  );
};

export default AITradingChartDebug;
