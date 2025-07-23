import React from 'react';
import { Box, Typography, LinearProgress, Chip, Avatar } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Psychology, AutoAwesome, Analytics } from '@mui/icons-material';

interface ThinkingState {
  message: string;
  agents?: string[];
  llm?: string;
}

interface ThinkingIndicatorProps {
  state: ThinkingState;
}

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ state }) => {
  const theme = useTheme();

  const getLLMInfo = (llm?: string) => {
    switch (llm) {
      case 'openai':
        return { name: 'GPT-4o', color: '#00A67E' };
      case 'anthropic':
        return { name: 'Claude 3 Opus', color: '#6B46C1' };
      case 'grok':
        return { name: 'Grok 4', color: '#1DA1F2' };
      default:
        return { name: 'AI', color: theme.palette.primary.main };
    }
  };

  const llmInfo = getLLMInfo(state.llm);

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 2,
        bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.05)' : 'rgba(255, 215, 0, 0.02)',
        border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.2)' : 'rgba(255, 215, 0, 0.1)'}`,
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      {/* Animated gradient background */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `linear-gradient(45deg, transparent 30%, ${theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.1)' : 'rgba(255, 215, 0, 0.05)'} 50%, transparent 70%)`,
          backgroundSize: '200% 200%',
          animation: 'thinking-wave 3s ease-in-out infinite',
          '@keyframes thinking-wave': {
            '0%': { backgroundPosition: '0% 50%' },
            '50%': { backgroundPosition: '100% 50%' },
            '100%': { backgroundPosition: '0% 50%' }
          }
        }}
      />

      <Box sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Psychology
            sx={{
              color: theme.palette.primary.main,
              animation: 'pulse 2s ease-in-out infinite',
              '@keyframes pulse': {
                '0%': { opacity: 0.6 },
                '50%': { opacity: 1 },
                '100%': { opacity: 0.6 }
              }
            }}
          />
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {state.message}
          </Typography>
        </Box>

        {state.llm && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Chip
              avatar={<Avatar sx={{ bgcolor: llmInfo.color, width: 24, height: 24 }}><AutoAwesome sx={{ fontSize: 16 }} /></Avatar>}
              label={llmInfo.name}
              size="small"
              sx={{
                bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
                '& .MuiChip-avatar': {
                  color: 'white'
                }
              }}
            />
          </Box>
        )}

        {state.agents && state.agents.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Consulting agents:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {state.agents.map((agent, index) => (
                <Chip
                  key={index}
                  icon={<Analytics />}
                  label={agent}
                  size="small"
                  sx={{
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.2)' : 'rgba(76, 175, 80, 0.1)',
                    '& .MuiChip-icon': {
                      color: theme.palette.success.main
                    }
                  }}
                />
              ))}
            </Box>
          </Box>
        )}

        <LinearProgress
          sx={{
            mt: 2,
            height: 2,
            borderRadius: 1,
            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
            '& .MuiLinearProgress-bar': {
              background: 'linear-gradient(90deg, #FFD700, #FFA500)',
              borderRadius: 1
            }
          }}
        />
      </Box>
    </Box>
  );
};
