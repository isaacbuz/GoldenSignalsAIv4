import React from 'react';
import { Box, Typography, Paper, Chip, Avatar } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Person, SmartToy, Build, Error as ErrorIcon, ShowChart, TrendingUp, BarChart } from '@mui/icons-material';
import { format } from 'date-fns';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'agent_consultation' | 'tool_execution' | 'error';
  content: string;
  timestamp: Date;
  agent?: string;
  data?: any;
}

interface MessageListProps {
  messages: Message[];
}

const agentIcons: Record<string, React.ReactNode> = {
  RSIAgent: <ShowChart />,
  MACDAgent: <TrendingUp />,
  PatternAgent: <BarChart />,
  VolumeAgent: <BarChart />,
  SentimentAgent: <SmartToy />,
  LSTMForecastAgent: <TrendingUp />,
  OptionsChainAgent: <ShowChart />,
  MarketRegimeAgent: <BarChart />,
  MomentumAgent: <TrendingUp />
};

export const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  const theme = useTheme();

  const getMessageStyle = (type: string) => {
    switch (type) {
      case 'user':
        return {
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.1)' : 'rgba(255, 215, 0, 0.05)',
          borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.3)' : 'rgba(255, 215, 0, 0.2)',
          alignSelf: 'flex-end',
          maxWidth: '80%'
        };
      case 'assistant':
        return {
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(66, 165, 245, 0.1)' : 'rgba(66, 165, 245, 0.05)',
          borderColor: theme.palette.mode === 'dark' ? 'rgba(66, 165, 245, 0.3)' : 'rgba(66, 165, 245, 0.2)',
          alignSelf: 'flex-start',
          maxWidth: '80%'
        };
      case 'agent_consultation':
        return {
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.05)',
          borderColor: theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.3)' : 'rgba(76, 175, 80, 0.2)',
          alignSelf: 'center',
          maxWidth: '90%'
        };
      case 'tool_execution':
        return {
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(156, 39, 176, 0.1)' : 'rgba(156, 39, 176, 0.05)',
          borderColor: theme.palette.mode === 'dark' ? 'rgba(156, 39, 176, 0.3)' : 'rgba(156, 39, 176, 0.2)',
          alignSelf: 'center',
          maxWidth: '90%'
        };
      case 'error':
        return {
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.05)',
          borderColor: theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.3)' : 'rgba(244, 67, 54, 0.2)',
          alignSelf: 'center',
          maxWidth: '90%'
        };
      default:
        return {};
    }
  };

  const getIcon = (message: Message) => {
    switch (message.type) {
      case 'user':
        return <Person />;
      case 'assistant':
        return <SmartToy />;
      case 'agent_consultation':
        return agentIcons[message.agent || ''] || <SmartToy />;
      case 'tool_execution':
        return <Build />;
      case 'error':
        return <ErrorIcon />;
      default:
        return <SmartToy />;
    }
  };

  const formatContent = (message: Message) => {
    if (message.type === 'agent_consultation' && message.data) {
      return (
        <Box>
          <Typography variant="body2">{message.content}</Typography>
          {message.data.signal && (
            <Chip
              label={`Signal: ${message.data.signal}`}
              size="small"
              sx={{ mt: 1, mr: 1 }}
              color={message.data.signal === 'BUY' ? 'success' : message.data.signal === 'SELL' ? 'error' : 'default'}
            />
          )}
          {message.data.confidence !== undefined && (
            <Chip
              label={`Confidence: ${(message.data.confidence * 100).toFixed(0)}%`}
              size="small"
              sx={{ mt: 1 }}
              color={message.data.confidence > 0.7 ? 'success' : message.data.confidence > 0.4 ? 'warning' : 'default'}
            />
          )}
        </Box>
      );
    }

    if (message.type === 'tool_execution' && message.data) {
      return (
        <Box>
          <Typography variant="body2">{message.content}</Typography>
          {message.data.success === false && (
            <Typography variant="caption" color="error" sx={{ mt: 0.5, display: 'block' }}>
              Error: {message.data.error}
            </Typography>
          )}
        </Box>
      );
    }

    // For assistant messages, preserve formatting
    if (message.type === 'assistant') {
      return (
        <Typography
          variant="body2"
          component="div"
          sx={{ whiteSpace: 'pre-wrap' }}
        >
          {message.content}
        </Typography>
      );
    }

    return <Typography variant="body2">{message.content}</Typography>;
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {messages.map((message) => (
        <Box
          key={message.id}
          sx={{
            display: 'flex',
            ...getMessageStyle(message.type)
          }}
        >
          <Paper
            elevation={0}
            sx={{
              p: 2,
              border: '1px solid',
              ...getMessageStyle(message.type),
              borderRadius: 2,
              position: 'relative'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'
                }}
              >
                {getIcon(message)}
              </Avatar>
              <Box sx={{ flex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    {message.type === 'user' ? 'You' :
                     message.type === 'assistant' ? 'Golden Eye AI' :
                     message.type === 'agent_consultation' ? message.agent :
                     message.type === 'tool_execution' ? 'Tool Execution' :
                     'System'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {format(message.timestamp, 'HH:mm:ss')}
                  </Typography>
                </Box>
                {formatContent(message)}
              </Box>
            </Box>
          </Paper>
        </Box>
      ))}
    </Box>
  );
};
