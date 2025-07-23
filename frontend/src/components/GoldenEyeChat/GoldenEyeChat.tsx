import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, TextField, IconButton, Paper, Typography, Chip, CircularProgress, Fade, Tooltip } from '@mui/material';
import { Send, AutoAwesome, TrendingUp, Psychology, Assessment, Warning } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { MCPClient } from '../../services/mcp/MCPClient';
import { MessageList } from './MessageList';
import { ThinkingIndicator } from './ThinkingIndicator';
import { AgentStatusPanel } from './AgentStatusPanel';
import './GoldenEyeChat.css';
import logger from '../../services/logger';


export interface ChartAction {
  type: 'draw_prediction' | 'add_agent_signals' | 'mark_entry_point' | 'mark_exit_point' | 'draw_levels' | 'highlight_pattern';
  data: any;
}

interface GoldenEyeChatProps {
  currentSymbol: string;
  onChartAction: (action: ChartAction) => void;
  chartTimeframe?: string;
  height?: string;
}

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'agent_consultation' | 'tool_execution' | 'error';
  content: string;
  timestamp: Date;
  agent?: string;
  data?: any;
}

interface ThinkingState {
  message: string;
  agents?: string[];
  llm?: string;
}

export const GoldenEyeChat: React.FC<GoldenEyeChatProps> = ({
  currentSymbol,
  onChartAction,
  chartTimeframe = '1h',
  height = '400px'
}) => {
  const theme = useTheme();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [thinking, setThinking] = useState<ThinkingState | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [agentStatus, setAgentStatus] = useState<Record<string, any>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mcpClient = useRef<MCPClient | null>(null);

  // Initialize MCP client
  useEffect(() => {
    mcpClient.current = new MCPClient();

    // Load initial suggestions
    setSuggestions([
      `Predict ${currentSymbol} price for next 24 hours`,
      `Show technical analysis for ${currentSymbol}`,
      `What's the risk assessment for ${currentSymbol}?`,
      `Should I enter a position in ${currentSymbol}?`,
      `Show support and resistance levels`
    ]);

    // Get agent status
    mcpClient.current.getAgentStatus().then(setAgentStatus);

    return () => {
      mcpClient.current?.disconnect();
    };
  }, [currentSymbol]);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, thinking]);

  // Process user query
  const processQuery = useCallback(async (query: string) => {
    if (!query.trim() || !mcpClient.current) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);
    setThinking(null);

    try {
      // Stream query to backend
      const stream = mcpClient.current.streamGoldenEyeQuery({
        query,
        symbol: currentSymbol,
        timeframe: chartTimeframe,
        context: {
          chartVisible: true,
          agentStatus
        }
      });

      let assistantMessage = '';
      let currentMessageId = Date.now().toString();

      for await (const event of stream) {
        switch (event.type) {
          case 'thinking':
            setThinking({
              message: event.message,
              agents: event.agents,
              llm: event.llm
            });
            break;

          case 'text':
            assistantMessage += event.content;
            // Update or create assistant message
            setMessages(prev => {
              const existing = prev.find(m => m.id === currentMessageId);
              if (existing) {
                return prev.map(m =>
                  m.id === currentMessageId
                    ? { ...m, content: assistantMessage }
                    : m
                );
              } else {
                return [...prev, {
                  id: currentMessageId,
                  type: 'assistant' as const,
                  content: assistantMessage,
                  timestamp: new Date()
                }];
              }
            });
            break;

          case 'tool_execution':
            setMessages(prev => [...prev, {
              id: Date.now().toString(),
              type: 'tool_execution',
              content: `Executed ${event.tool}`,
              timestamp: new Date(),
              data: event.result
            }]);
            break;

          case 'agent_consultation':
            setMessages(prev => [...prev, {
              id: Date.now().toString(),
              type: 'agent_consultation',
              content: `Consulted ${event.agent}`,
              timestamp: new Date(),
              agent: event.agent,
              data: event.result
            }]);
            break;

          case 'chart_action':
            onChartAction(event.action);
            break;

          case 'error':
            setMessages(prev => [...prev, {
              id: Date.now().toString(),
              type: 'error',
              content: event.message,
              timestamp: new Date()
            }]);
            break;
        }
      }

    } catch (error) {
      logger.error('Query processing error:', error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        type: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date()
      }]);
    } finally {
      setIsProcessing(false);
      setThinking(null);
    }
  }, [currentSymbol, chartTimeframe, agentStatus, onChartAction]);

  // Handle input submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    processQuery(input);
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    processQuery(suggestion);
  };

  // Get intent icon
  const getIntentIcon = (intent?: string) => {
    switch (intent) {
      case 'prediction':
        return <TrendingUp />;
      case 'technical_analysis':
        return <Assessment />;
      case 'risk_assessment':
        return <Warning />;
      case 'market_sentiment':
        return <Psychology />;
      default:
        return <AutoAwesome />;
    }
  };

  return (
    <Paper
      sx={{
        height,
        display: 'flex',
        flexDirection: 'column',
        bgcolor: theme.palette.mode === 'dark' ? 'rgba(18, 18, 18, 0.95)' : 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.2)' : 'rgba(0, 0, 0, 0.1)'}`,
        borderRadius: 2,
        overflow: 'hidden',
        position: 'relative'
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: `1px solid ${theme.palette.divider}`,
          background: theme.palette.mode === 'dark'
            ? 'linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%)'
            : 'linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 165, 0, 0.02) 100%)'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AutoAwesome sx={{ color: theme.palette.primary.main }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Golden Eye AI Prophet
          </Typography>
          <Box sx={{ ml: 'auto' }}>
            <AgentStatusPanel status={agentStatus} />
          </Box>
        </Box>
      </Box>

      {/* Messages Area */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        <MessageList messages={messages} />

        {thinking && (
          <Fade in={true}>
            <Box>
              <ThinkingIndicator state={thinking} />
            </Box>
          </Fade>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Suggestions */}
      {!isProcessing && messages.length === 0 && (
        <Box sx={{ px: 2, pb: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
            Try asking:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {suggestions.map((suggestion, index) => (
              <Chip
                key={index}
                label={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                sx={{
                  cursor: 'pointer',
                  '&:hover': {
                    bgcolor: theme.palette.action.hover
                  }
                }}
                icon={getIntentIcon()}
                size="small"
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Input Area */}
      <Box
        component="form"
        onSubmit={handleSubmit}
        sx={{
          p: 2,
          borderTop: `1px solid ${theme.palette.divider}`,
          display: 'flex',
          gap: 1
        }}
      >
        <TextField
          fullWidth
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Ask about ${currentSymbol}...`}
          disabled={isProcessing}
          size="small"
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 2
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <Tooltip title="Send message (Enter)">
          <IconButton
            type="submit"
            disabled={!input.trim() || isProcessing}
            sx={{
              bgcolor: theme.palette.primary.main,
              color: 'white',
              '&:hover': {
                bgcolor: theme.palette.primary.dark
              },
              '&:disabled': {
                bgcolor: theme.palette.action.disabledBackground
              }
            }}
          >
            {isProcessing ? <CircularProgress size={20} color="inherit" /> : <Send />}
          </IconButton>
        </Tooltip>
      </Box>
    </Paper>
  );
};
