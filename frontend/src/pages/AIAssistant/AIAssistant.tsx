import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  IconButton,
  Button,
  Avatar,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Paper,
  Tooltip,
  CircularProgress,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Send,
  AutoAwesome,
  Mic,
  AttachFile,
  MoreVert,
  ContentCopy,
  ThumbUp,
  ThumbDown,
  Refresh,
  Psychology,
  TipsAndUpdates,
  Code,
  ShowChart,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';

const ChatContainer = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: 'calc(100vh - 200px)',
  display: 'flex',
  flexDirection: 'column',
}));

const MessageBubble = styled(Box)(({ theme, isUser }: { theme: any; isUser: boolean }) => ({
  maxWidth: '70%',
  padding: theme.spacing(2),
  borderRadius: theme.spacing(2),
  marginBottom: theme.spacing(2),
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  backgroundColor: isUser 
    ? 'rgba(255, 215, 0, 0.1)' 
    : 'rgba(255, 255, 255, 0.05)',
  border: `1px solid ${isUser ? 'rgba(255, 215, 0, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
}));

const QuickActionButton = styled(Button)(({ theme }) => ({
  textTransform: 'none',
  borderRadius: theme.spacing(3),
  padding: theme.spacing(1, 2),
  border: '1px solid rgba(255, 215, 0, 0.3)',
  '&:hover': {
    backgroundColor: 'rgba(255, 215, 0, 0.1)',
    borderColor: 'rgba(255, 215, 0, 0.5)',
  },
}));

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type?: 'text' | 'signal' | 'analysis' | 'code';
  metadata?: any;
}

const AIAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! I'm your AI Signal Intelligence Assistant. I can help you understand market signals, analyze trends, and create custom alerts. What would you like to know?",
      sender: 'ai',
      timestamp: new Date(),
      type: 'text',
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedMessage, setSelectedMessage] = useState<string | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const quickActions = [
    { label: 'Latest signals', icon: <ShowChart /> },
    { label: 'Market analysis', icon: <Psychology /> },
    { label: 'Create alert', icon: <TipsAndUpdates /> },
    { label: 'Explain signal', icon: <AutoAwesome /> },
  ];

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
      type: 'text',
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: generateAIResponse(inputMessage),
        sender: 'ai',
        timestamp: new Date(),
        type: detectMessageType(inputMessage),
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const generateAIResponse = (input: string): string => {
    const lowerInput = input.toLowerCase();
    
    if (lowerInput.includes('signal') && (lowerInput.includes('latest') || lowerInput.includes('recent'))) {
      return "Here are the latest high-confidence signals:\n\n📈 NVDA - BUY (94.1% confidence)\nConsensus from flow, technical, and sentiment agents. Strong institutional buying detected.\n\n📊 AAPL - BUY (92.5% confidence)\nPositive sentiment and technical breakout pattern identified.\n\n📉 TSLA - SELL (85.7% confidence)\nRisk indicators elevated, bearish divergence detected.";
    } else if (lowerInput.includes('explain')) {
      return "Let me explain this signal for you:\n\nThe BUY signal is generated through our multi-agent consensus system. Here's how it works:\n\n1. **Sentiment Analysis** (92% bullish): Scanning news and social media shows overwhelmingly positive sentiment\n2. **Technical Indicators** (94% bullish): RSI, MACD, and moving averages all align bullishly\n3. **Options Flow** (96% bullish): Large institutional call buying detected\n4. **Risk Assessment** (Low): Volatility within acceptable ranges\n\nThe final consensus of 94.1% indicates a very strong buy signal.";
    } else if (lowerInput.includes('alert')) {
      return "I'll help you create a custom alert. What conditions would you like to monitor?\n\nExample alerts I can set up:\n- When consensus confidence > 90%\n- When specific stocks generate signals\n- When multiple agents agree on a direction\n- When unusual options activity is detected\n\nJust describe what you want to track!";
    } else {
      return "I understand you're asking about: \"" + input + "\"\n\nI can help with:\n- Analyzing current market signals\n- Explaining AI consensus decisions\n- Creating custom alerts\n- Identifying trading opportunities\n- Understanding agent reasoning\n\nWhat specific aspect would you like me to focus on?";
    }
  };

  const detectMessageType = (input: string): 'text' | 'signal' | 'analysis' | 'code' => {
    const lowerInput = input.toLowerCase();
    if (lowerInput.includes('signal')) return 'signal';
    if (lowerInput.includes('analyze') || lowerInput.includes('analysis')) return 'analysis';
    if (lowerInput.includes('code') || lowerInput.includes('script')) return 'code';
    return 'text';
  };

  const handleQuickAction = (action: string) => {
    setInputMessage(action);
    handleSendMessage();
  };

  const handleVoiceInput = () => {
    setIsListening(!isListening);
    // Voice input implementation would go here
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
          AI Assistant
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Your intelligent market analysis companion
        </Typography>
      </Box>

      {/* Quick Actions */}
      <Box sx={{ mb: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {quickActions.map((action) => (
          <QuickActionButton
            key={action.label}
            startIcon={action.icon}
            onClick={() => handleQuickAction(action.label)}
            size="small"
          >
            {action.label}
          </QuickActionButton>
        ))}
      </Box>

      {/* Chat Container */}
      <ChatContainer>
        <CardContent sx={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
          {messages.map((message) => (
            <Box
              key={message.id}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: message.sender === 'user' ? 'flex-end' : 'flex-start',
              }}
            >
              {message.sender === 'ai' && (
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar
                    sx={{
                      width: 32,
                      height: 32,
                      backgroundColor: 'rgba(255, 215, 0, 0.2)',
                      mr: 1,
                    }}
                  >
                    <AutoAwesome sx={{ fontSize: 18, color: '#FFD700' }} />
                  </Avatar>
                  <Typography variant="caption" color="text.secondary">
                    AI Assistant
                  </Typography>
                </Box>
              )}
              
              <MessageBubble isUser={message.sender === 'user'}>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {message.content}
                </Typography>
                
                {message.sender === 'ai' && (
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Tooltip title="Copy">
                      <IconButton size="small">
                        <ContentCopy fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Helpful">
                      <IconButton size="small">
                        <ThumbUp fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Not helpful">
                      <IconButton size="small">
                        <ThumbDown fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <IconButton 
                      size="small"
                      onClick={(e) => {
                        setMenuAnchorEl(e.currentTarget);
                        setSelectedMessage(message.id);
                      }}
                    >
                      <MoreVert fontSize="small" />
                    </IconButton>
                  </Box>
                )}
                
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  {message.timestamp.toLocaleTimeString()}
                </Typography>
              </MessageBubble>
            </Box>
          ))}
          
          {isTyping && (
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  backgroundColor: 'rgba(255, 215, 0, 0.2)',
                  mr: 1,
                }}
              >
                <AutoAwesome sx={{ fontSize: 18, color: '#FFD700' }} />
              </Avatar>
              <Box sx={{ ...utilityClasses.glassmorphism, p: 2, borderRadius: 2 }}>
                <CircularProgress size={16} sx={{ color: '#FFD700' }} />
                <Typography variant="caption" sx={{ ml: 1 }}>AI is thinking...</Typography>
              </Box>
            </Box>
          )}
          
          <div ref={messagesEndRef} />
        </CardContent>

        {/* Input Area */}
        <Divider />
        <Box sx={{ p: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
          <IconButton>
            <AttachFile />
          </IconButton>
          
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Ask about signals, market analysis, or create alerts..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 3,
              },
            }}
          />
          
          <IconButton onClick={handleVoiceInput} color={isListening ? 'primary' : 'default'}>
            <Mic />
          </IconButton>
          
          <IconButton onClick={handleSendMessage} color="primary">
            <Send />
          </IconButton>
        </Box>
      </ChatContainer>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={() => setMenuAnchorEl(null)}
      >
        <MenuItem onClick={() => setMenuAnchorEl(null)}>
          <Refresh sx={{ mr: 1 }} /> Regenerate response
        </MenuItem>
        <MenuItem onClick={() => setMenuAnchorEl(null)}>
          <Code sx={{ mr: 1 }} /> View as code
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default AIAssistant;
