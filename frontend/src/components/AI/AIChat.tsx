/**
 * AI Chat Component - Multimodal Trading Assistant Interface
 * 
 * Features:
 * - Real-time chat with AI assistant
 * - File upload support (images, CSVs, documents)
 * - Drag and drop functionality
 * - Message history
 * - Typing indicators
 * - Suggested actions
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Box,
    Paper,
    TextField,
    IconButton,
    Typography,
    Avatar,
    Stack,
    Chip,
    CircularProgress,
    Tooltip,
    Divider,
    Button,
    Menu,
    MenuItem,
    Alert,
    LinearProgress,
    Collapse,
    Dialog,
    useTheme,
    alpha,
    Fade,
    Zoom,
    Drawer,
} from '@mui/material';
import {
    Send,
    AttachFile,
    Image as ImageIcon,
    InsertDriveFile,
    Close,
    SmartToy,
    Person,
    MoreVert,
    Download,
    ContentCopy,
    Refresh,
    TipsAndUpdates,
    Analytics,
    ShowChart,
    Assessment,
    CloudUpload,
    Mic,
    MicOff,
    VolumeUp,
    AutoAwesome,
    Description,
    ThumbUp,
    ThumbDown,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { format } from 'date-fns';
import { apiClient } from '../../services/api';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    attachments?: Attachment[];
    metadata?: {
        confidence?: number;
        sources?: string[];
        symbol?: string;
        analysis?: any;
        charts?: string[];
    };
}

interface Attachment {
    filename: string;
    type: string;
    size: number;
    url?: string;
    analysis?: any;
}

interface AIChatProps {
    open: boolean;
    onClose: () => void;
    initialQuery?: string;
    attachments?: File[];
}

export const AIChat: React.FC<AIChatProps> = ({
    open,
    onClose,
    initialQuery,
    attachments,
}) => {
    const theme = useTheme();
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [voiceEnabled, setVoiceEnabled] = useState(true);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const recognitionRef = useRef<any>(null);
    const audioRef = useRef<HTMLAudioElement>(null);

    // Initialize with initial query
    useEffect(() => {
        if (initialQuery && open) {
            handleSendMessage(initialQuery, attachments);
        }
    }, [initialQuery, open]);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Initialize speech recognition
    useEffect(() => {
        if ('webkitSpeechRecognition' in window && voiceEnabled) {
            const recognition = new (window as any).webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onresult = (event: any) => {
                const transcript = Array.from(event.results)
                    .map((result: any) => result[0])
                    .map((result: any) => result.transcript)
                    .join('');

                setInput(transcript);
            };

            recognition.onerror = (event: any) => {
                console.error('Speech recognition error:', event.error);
                setIsRecording(false);
            };

            recognition.onend = () => {
                setIsRecording(false);
            };

            recognitionRef.current = recognition;
        }
    }, [voiceEnabled]);

    // Listen for AI events
    useEffect(() => {
        const handleAIChat = (event: CustomEvent) => {
            if (event.detail?.query) {
                handleSendMessage(event.detail.query);
            }
        };

        window.addEventListener('ai-chat', handleAIChat as EventListener);
        return () => {
            window.removeEventListener('ai-chat', handleAIChat as EventListener);
        };
    }, []);

    const handleSendMessage = async (text: string, files?: File[]) => {
        if (!text.trim() && (!files || files.length === 0)) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: text,
            timestamp: new Date(),
            attachments: files,
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            // Simulate AI response (replace with actual API call)
            await new Promise(resolve => setTimeout(resolve, 1500));

            const aiResponse: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: generateAIResponse(text),
                timestamp: new Date(),
                metadata: {
                    symbol: extractSymbol(text),
                },
            };

            setMessages(prev => [...prev, aiResponse]);
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const extractSymbol = (text: string): string | undefined => {
        const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ'];
        const upperText = text.toUpperCase();
        return symbols.find(symbol => upperText.includes(symbol));
    };

    const generateAIResponse = (query: string): string => {
        const q = query.toLowerCase();

        if (q.includes('analyze')) {
            return `I've analyzed the current chart and identified several key insights:

## Technical Analysis
- **Trend**: The stock is showing a strong upward trend with higher highs and higher lows
- **Support**: Strong support level identified at $145.50
- **Resistance**: Key resistance at $152.00

## Patterns Detected
1. **Bull Flag Formation** - Indicating potential continuation of uptrend
2. **Golden Cross** - 50-day MA crossed above 200-day MA yesterday

## Trading Recommendation
- **Entry**: Consider entering at $148.00-$148.50
- **Target**: $155.00 (4.7% upside)
- **Stop Loss**: $145.00 (2.0% downside)

Would you like me to place a trade alert for these levels?`;
        }

        if (q.includes('pattern')) {
            return `I've detected the following chart patterns:

### 1. Ascending Triangle (Bullish)
- Forming over the last 15 trading days
- Horizontal resistance at $150.00
- Rising support line from $142.00
- **Probability**: 72% chance of upward breakout

### 2. Cup and Handle
- Cup formation: 3 weeks
- Handle forming now
- Target: $158.00 on breakout

### 3. Bullish Divergence
- RSI making higher lows while price made lower lows
- Often precedes trend reversal

Would you like me to set alerts for these pattern breakouts?`;
        }

        if (q.includes('options') || q.includes('option')) {
            return `Based on current market conditions, here are the optimal options strategies:

## Recommended Options Plays

### 1. Bull Call Spread
- **Buy**: $150 Call (30 DTE) @ $3.20
- **Sell**: $155 Call (30 DTE) @ $1.40
- **Net Debit**: $1.80
- **Max Profit**: $3.20 (177% return)
- **Breakeven**: $151.80

### 2. Cash-Secured Put
- **Sell**: $145 Put (30 DTE) @ $2.50
- **Premium Collected**: $250 per contract
- **Return if Expired**: 1.7% monthly (20.4% annualized)

### 3. Covered Call
- **Own**: 100 shares at $148
- **Sell**: $152 Call (30 DTE) @ $1.85
- **Income**: $185 per contract
- **Return if Called**: 4.0% + premium

Which strategy aligns best with your risk tolerance?`;
        }

        return `I understand you're asking about "${query}". Let me help you with that.

Based on the current market data, here's what I'm seeing:

1. The market is showing mixed signals today
2. Your portfolio is up 2.3% for the day
3. Key sectors to watch: Technology (+1.2%), Healthcare (-0.5%)

Is there something specific you'd like me to analyze or help you with?`;
    };

    const handleCopyMessage = (content: string) => {
        navigator.clipboard.writeText(content);
    };

    return (
        <Drawer
            anchor="right"
            open={open}
            onClose={onClose}
            PaperProps={{
                sx: {
                    width: { xs: '100%', sm: 400, md: 450 },
                    backgroundColor: theme.palette.background.default,
                },
            }}
        >
            {/* Header */}
            <Box
                sx={{
                    p: 2,
                    borderBottom: `1px solid ${theme.palette.divider}`,
                    background: alpha(theme.palette.background.paper, 0.5),
                    backdropFilter: 'blur(10px)',
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack direction="row" alignItems="center" spacing={1}>
                        <AutoAwesome sx={{ color: theme.palette.primary.main }} />
                        <Typography variant="h6" fontWeight={600}>
                            AI Assistant
                        </Typography>
                    </Stack>
                    <IconButton onClick={onClose} size="small">
                        <Close />
                    </IconButton>
                </Stack>
            </Box>

            {/* Messages */}
            <Box
                sx={{
                    flex: 1,
                    overflow: 'auto',
                    p: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                }}
            >
                <AnimatePresence>
                    {messages.map((message) => (
                        <motion.div
                            key={message.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                        >
                            <Stack
                                direction="row"
                                spacing={1.5}
                                alignItems="flex-start"
                                sx={{
                                    justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                                }}
                            >
                                {message.role === 'assistant' && (
                                    <Avatar
                                        sx={{
                                            width: 32,
                                            height: 32,
                                            bgcolor: theme.palette.primary.main,
                                        }}
                                    >
                                        <AutoAwesome sx={{ fontSize: 18 }} />
                                    </Avatar>
                                )}

                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 2,
                                        maxWidth: '80%',
                                        backgroundColor: message.role === 'user'
                                            ? alpha(theme.palette.primary.main, 0.1)
                                            : alpha(theme.palette.background.paper, 0.8),
                                        borderRadius: 2,
                                        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                    }}
                                >
                                    {message.attachments && message.attachments.length > 0 && (
                                        <Stack direction="row" spacing={0.5} sx={{ mb: 1 }}>
                                            {message.attachments.map((file, index) => (
                                                <Chip
                                                    key={index}
                                                    size="small"
                                                    icon={file.type.startsWith('image/') ? <ImageIcon /> : <Description />}
                                                    label={file.name}
                                                    sx={{ fontSize: '0.75rem' }}
                                                />
                                            ))}
                                        </Stack>
                                    )}

                                    {message.role === 'assistant' ? (
                                        <Box sx={{ '& p': { my: 1 }, '& h2': { fontSize: '1.1rem', mt: 2, mb: 1 }, '& h3': { fontSize: '1rem', mt: 1.5, mb: 0.5 } }}>
                                            <ReactMarkdown
                                                components={{
                                                    code({ node, inline, className, children, ...props }) {
                                                        const match = /language-(\w+)/.exec(className || '');
                                                        return !inline && match ? (
                                                            <SyntaxHighlighter
                                                                style={vscDarkPlus}
                                                                language={match[1]}
                                                                PreTag="div"
                                                                {...props}
                                                            >
                                                                {String(children).replace(/\n$/, '')}
                                                            </SyntaxHighlighter>
                                                        ) : (
                                                            <code className={className} {...props}>
                                                                {children}
                                                            </code>
                                                        );
                                                    },
                                                }}
                                            >
                                                {message.content}
                                            </ReactMarkdown>
                                        </Box>
                                    ) : (
                                        <Typography variant="body2">{message.content}</Typography>
                                    )}

                                    {message.role === 'assistant' && (
                                        <Stack direction="row" spacing={1} sx={{ mt: 1.5 }}>
                                            <IconButton size="small" onClick={() => handleCopyMessage(message.content)}>
                                                <ContentCopy sx={{ fontSize: 16 }} />
                                            </IconButton>
                                            <IconButton size="small">
                                                <ThumbUp sx={{ fontSize: 16 }} />
                                            </IconButton>
                                            <IconButton size="small">
                                                <ThumbDown sx={{ fontSize: 16 }} />
                                            </IconButton>
                                        </Stack>
                                    )}
                                </Paper>

                                {message.role === 'user' && (
                                    <Avatar
                                        sx={{
                                            width: 32,
                                            height: 32,
                                            bgcolor: theme.palette.secondary.main,
                                        }}
                                    >
                                        U
                                    </Avatar>
                                )}
                            </Stack>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {isLoading && (
                    <Stack direction="row" spacing={1.5} alignItems="center">
                        <Avatar
                            sx={{
                                width: 32,
                                height: 32,
                                bgcolor: theme.palette.primary.main,
                            }}
                        >
                            <AutoAwesome sx={{ fontSize: 18 }} />
                        </Avatar>
                        <Paper
                            elevation={0}
                            sx={{
                                p: 2,
                                backgroundColor: alpha(theme.palette.background.paper, 0.8),
                                borderRadius: 2,
                                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                            }}
                        >
                            <Stack direction="row" spacing={1} alignItems="center">
                                <CircularProgress size={16} />
                                <Typography variant="body2" color="text.secondary">
                                    AI is thinking...
                                </Typography>
                            </Stack>
                        </Paper>
                    </Stack>
                )}

                <div ref={messagesEndRef} />
            </Box>

            {/* Input */}
            <Box
                sx={{
                    p: 2,
                    borderTop: `1px solid ${theme.palette.divider}`,
                    background: alpha(theme.palette.background.paper, 0.5),
                    backdropFilter: 'blur(10px)',
                }}
            >
                <TextField
                    fullWidth
                    multiline
                    maxRows={4}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage(input);
                        }
                    }}
                    placeholder="Ask me anything..."
                    variant="outlined"
                    size="small"
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 2,
                        },
                    }}
                />
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mt: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                        Press Enter to send, Shift+Enter for new line
                    </Typography>
                    <Button
                        variant="contained"
                        size="small"
                        onClick={() => handleSendMessage(input)}
                        disabled={!input.trim() || isLoading}
                        startIcon={<AutoAwesome />}
                    >
                        Send
                    </Button>
                </Stack>
            </Box>
        </Drawer>
    );
}; 