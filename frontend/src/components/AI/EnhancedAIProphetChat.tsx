/**
 * Enhanced AI Prophet Chat Interface
 * 
 * A mystical and powerful AI chat interface with file upload capabilities
 * Features:
 * - File upload for chart analysis
 * - Voice input capabilities
 * - Strategy generation
 * - Educational explanations
 * - Real-time market insights
 */

import React, { useState, useRef, useCallback } from 'react';
import {
    Box,
    TextField,
    Button,
    Typography,
    Paper,
    Avatar,
    Stack,
    Chip,
    IconButton,
    Divider,
    LinearProgress,
    Alert,
    Card,
    CardContent,
    useTheme,
    alpha,
    Tooltip,
} from '@mui/material';
import {
    Send as SendIcon,
    AttachFile as AttachFileIcon,
    Mic as MicIcon,
    MicOff as MicOffIcon,
    Psychology as PsychologyIcon,
    AutoAwesome as AutoAwesomeIcon,
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Timeline as TimelineIcon,
    Lightbulb as LightbulbIcon,
    Warning as WarningIcon,
    CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
    id: string;
    type: 'user' | 'ai';
    content: string;
    timestamp: Date;
    confidence?: number;
    signal?: {
        action: 'BUY' | 'SELL' | 'HOLD';
        symbol: string;
        target: number;
        stopLoss: number;
        confidence: number;
    };
    file?: {
        name: string;
        type: string;
        url: string;
    };
}

interface EnhancedAIProphetChatProps {
    onSignalGenerated?: (signal: any) => void;
}

const SAMPLE_RESPONSES = [
    {
        content: "I sense strong bullish momentum in SPY. The golden cross formation suggests a breakout above $450 resistance.",
        confidence: 94,
        signal: {
            action: 'BUY' as const,
            symbol: 'SPY',
            target: 455.00,
            stopLoss: 442.00,
            confidence: 94
        }
    },
    {
        content: "The chart pattern reveals a descending triangle. I recommend caution and suggest waiting for a clear breakout.",
        confidence: 78,
        signal: {
            action: 'HOLD' as const,
            symbol: 'AAPL',
            target: 195.00,
            stopLoss: 185.00,
            confidence: 78
        }
    }
];

export const EnhancedAIProphetChat: React.FC<EnhancedAIProphetChatProps> = ({ onSignalGenerated }) => {
    const theme = useTheme();
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            type: 'ai',
            content: "🔮 Welcome, seeker of market wisdom. I am the Golden Eye AI Prophet. Upload a chart, ask about any symbol, or seek guidance on your trading journey. The markets hold many secrets...",
            timestamp: new Date(),
        }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleSendMessage = useCallback(async () => {
        if (!inputValue.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            type: 'user',
            content: inputValue,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);

        // Simulate AI processing
        setTimeout(() => {
            const randomResponse = SAMPLE_RESPONSES[Math.floor(Math.random() * SAMPLE_RESPONSES.length)];
            const aiMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: 'ai',
                content: randomResponse.content,
                timestamp: new Date(),
                confidence: randomResponse.confidence,
                signal: randomResponse.signal,
            };

            setMessages(prev => [...prev, aiMessage]);
            setIsLoading(false);

            if (randomResponse.signal && onSignalGenerated) {
                onSignalGenerated(randomResponse.signal);
            }
        }, 2000);
    }, [inputValue, isLoading, onSignalGenerated]);

    const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const fileMessage: Message = {
            id: Date.now().toString(),
            type: 'user',
            content: `Uploaded ${file.name} for analysis`,
            timestamp: new Date(),
            file: {
                name: file.name,
                type: file.type,
                url: URL.createObjectURL(file),
            },
        };

        setMessages(prev => [...prev, fileMessage]);
        setIsLoading(true);

        // Simulate AI chart analysis
        setTimeout(() => {
            const aiMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: 'ai',
                content: `📊 I've analyzed your chart. I see a clear ascending triangle pattern with volume confirmation. This suggests a bullish breakout is imminent. Entry recommended at current levels with a target of +8% and stop loss at -3%.`,
                timestamp: new Date(),
                confidence: 91,
                signal: {
                    action: 'BUY',
                    symbol: 'CHART_ANALYSIS',
                    target: 108,
                    stopLoss: 97,
                    confidence: 91
                }
            };

            setMessages(prev => [...prev, aiMessage]);
            setIsLoading(false);
        }, 3000);
    }, []);

    const toggleVoiceInput = useCallback(() => {
        setIsListening(!isListening);
        // In a real implementation, you'd integrate with Web Speech API
    }, [isListening]);

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Chat Header */}
            <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                <Stack direction="row" alignItems="center" spacing={2}>
                    <Avatar sx={{
                        bgcolor: 'primary.main',
                        background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
                    }}>
                        <PsychologyIcon />
                    </Avatar>
                    <Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                            Golden Eye AI Prophet
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            Market Oracle & Signal Generator
                        </Typography>
                    </Box>
                    <Box sx={{ flex: 1 }} />
                    <Chip
                        label="ONLINE"
                        color="success"
                        size="small"
                        sx={{ fontWeight: 600 }}
                    />
                </Stack>
            </Box>

            {/* Chat Messages */}
            <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
                <Stack spacing={2}>
                    <AnimatePresence>
                        {messages.map((message) => (
                            <motion.div
                                key={message.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <Box sx={{
                                    display: 'flex',
                                    justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
                                    mb: 1,
                                }}>
                                    <Paper
                                        elevation={1}
                                        sx={{
                                            p: 2,
                                            maxWidth: '80%',
                                            bgcolor: message.type === 'user'
                                                ? 'primary.main'
                                                : 'background.paper',
                                            color: message.type === 'user'
                                                ? 'primary.contrastText'
                                                : 'text.primary',
                                            borderRadius: 2,
                                            border: message.type === 'ai'
                                                ? `1px solid ${alpha('#FFD700', 0.3)}`
                                                : 'none',
                                        }}
                                    >
                                        {/* File attachment */}
                                        {message.file && (
                                            <Card sx={{ mb: 1, maxWidth: 200 }}>
                                                <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                                                    <Stack direction="row" alignItems="center" spacing={1}>
                                                        <AttachFileIcon fontSize="small" />
                                                        <Typography variant="caption" noWrap>
                                                            {message.file.name}
                                                        </Typography>
                                                    </Stack>
                                                </CardContent>
                                            </Card>
                                        )}

                                        <Typography variant="body2">
                                            {message.content}
                                        </Typography>

                                        {/* AI Signal Card */}
                                        {message.signal && (
                                            <Card sx={{ mt: 2, bgcolor: 'action.hover' }}>
                                                <CardContent sx={{ p: 2 }}>
                                                    <Stack spacing={1}>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                                                Signal Generated
                                                            </Typography>
                                                            <Chip
                                                                label={message.signal.action}
                                                                color={
                                                                    message.signal.action === 'BUY' ? 'success' :
                                                                        message.signal.action === 'SELL' ? 'error' : 'warning'
                                                                }
                                                                size="small"
                                                            />
                                                        </Box>

                                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <Typography variant="caption">Symbol:</Typography>
                                                            <Typography variant="caption" sx={{ fontWeight: 500 }}>
                                                                {message.signal.symbol}
                                                            </Typography>
                                                        </Box>

                                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <Typography variant="caption">Target:</Typography>
                                                            <Typography variant="caption" sx={{ fontWeight: 500 }}>
                                                                {message.signal.target}
                                                            </Typography>
                                                        </Box>

                                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <Typography variant="caption">Stop Loss:</Typography>
                                                            <Typography variant="caption" sx={{ fontWeight: 500 }}>
                                                                {message.signal.stopLoss}
                                                            </Typography>
                                                        </Box>

                                                        <Box sx={{ mt: 1 }}>
                                                            <Typography variant="caption" color="text.secondary">
                                                                Confidence: {message.signal.confidence}%
                                                            </Typography>
                                                            <LinearProgress
                                                                variant="determinate"
                                                                value={message.signal.confidence}
                                                                sx={{
                                                                    mt: 0.5,
                                                                    height: 4,
                                                                    borderRadius: 2,
                                                                }}
                                                            />
                                                        </Box>
                                                    </Stack>
                                                </CardContent>
                                            </Card>
                                        )}

                                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                                            {message.timestamp.toLocaleTimeString()}
                                        </Typography>
                                    </Paper>
                                </Box>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {/* Loading indicator */}
                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                        >
                            <Box sx={{ display: 'flex', justifyContent: 'flex-start' }}>
                                <Paper elevation={1} sx={{ p: 2, bgcolor: 'background.paper' }}>
                                    <Stack direction="row" alignItems="center" spacing={1}>
                                        <AutoAwesomeIcon sx={{ color: 'primary.main', animation: 'pulse 2s infinite' }} />
                                        <Typography variant="body2" color="text.secondary">
                                            The Oracle is consulting the charts...
                                        </Typography>
                                    </Stack>
                                </Paper>
                            </Box>
                        </motion.div>
                    )}
                </Stack>
            </Box>

            {/* Quick Actions */}
            <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
                <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                    <Chip
                        label="📈 Analyze SPY"
                        variant="outlined"
                        size="small"
                        onClick={() => setInputValue("What's your analysis on SPY?")}
                        sx={{ cursor: 'pointer' }}
                    />
                    <Chip
                        label="🎯 Generate Strategy"
                        variant="outlined"
                        size="small"
                        onClick={() => setInputValue("Generate a trading strategy for tech stocks")}
                        sx={{ cursor: 'pointer' }}
                    />
                    <Chip
                        label="📊 Market Outlook"
                        variant="outlined"
                        size="small"
                        onClick={() => setInputValue("What's your market outlook for this week?")}
                        sx={{ cursor: 'pointer' }}
                    />
                </Stack>

                {/* Input Area */}
                <Stack direction="row" spacing={1} alignItems="flex-end">
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*,.pdf,.csv,.xlsx"
                        style={{ display: 'none' }}
                        onChange={handleFileUpload}
                    />

                    <Tooltip title="Upload Chart or Data">
                        <IconButton
                            onClick={() => fileInputRef.current?.click()}
                            color="primary"
                        >
                            <AttachFileIcon />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title={isListening ? "Stop Voice Input" : "Voice Input"}>
                        <IconButton
                            onClick={toggleVoiceInput}
                            color={isListening ? "error" : "default"}
                        >
                            {isListening ? <MicOffIcon /> : <MicIcon />}
                        </IconButton>
                    </Tooltip>

                    <TextField
                        fullWidth
                        multiline
                        maxRows={3}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder="Ask the Oracle anything about the markets..."
                        variant="outlined"
                        size="small"
                        onKeyPress={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSendMessage();
                            }
                        }}
                        disabled={isLoading}
                    />

                    <Button
                        variant="contained"
                        onClick={handleSendMessage}
                        disabled={!inputValue.trim() || isLoading}
                        sx={{
                            minWidth: 'auto',
                            px: 2,
                            background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
                        }}
                    >
                        <SendIcon />
                    </Button>
                </Stack>
            </Box>
        </Box>
    );
}; 