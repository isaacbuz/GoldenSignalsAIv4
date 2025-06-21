import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Paper,
    Typography,
    TextField,
    IconButton,
    List,
    ListItem,
    ListItemAvatar,
    ListItemText,
    Avatar,
    Chip,
    Button,
    Divider,
    Badge,
    InputAdornment,
    Menu,
    MenuItem,
    Tooltip,
    Grid,
} from '@mui/material';
import {
    Send as SendIcon,
    AttachFile as AttachIcon,
    Mic as MicIcon,
    SmartToy as AIIcon,
    Person as PersonIcon,
    TrendingUp as BullishIcon,
    TrendingDown as BearishIcon,
    Warning as AlertIcon,
    CheckCircle as SuccessIcon,
    MoreVert as MoreIcon,
    Telegram as TelegramIcon,
    Twitter as TwitterIcon,
    Forum as DiscordIcon,
    Chat as WhatsAppIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface TradingCommunityChatProps {
    platform: 'all' | 'discord' | 'whatsapp' | 'twitter' | 'custom';
    aiActive: boolean;
}

interface Message {
    id: string;
    sender: string;
    senderType: 'ai' | 'user' | 'admin';
    platform: 'discord' | 'whatsapp' | 'twitter' | 'custom';
    content: string;
    timestamp: Date;
    type: 'text' | 'trade_signal' | 'analysis' | 'risk_alert';
    tradeData?: {
        symbol: string;
        direction: 'CALL' | 'PUT';
        entry: number;
        stop: number;
        targets: number[];
        confidence: number;
    };
    reactions?: { emoji: string; count: number }[];
}

const TradingCommunityChat: React.FC<TradingCommunityChatProps> = ({ platform, aiActive }) => {
    const theme = useTheme();
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);

    useEffect(() => {
        // Initialize with some messages
        const initialMessages: Message[] = [
            {
                id: '1',
                sender: 'Atlas AI',
                senderType: 'ai',
                platform: 'discord',
                content: '🌅 Good morning traders! Market is looking bullish today. SPY showing strength above 450.',
                timestamp: new Date(Date.now() - 3600000),
                type: 'text',
            },
            {
                id: '2',
                sender: 'Atlas AI',
                senderType: 'ai',
                platform: 'discord',
                content: '🚨 NEW TRADE SETUP 🚨\n\nNVDA CALL Option Alert!',
                timestamp: new Date(Date.now() - 1800000),
                type: 'trade_signal',
                tradeData: {
                    symbol: 'NVDA',
                    direction: 'CALL',
                    entry: 745.50,
                    stop: 740.00,
                    targets: [755, 760, 765],
                    confidence: 87,
                },
            },
            {
                id: '3',
                sender: 'TraderMike',
                senderType: 'user',
                platform: 'discord',
                content: 'Thanks Atlas! Already in with 2 contracts 🚀',
                timestamp: new Date(Date.now() - 1700000),
                type: 'text',
                reactions: [{ emoji: '👍', count: 5 }, { emoji: '🚀', count: 3 }],
            },
        ];
        setMessages(initialMessages);
    }, []);

    useEffect(() => {
        // Simulate AI messages
        if (!aiActive) return;

        const interval = setInterval(() => {
            generateAIMessage();
        }, 15000); // Every 15 seconds

        return () => clearInterval(interval);
    }, [aiActive]);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const generateAIMessage = () => {
        const aiMessages = [
            {
                content: '📊 Quick market update: Tech sector showing relative strength. Watch QQQ for breakout above 380.',
                type: 'analysis' as const,
            },
            {
                content: '⚠️ Risk Alert: VIX climbing. Consider reducing position sizes or tightening stops.',
                type: 'risk_alert' as const,
            },
            {
                content: '✅ NVDA target 1 hit! Consider taking partial profits here.',
                type: 'text' as const,
            },
        ];

        const randomMessage = aiMessages[Math.floor(Math.random() * aiMessages.length)];

        const newMessage: Message = {
            id: `msg-${Date.now()}`,
            sender: 'Atlas AI',
            senderType: 'ai',
            platform: platform === 'all' ? 'discord' : platform,
            content: randomMessage.content,
            timestamp: new Date(),
            type: randomMessage.type,
        };

        setMessages(prev => [...prev, newMessage]);
    };

    const handleSendMessage = () => {
        if (!inputMessage.trim()) return;

        const newMessage: Message = {
            id: `msg-${Date.now()}`,
            sender: 'You',
            senderType: 'user',
            platform: 'custom',
            content: inputMessage,
            timestamp: new Date(),
            type: 'text',
        };

        setMessages(prev => [...prev, newMessage]);
        setInputMessage('');

        // Simulate AI response
        if (inputMessage.toLowerCase().includes('analysis') || inputMessage.includes('?')) {
            setIsTyping(true);
            setTimeout(() => {
                const aiResponse: Message = {
                    id: `msg-${Date.now()}`,
                    sender: 'Atlas AI',
                    senderType: 'ai',
                    platform: 'custom',
                    content: `Great question! Based on my analysis, ${inputMessage.includes('AAPL') ? 'AAPL is showing a bullish flag pattern on the 15m chart. Watch for breakout above $185.' : 'the market is currently in a consolidation phase. Key levels to watch are SPY 450 and QQQ 380.'}`,
                    timestamp: new Date(),
                    type: 'analysis',
                };
                setMessages(prev => [...prev, aiResponse]);
                setIsTyping(false);
            }, 2000);
        }
    };

    const getPlatformIcon = (platform: string) => {
        switch (platform) {
            case 'discord':
                return <DiscordIcon sx={{ fontSize: 16 }} />;
            case 'whatsapp':
                return <WhatsAppIcon sx={{ fontSize: 16 }} />;
            case 'twitter':
                return <TwitterIcon sx={{ fontSize: 16 }} />;
            case 'telegram':
                return <TelegramIcon sx={{ fontSize: 16 }} />;
            default:
                return null;
        }
    };

    const renderTradeSignal = (message: Message) => {
        if (!message.tradeData) return null;

        const { symbol, direction, entry, stop, targets, confidence } = message.tradeData;

        return (
            <Paper
                sx={{
                    p: 2,
                    mt: 1,
                    backgroundColor: direction === 'CALL'
                        ? 'rgba(76, 175, 80, 0.1)'
                        : 'rgba(244, 67, 54, 0.1)',
                    border: `1px solid ${direction === 'CALL' ? theme.palette.success.main : theme.palette.error.main}`,
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {direction === 'CALL' ? <BullishIcon color="success" /> : <BearishIcon color="error" />}
                        <Typography variant="h6" fontWeight="bold">
                            {symbol} {direction}
                        </Typography>
                    </Box>
                    <Chip label={`${confidence}% confidence`} size="small" color="primary" />
                </Box>

                <Grid container spacing={2}>
                    <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Entry</Typography>
                        <Typography variant="body1" fontWeight="bold">${entry}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Stop Loss</Typography>
                        <Typography variant="body1" fontWeight="bold" color="error">${stop}</Typography>
                    </Grid>
                    <Grid item xs={12}>
                        <Typography variant="body2" color="text.secondary">Targets</Typography>
                        <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                            {targets.map((target, idx) => (
                                <Chip
                                    key={idx}
                                    label={`T${idx + 1}: $${target}`}
                                    size="small"
                                    color="success"
                                    variant="outlined"
                                />
                            ))}
                        </Box>
                    </Grid>
                </Grid>

                <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button size="small" variant="contained" color="success">
                        Copy Trade
                    </Button>
                    <Button size="small" variant="outlined">
                        View Chart
                    </Button>
                </Box>
            </Paper>
        );
    };

    const handleMenuClick = (event: React.MouseEvent<HTMLElement>, message: Message) => {
        setAnchorEl(event.currentTarget);
        setSelectedMessage(message);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedMessage(null);
    };

    return (
        <Paper sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h6">Trading Community</Typography>
                        <Badge badgeContent={messages.length} color="primary">
                            <ChatIcon />
                        </Badge>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                            icon={<AIIcon />}
                            label={aiActive ? 'AI Active' : 'AI Offline'}
                            size="small"
                            color={aiActive ? 'success' : 'default'}
                        />
                        <Typography variant="caption" color="text.secondary">
                            {platform === 'all' ? 'All Platforms' : platform}
                        </Typography>
                    </Box>
                </Box>
            </Box>

            {/* Messages */}
            <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                <List>
                    {messages.map((message, index) => (
                        <ListItem
                            key={message.id}
                            alignItems="flex-start"
                            sx={{
                                flexDirection: message.senderType === 'user' && message.sender === 'You' ? 'row-reverse' : 'row',
                                gap: 1,
                            }}
                        >
                            <ListItemAvatar>
                                <Avatar
                                    sx={{
                                        bgcolor: message.senderType === 'ai'
                                            ? theme.palette.primary.main
                                            : message.senderType === 'admin'
                                                ? theme.palette.warning.main
                                                : theme.palette.grey[500],
                                    }}
                                >
                                    {message.senderType === 'ai' ? <AIIcon /> : <PersonIcon />}
                                </Avatar>
                            </ListItemAvatar>

                            <Box
                                sx={{
                                    maxWidth: '70%',
                                    backgroundColor: message.senderType === 'user' && message.sender === 'You'
                                        ? theme.palette.primary.main
                                        : theme.palette.background.paper,
                                    color: message.senderType === 'user' && message.sender === 'You'
                                        ? 'white'
                                        : 'inherit',
                                    borderRadius: 2,
                                    p: 2,
                                    boxShadow: 1,
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography variant="subtitle2" fontWeight="bold">
                                            {message.sender}
                                        </Typography>
                                        {getPlatformIcon(message.platform)}
                                    </Box>
                                    <IconButton
                                        size="small"
                                        onClick={(e) => handleMenuClick(e, message)}
                                    >
                                        <MoreIcon fontSize="small" />
                                    </IconButton>
                                </Box>

                                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                                    {message.content}
                                </Typography>

                                {message.type === 'trade_signal' && renderTradeSignal(message)}

                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 1 }}>
                                    <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                        {new Date(message.timestamp).toLocaleTimeString()}
                                    </Typography>

                                    {message.reactions && (
                                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                                            {message.reactions.map((reaction, idx) => (
                                                <Chip
                                                    key={idx}
                                                    label={`${reaction.emoji} ${reaction.count}`}
                                                    size="small"
                                                    variant="outlined"
                                                    sx={{ height: 24 }}
                                                />
                                            ))}
                                        </Box>
                                    )}
                                </Box>
                            </Box>
                        </ListItem>
                    ))}

                    {isTyping && (
                        <ListItem>
                            <ListItemAvatar>
                                <Avatar sx={{ bgcolor: theme.palette.primary.main }}>
                                    <AIIcon />
                                </Avatar>
                            </ListItemAvatar>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body2" color="text.secondary">
                                    Atlas AI is typing
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 0.5 }}>
                                    <Box
                                        sx={{
                                            width: 8,
                                            height: 8,
                                            borderRadius: '50%',
                                            bgcolor: 'text.secondary',
                                            animation: 'bounce 1.4s infinite',
                                            animationDelay: '0s',
                                            '@keyframes bounce': {
                                                '0%, 60%, 100%': { transform: 'translateY(0)' },
                                                '30%': { transform: 'translateY(-10px)' },
                                            },
                                        }}
                                    />
                                    <Box
                                        sx={{
                                            width: 8,
                                            height: 8,
                                            borderRadius: '50%',
                                            bgcolor: 'text.secondary',
                                            animation: 'bounce 1.4s infinite',
                                            animationDelay: '0.2s',
                                        }}
                                    />
                                    <Box
                                        sx={{
                                            width: 8,
                                            height: 8,
                                            borderRadius: '50%',
                                            bgcolor: 'text.secondary',
                                            animation: 'bounce 1.4s infinite',
                                            animationDelay: '0.4s',
                                        }}
                                    />
                                </Box>
                            </Box>
                        </ListItem>
                    )}

                    <div ref={messagesEndRef} />
                </List>
            </Box>

            {/* Input */}
            <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                <TextField
                    fullWidth
                    placeholder="Ask Atlas AI or share with the community..."
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <IconButton size="small">
                                    <AttachIcon />
                                </IconButton>
                            </InputAdornment>
                        ),
                        endAdornment: (
                            <InputAdornment position="end">
                                <IconButton size="small">
                                    <MicIcon />
                                </IconButton>
                                <IconButton
                                    size="small"
                                    color="primary"
                                    onClick={handleSendMessage}
                                    disabled={!inputMessage.trim()}
                                >
                                    <SendIcon />
                                </IconButton>
                            </InputAdornment>
                        ),
                    }}
                />
            </Box>

            {/* Context Menu */}
            <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
            >
                <MenuItem onClick={handleMenuClose}>Copy Message</MenuItem>
                <MenuItem onClick={handleMenuClose}>Reply</MenuItem>
                <MenuItem onClick={handleMenuClose}>React</MenuItem>
                <Divider />
                <MenuItem onClick={handleMenuClose}>Report</MenuItem>
            </Menu>
        </Paper>
    );
};

export default TradingCommunityChat; 