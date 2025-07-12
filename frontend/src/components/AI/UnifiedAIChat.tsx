import React, { useState, useCallback, useMemo } from 'react';
import {
    Box,
    TextField,
    IconButton,
    Typography,
    Paper,
    Avatar,
    Chip,
    CircularProgress,
    Fade,
    useTheme,
} from '@mui/material';
import {
    Send as SendIcon,
    Psychology as PsychologyIcon,
    AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const ChatContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    maxHeight: 400,
    backgroundColor: theme.palette.background.paper,
    borderRadius: 8,
    border: `1px solid ${theme.palette.divider}`,
}));

const MessagesContainer = styled(Box)(({ theme }) => ({
    flex: 1,
    padding: theme.spacing(1),
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
}));

const MessageBubble = styled(Paper)<{ isUser?: boolean }>(({ theme, isUser }) => ({
    padding: theme.spacing(1, 2),
    maxWidth: '80%',
    alignSelf: isUser ? 'flex-end' : 'flex-start',
    backgroundColor: isUser ? theme.palette.primary.main : theme.palette.background.default,
    color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
    borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
}));

const InputContainer = styled(Box)(({ theme }) => ({
    padding: theme.spacing(1),
    borderTop: `1px solid ${theme.palette.divider}`,
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
}));

interface Message {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
}

interface UnifiedAIChatProps {
    placeholder?: string;
    onMessage?: (message: string) => void;
}

export const UnifiedAIChat: React.FC<UnifiedAIChatProps> = ({
    placeholder = "Ask Golden Eye AI Prophet...",
    onMessage,
}) => {
    const theme = useTheme();
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            content: "Hello! I'm your AI Prophet. Ask me about market insights, trading signals, or any financial questions.",
            isUser: false,
            timestamp: new Date(),
        },
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSendMessage = useCallback(async () => {
        if (!inputValue.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            content: inputValue,
            isUser: true,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);
        onMessage?.(inputValue);

        // Simulate AI response
        setTimeout(() => {
            const aiResponse: Message = {
                id: (Date.now() + 1).toString(),
                content: `I sense you're asking about "${inputValue}". The market spirits whisper of opportunities ahead. Based on current patterns, I recommend staying vigilant for emerging signals.`,
                isUser: false,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, aiResponse]);
            setIsLoading(false);
        }, 1500);
    }, [inputValue, isLoading, onMessage]);

    const handleKeyPress = useCallback((event: React.KeyboardEvent) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    }, [handleSendMessage]);

    return (
        <ChatContainer>
            <MessagesContainer>
                {messages.map((message) => (
                    <Fade key={message.id} in timeout={300}>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                            {!message.isUser && (
                                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                                    <PsychologyIcon sx={{ fontSize: 18 }} />
                                </Avatar>
                            )}
                            <MessageBubble isUser={message.isUser}>
                                <Typography variant="body2">
                                    {message.content}
                                </Typography>
                            </MessageBubble>
                            {message.isUser && (
                                <Avatar sx={{ width: 32, height: 32, bgcolor: 'secondary.main' }}>
                                    U
                                </Avatar>
                            )}
                        </Box>
                    </Fade>
                ))}
                {isLoading && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                            <PsychologyIcon sx={{ fontSize: 18 }} />
                        </Avatar>
                        <Paper sx={{ p: 2, borderRadius: '16px 16px 16px 4px' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <CircularProgress size={16} />
                                <Typography variant="body2" color="text.secondary">
                                    AI Prophet is thinking...
                                </Typography>
                            </Box>
                        </Paper>
                    </Box>
                )}
            </MessagesContainer>

            <InputContainer>
                <TextField
                    fullWidth
                    size="small"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={placeholder}
                    disabled={isLoading}
                    variant="outlined"
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 3,
                        },
                    }}
                />
                <IconButton
                    onClick={handleSendMessage}
                    disabled={!inputValue.trim() || isLoading}
                    color="primary"
                >
                    <SendIcon />
                </IconButton>
            </InputContainer>
        </ChatContainer>
    );
};

export default UnifiedAIChat; 