/**
 * Golden Eye AI Prophet Chat
 * 
 * Simplified mystical chat interface with the AI Prophet.
 * Preserves the golden theme while being stable and performant.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
    Box,
    TextField,
    IconButton,
    Typography,
    Paper,
    Avatar,
    Stack,
    Chip,
    Fade,
    alpha,
} from '@mui/material';
import {
    Send as SendIcon,
    Psychology as PsychologyIcon,
    Close as CloseIcon,
    AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import {
    goldenEyeColors,
    goldenEyeAnimations,
    goldenEyeGradients,
    getGlassmorphismStyle
} from '../../theme/goldenEye';
import { useAppStore, useAIChat } from '../../store/appStore';

// === STYLED COMPONENTS ===
const ChatContainer = styled(Box)({
    position: 'fixed',
    bottom: 100,
    right: 24,
    width: 400,
    height: 500,
    zIndex: 1300,
    animation: `${goldenEyeAnimations.fadeInUp} 0.3s ease-out`,
});

const ChatPaper = styled(Paper)(({ theme }) => ({
    ...getGlassmorphismStyle(0.2),
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    borderRadius: 16,
    overflow: 'hidden',
    border: `2px solid ${alpha(goldenEyeColors.primary, 0.3)}`,
}));

const ChatHeader = styled(Box)({
    background: goldenEyeGradients.primary,
    padding: '12px 16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
});

const MessagesContainer = styled(Box)({
    flex: 1,
    overflowY: 'auto',
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',

    // Custom scrollbar
    '&::-webkit-scrollbar': {
        width: '6px',
    },
    '&::-webkit-scrollbar-track': {
        background: alpha(goldenEyeColors.surface, 0.3),
    },
    '&::-webkit-scrollbar-thumb': {
        background: alpha(goldenEyeColors.primary, 0.5),
        borderRadius: '3px',
    },
});

const MessageBubble = styled(Box)<{ isUser?: boolean }>(({ isUser }) => ({
    padding: '12px 16px',
    borderRadius: '16px',
    maxWidth: '80%',
    alignSelf: isUser ? 'flex-end' : 'flex-start',
    background: isUser
        ? goldenEyeGradients.primary
        : alpha(goldenEyeColors.surface, 0.8),
    color: isUser
        ? goldenEyeColors.background
        : goldenEyeColors.textPrimary,
    border: `1px solid ${alpha(goldenEyeColors.primary, 0.2)}`,
    wordBreak: 'break-word',
}));

const InputContainer = styled(Box)({
    padding: '16px',
    borderTop: `1px solid ${alpha(goldenEyeColors.primary, 0.2)}`,
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-end',
});

const StyledTextField = styled(TextField)(({ theme }) => ({
    flex: 1,
    '& .MuiOutlinedInput-root': {
        background: alpha(goldenEyeColors.surface, 0.6),
        borderRadius: '12px',
        '& fieldset': {
            borderColor: alpha(goldenEyeColors.primary, 0.3),
        },
        '&:hover fieldset': {
            borderColor: alpha(goldenEyeColors.primary, 0.5),
        },
        '&.Mui-focused fieldset': {
            borderColor: goldenEyeColors.primary,
        },
    },
    '& .MuiInputBase-input': {
        color: goldenEyeColors.textPrimary,
    },
}));

const TypingIndicator = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '12px 16px',
    maxWidth: '80%',
    alignSelf: 'flex-start',
});

const TypingDot = styled(Box)({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: goldenEyeColors.primary,
    animation: `${goldenEyeAnimations.orbPulse} 1.4s infinite ease-in-out`,
    '&:nth-of-type(1)': { animationDelay: '-0.32s' },
    '&:nth-of-type(2)': { animationDelay: '-0.16s' },
    '&:nth-of-type(3)': { animationDelay: '0s' },
});

// === COMPONENT ===
export const GoldenEyeChat: React.FC = () => {
    const [inputValue, setInputValue] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const { isOpen, messages, isTyping } = useAIChat();
    const { toggleAIChat, addAIMessage, setAITyping } = useAppStore();

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    // Handle sending messages
    const handleSendMessage = useCallback(async () => {
        if (!inputValue.trim() || isTyping) return;

        const userMessage = inputValue.trim();
        setInputValue('');

        // Add user message
        addAIMessage({
            type: 'user',
            content: userMessage,
        });

        // Simulate AI typing
        setAITyping(true);

        // Simulate AI response delay
        setTimeout(() => {
            const responses = [
                `🔮 The markets whisper secrets about ${userMessage}. I sense strong patterns forming...`,
                `⚡ Your question about ${userMessage} resonates with the market energies. The signals suggest opportunity ahead.`,
                `🌟 Ah, you seek wisdom about ${userMessage}. The golden patterns reveal interesting movements in the near future.`,
                `💫 The oracle sees potential in your inquiry about ${userMessage}. Watch for confluence signals at key levels.`,
                `🎯 Your question touches the essence of market dynamics. ${userMessage} shows promising technical setups.`,
            ];

            const randomResponse = responses[Math.floor(Math.random() * responses.length)];

            addAIMessage({
                type: 'ai',
                content: randomResponse,
            });
        }, 1500 + Math.random() * 1000); // 1.5-2.5 second delay
    }, [inputValue, isTyping, addAIMessage, setAITyping]);

    // Handle Enter key
    const handleKeyPress = useCallback((event: React.KeyboardEvent) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    }, [handleSendMessage]);

    if (!isOpen) return null;

    return (
        <ChatContainer>
            <ChatPaper elevation={8}>
                {/* Header */}
                <ChatHeader>
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Avatar sx={{
                            background: goldenEyeGradients.mystical,
                            width: 32,
                            height: 32,
                        }}>
                            <PsychologyIcon fontSize="small" />
                        </Avatar>
                        <Box>
                            <Typography
                                variant="subtitle2"
                                sx={{
                                    fontWeight: 700,
                                    color: goldenEyeColors.background,
                                }}
                            >
                                Golden Eye AI Prophet
                            </Typography>
                            <Typography
                                variant="caption"
                                sx={{
                                    color: alpha(goldenEyeColors.background, 0.8),
                                }}
                            >
                                Market Oracle
                            </Typography>
                        </Box>
                    </Stack>

                    <Stack direction="row" spacing={1}>
                        <Chip
                            label="ONLINE"
                            size="small"
                            sx={{
                                background: alpha(goldenEyeColors.bullish, 0.2),
                                color: goldenEyeColors.bullish,
                                fontWeight: 600,
                                fontSize: '10px',
                            }}
                        />
                        <IconButton
                            size="small"
                            onClick={toggleAIChat}
                            sx={{ color: goldenEyeColors.background }}
                        >
                            <CloseIcon fontSize="small" />
                        </IconButton>
                    </Stack>
                </ChatHeader>

                {/* Messages */}
                <MessagesContainer>
                    {messages.map((message) => (
                        <Fade key={message.id} in timeout={300}>
                            <MessageBubble isUser={message.type === 'user'}>
                                <Typography variant="body2">
                                    {message.content}
                                </Typography>
                                <Typography
                                    variant="caption"
                                    sx={{
                                        opacity: 0.7,
                                        display: 'block',
                                        mt: 0.5,
                                        fontSize: '10px',
                                    }}
                                >
                                    {new Date(message.timestamp).toLocaleTimeString()}
                                </Typography>
                            </MessageBubble>
                        </Fade>
                    ))}

                    {/* Typing indicator */}
                    {isTyping && (
                        <Fade in timeout={300}>
                            <TypingIndicator>
                                <Avatar sx={{
                                    background: goldenEyeGradients.mystical,
                                    width: 24,
                                    height: 24,
                                }}>
                                    <AutoAwesomeIcon sx={{ fontSize: 14 }} />
                                </Avatar>
                                <Box sx={{ display: 'flex', gap: '4px' }}>
                                    <TypingDot />
                                    <TypingDot />
                                    <TypingDot />
                                </Box>
                            </TypingIndicator>
                        </Fade>
                    )}

                    <div ref={messagesEndRef} />
                </MessagesContainer>

                {/* Input */}
                <InputContainer>
                    <StyledTextField
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask the Oracle anything..."
                        multiline
                        maxRows={3}
                        size="small"
                        disabled={isTyping}
                    />
                    <IconButton
                        onClick={handleSendMessage}
                        disabled={!inputValue.trim() || isTyping}
                        sx={{
                            background: goldenEyeGradients.primary,
                            color: goldenEyeColors.background,
                            '&:hover': {
                                background: goldenEyeGradients.primary,
                                transform: 'scale(1.05)',
                            },
                            '&:disabled': {
                                background: alpha(goldenEyeColors.neutral, 0.3),
                                color: alpha(goldenEyeColors.textSecondary, 0.5),
                            },
                        }}
                    >
                        <SendIcon fontSize="small" />
                    </IconButton>
                </InputContainer>
            </ChatPaper>
        </ChatContainer>
    );
};

export default GoldenEyeChat; 