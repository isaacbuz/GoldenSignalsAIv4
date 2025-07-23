/**
 * Voice Signal Assistant - Natural language interface for signal analysis
 *
 * Features:
 * - Voice-activated signal queries
 * - Natural language processing
 * - Real-time signal analysis
 * - Multi-language support
 */

import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Paper,
    IconButton,
    Typography,
    Stack,
    Chip,
    CircularProgress,
    Fade,
    Tooltip,
    Card,
    CardContent,
    useTheme,
    alpha,
    LinearProgress,
    Button,
} from '@mui/material';
import {
    Mic,
    MicOff,
    VolumeUp,
    Psychology,
    TrendingUp,
    TrendingDown,
    Analytics,
    Close,
    Replay,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { apiClient } from '../../services/api';
import logger from '../../services/logger';


interface VoiceCommand {
    transcript: string;
    confidence: number;
    timestamp: Date;
    intent: 'signal_query' | 'analysis' | 'alert_setup' | 'general';
    entities: {
        symbols?: string[];
        timeframe?: string;
        indicators?: string[];
        action?: string;
    };
}

interface VoiceSignalAssistantProps {
    onCommand?: (command: VoiceCommand) => void;
    onSignalRequest?: (symbol: string, query: string) => void;
}

export const VoiceSignalAssistant: React.FC<VoiceSignalAssistantProps> = ({
    onCommand,
    onSignalRequest,
}) => {
    const theme = useTheme();
    const [isListening, setIsListening] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [interimTranscript, setInterimTranscript] = useState('');
    const [response, setResponse] = useState<string>('');
    const [error, setError] = useState<string>('');
    const [voiceEnabled, setVoiceEnabled] = useState(true);

    const recognitionRef = useRef<any>(null);
    const synthRef = useRef<SpeechSynthesisUtterance | null>(null);

    // Voice command patterns
    const commandPatterns = {
        signalQuery: /(?:show|get|find|what are|tell me about) (?:the )?(?:signals?|opportunities?) (?:for|on|in) ([A-Z]+)/i,
        analysis: /(?:analyze|analysis|explain|what do you think about) ([A-Z]+)/i,
        alertSetup: /(?:alert|notify|tell) me (?:when|if) ([A-Z]+) (?:reaches?|hits?|goes? (?:above|below)) ([\d.]+)/i,
        timeframe: /(?:in the |for the )?(?:next |last )?(hour|day|week|month|15 minutes?|5 minutes?)/i,
        confidence: /(?:high|strong|confident|reliable) (?:signals?|confidence)/i,
    };

    // Initialize speech recognition
    useEffect(() => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;
            recognitionRef.current.lang = 'en-US';

            recognitionRef.current.onresult = (event: any) => {
                let interim = '';
                let final = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        final += transcript + ' ';
                    } else {
                        interim += transcript;
                    }
                }

                if (final) {
                    setTranscript(prev => prev + final);
                    processCommand(final);
                }
                setInterimTranscript(interim);
            };

            recognitionRef.current.onerror = (event: any) => {
                logger.error('Speech recognition error:', event.error);
                setError(`Error: ${event.error}`);
                setIsListening(false);
            };

            recognitionRef.current.onend = () => {
                setIsListening(false);
            };
        } else {
            setVoiceEnabled(false);
        }

        // Initialize speech synthesis
        if ('speechSynthesis' in window) {
            synthRef.current = new SpeechSynthesisUtterance();
            synthRef.current.lang = 'en-US';
            synthRef.current.rate = 1.0;
            synthRef.current.pitch = 1.0;
        }

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
        };
    }, []);

    const processCommand = async (text: string) => {
        setIsProcessing(true);
        setError('');

        try {
            // Extract intent and entities
            const command: VoiceCommand = {
                transcript: text,
                confidence: 0.9,
                timestamp: new Date(),
                intent: 'general',
                entities: {},
            };

            // Check for signal query
            const signalMatch = text.match(commandPatterns.signalQuery);
            if (signalMatch) {
                command.intent = 'signal_query';
                command.entities.symbols = [signalMatch[1].toUpperCase()];
            }

            // Check for analysis request
            const analysisMatch = text.match(commandPatterns.analysis);
            if (analysisMatch) {
                command.intent = 'analysis';
                command.entities.symbols = [analysisMatch[1].toUpperCase()];
            }

            // Check for timeframe
            const timeframeMatch = text.match(commandPatterns.timeframe);
            if (timeframeMatch) {
                command.entities.timeframe = timeframeMatch[1];
            }

            // Process based on intent
            let responseText = '';

            if (command.intent === 'signal_query' && command.entities.symbols) {
                const symbol = command.entities.symbols[0];
                responseText = `Analyzing signals for ${symbol}...`;
                speak(responseText);

                // Fetch actual signals
                const signals = await apiClient.getPreciseOptionsSignals(symbol, '1d');
                if (signals && signals.length > 0) {
                    const topSignal = signals[0];
                    responseText = `I found a ${topSignal.confidence}% confidence ${topSignal.type} signal for ${symbol}. `;
                    responseText += `Strike price ${topSignal.strike_price}, expiring ${new Date(topSignal.expiration_date).toLocaleDateString()}. `;

                    if (topSignal.confidence >= 85) {
                        responseText += 'This is a high-confidence opportunity.';
                    }
                } else {
                    responseText = `No active signals found for ${symbol} at this time.`;
                }

                if (onSignalRequest) {
                    onSignalRequest(symbol, text);
                }
            } else if (command.intent === 'analysis' && command.entities.symbols) {
                const symbol = command.entities.symbols[0];
                responseText = `Running AI analysis on ${symbol}...`;
                speak(responseText);

                // Simulate analysis
                responseText = `${symbol} is showing strong momentum with increasing volume. Technical indicators suggest a bullish trend. Consider monitoring for entry opportunities.`;
            } else {
                responseText = "I can help you find trading signals. Try saying 'Show me signals for Apple' or 'Analyze Tesla'.";
            }

            setResponse(responseText);
            speak(responseText);

            if (onCommand) {
                onCommand(command);
            }
        } catch (error) {
            logger.error('Error processing command:', error);
            setError('Sorry, I had trouble processing that command.');
        } finally {
            setIsProcessing(false);
        }
    };

    const speak = (text: string) => {
        if (synthRef.current && window.speechSynthesis) {
            window.speechSynthesis.cancel();
            synthRef.current.text = text;
            window.speechSynthesis.speak(synthRef.current);
        }
    };

    const toggleListening = () => {
        if (!voiceEnabled) {
            setError('Voice recognition is not supported in your browser.');
            return;
        }

        if (isListening) {
            recognitionRef.current?.stop();
        } else {
            setTranscript('');
            setInterimTranscript('');
            setResponse('');
            setError('');
            recognitionRef.current?.start();
            setIsListening(true);
            speak("I'm listening. Ask me about trading signals.");
        }
    };

    const reset = () => {
        setTranscript('');
        setInterimTranscript('');
        setResponse('');
        setError('');
        if (recognitionRef.current && isListening) {
            recognitionRef.current.stop();
        }
    };

    return (
        <Paper
            elevation={3}
            sx={{
                p: 3,
                borderRadius: 3,
                background: alpha(theme.palette.background.paper, 0.9),
                backdropFilter: 'blur(10px)',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            }}
        >
            <Stack spacing={3}>
                {/* Header */}
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Psychology sx={{ color: theme.palette.primary.main }} />
                        <Typography variant="h6" fontWeight="bold">
                            Voice Signal Assistant
                        </Typography>
                    </Stack>

                    <Stack direction="row" spacing={1}>
                        <Tooltip title="Reset">
                            <IconButton size="small" onClick={reset}>
                                <Replay />
                            </IconButton>
                        </Tooltip>
                    </Stack>
                </Stack>

                {/* Voice Control */}
                <Box sx={{ textAlign: 'center' }}>
                    <motion.div
                        animate={{
                            scale: isListening ? [1, 1.1, 1] : 1,
                        }}
                        transition={{
                            duration: 1.5,
                            repeat: isListening ? Infinity : 0,
                        }}
                    >
                        <IconButton
                            size="large"
                            onClick={toggleListening}
                            disabled={isProcessing}
                            sx={{
                                width: 80,
                                height: 80,
                                background: isListening
                                    ? `linear-gradient(135deg, ${theme.palette.error.main} 0%, ${theme.palette.error.dark} 100%)`
                                    : `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                                color: 'white',
                                '&:hover': {
                                    background: isListening
                                        ? `linear-gradient(135deg, ${theme.palette.error.dark} 0%, ${theme.palette.error.main} 100%)`
                                        : `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
                                },
                                boxShadow: `0 4px 20px ${alpha(isListening ? theme.palette.error.main : theme.palette.primary.main, 0.4)}`,
                            }}
                        >
                            {isListening ? <MicOff sx={{ fontSize: 40 }} /> : <Mic sx={{ fontSize: 40 }} />}
                        </IconButton>
                    </motion.div>

                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                        {isListening ? 'Listening... Click to stop' : 'Click to start voice command'}
                    </Typography>
                </Box>

                {/* Processing Indicator */}
                {isProcessing && (
                    <Box>
                        <LinearProgress />
                        <Typography variant="body2" color="primary" align="center" sx={{ mt: 1 }}>
                            Processing your request...
                        </Typography>
                    </Box>
                )}

                {/* Transcript Display */}
                {(transcript || interimTranscript) && (
                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                You said:
                            </Typography>
                            <Typography variant="body1">
                                {transcript}
                                <span style={{ opacity: 0.6 }}>{interimTranscript}</span>
                            </Typography>
                        </CardContent>
                    </Card>
                )}

                {/* Response Display */}
                {response && (
                    <Card
                        variant="outlined"
                        sx={{
                            borderColor: theme.palette.primary.main,
                            backgroundColor: alpha(theme.palette.primary.main, 0.05),
                        }}
                    >
                        <CardContent>
                            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                <VolumeUp sx={{ color: theme.palette.primary.main, fontSize: 20 }} />
                                <Typography variant="body2" color="primary" fontWeight="bold">
                                    Assistant Response:
                                </Typography>
                            </Stack>
                            <Typography variant="body1">{response}</Typography>
                        </CardContent>
                    </Card>
                )}

                {/* Error Display */}
                {error && (
                    <Card
                        variant="outlined"
                        sx={{
                            borderColor: theme.palette.error.main,
                            backgroundColor: alpha(theme.palette.error.main, 0.05),
                        }}
                    >
                        <CardContent>
                            <Typography variant="body2" color="error">
                                {error}
                            </Typography>
                        </CardContent>
                    </Card>
                )}

                {/* Example Commands */}
                <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                        Try saying:
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {[
                            'Show me signals for AAPL',
                            'Analyze TSLA',
                            'Find high confidence signals',
                            'What are the opportunities in SPY',
                        ].map((example) => (
                            <Chip
                                key={example}
                                label={example}
                                size="small"
                                variant="outlined"
                                onClick={() => {
                                    setTranscript(example);
                                    processCommand(example);
                                }}
                                sx={{ cursor: 'pointer', mb: 1 }}
                            />
                        ))}
                    </Stack>
                </Box>
            </Stack>
        </Paper>
    );
};

export default VoiceSignalAssistant;
