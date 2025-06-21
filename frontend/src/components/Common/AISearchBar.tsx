import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
    Box,
    Paper,
    InputBase,
    IconButton,
    Tooltip,
    Chip,
    Stack,
    CircularProgress,
    Fade,
    Popper,
    ClickAwayListener,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Typography,
    alpha,
    useTheme,
    Divider,
    Badge,
} from '@mui/material';
import {
    Search as SearchIcon,
    Mic as MicIcon,
    MicOff as MicOffIcon,
    AttachFile as AttachFileIcon,
    Image as ImageIcon,
    Send as SendIcon,
    Close as CloseIcon,
    TrendingUp,
    Analytics,
    Chat,
    ShowChart as PatternIcon,
    TrendingUp as StrategyIcon,
    AutoAwesome,
    Description,
    PhotoCamera,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

interface AISearchBarProps {
    onSearch?: (query: string, attachments?: File[]) => void;
    onSymbolChange?: (symbol: string) => void;
    onAnalyze?: () => void;
    onOpenChat?: (query: string) => void;
    currentSymbol?: string;
    onAddToFavorites?: (symbol: string) => void;
}

interface SearchSuggestion {
    type: 'symbol' | 'analysis' | 'question' | 'pattern' | 'strategy';
    text: string;
    description?: string;
    icon: React.ReactNode;
    action: () => void;
}

export const AISearchBar: React.FC<AISearchBarProps> = ({
    onSearch,
    onSymbolChange,
    onAnalyze,
    onOpenChat,
    currentSymbol = 'SPY',
}) => {
    const theme = useTheme();
    const navigate = useNavigate();
    const [query, setQuery] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [attachments, setAttachments] = useState<File[]>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);

    const inputRef = useRef<HTMLInputElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const imageInputRef = useRef<HTMLInputElement>(null);
    const anchorEl = useRef<HTMLDivElement>(null);
    const recognitionRef = useRef<any>(null);

    // Popular stock symbols with better organization
    const popularSymbols = [
        'SPY', 'QQQ', 'DIA', 'IWM', // ETFs first
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', // Tech giants
        'BRK.B', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', // Blue chips
        'AMD', 'NFLX', 'PYPL', 'ADBE', 'CRM', 'INTC', 'ORCL', 'CSCO' // More tech
    ];

    // Initialize speech recognition
    useEffect(() => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = false;
            recognitionRef.current.interimResults = true;
            recognitionRef.current.lang = 'en-US';

            recognitionRef.current.onresult = (event: any) => {
                const transcript = Array.from(event.results)
                    .map((result: any) => result[0])
                    .map((result: any) => result.transcript)
                    .join('');

                setQuery(transcript);
            };

            recognitionRef.current.onerror = (event: any) => {
                console.error('Speech recognition error', event.error);
                setIsListening(false);
            };

            recognitionRef.current.onend = () => {
                setIsListening(false);
            };
        }
    }, []);

    // Generate suggestions based on query
    useEffect(() => {
        if (!query.trim()) {
            setSuggestions([]);
            return;
        }

        const q = query.toLowerCase();
        const newSuggestions: SearchSuggestion[] = [];

        // Check for matching stock symbols (show multiple matches)
        const matchedSymbols = popularSymbols.filter(s =>
            s.toLowerCase().includes(q) ||
            (q.length === 1 && s.toLowerCase().startsWith(q))
        ).slice(0, 5);

        matchedSymbols.forEach(symbol => {
            newSuggestions.push({
                type: 'symbol',
                text: symbol,
                description: symbol === currentSymbol ? 'Current symbol' : 'View chart • Press ⌘+Click to favorite',
                icon: <TrendingUp />,
                action: () => {
                    onSymbolChange?.(symbol);
                    setQuery('');
                    setShowSuggestions(false);
                },
            });
        });

        // Analysis suggestions
        if (q.includes('analyze') || q.includes('analysis')) {
            newSuggestions.push({
                type: 'analysis',
                text: 'Run AI Analysis',
                description: 'Analyze current chart with AI',
                icon: <Analytics />,
                action: () => {
                    onAnalyze?.();
                    setQuery('');
                    setShowSuggestions(false);
                },
            });
        }

        // Pattern detection
        if (q.includes('pattern') || q.includes('find')) {
            newSuggestions.push({
                type: 'pattern',
                text: 'Detect Chart Patterns',
                description: 'Find technical patterns',
                icon: <PatternIcon />,
                action: () => {
                    onAnalyze?.();
                    setQuery('');
                    setShowSuggestions(false);
                },
            });
        }

        // General questions
        if (q.includes('what') || q.includes('how') || q.includes('why') || q.includes('when')) {
            newSuggestions.push({
                type: 'question',
                text: 'Ask AI Assistant',
                description: query,
                icon: <Chat />,
                action: () => {
                    onOpenChat?.(query);
                    setQuery('');
                    setShowSuggestions(false);
                },
            });
        }

        setSuggestions(newSuggestions);
    }, [query, onSymbolChange, onAnalyze, onOpenChat]);

    const handleVoiceInput = () => {
        if (!recognitionRef.current) {
            alert('Speech recognition is not supported in your browser');
            return;
        }

        if (isListening) {
            recognitionRef.current.stop();
        } else {
            recognitionRef.current.start();
            setIsListening(true);
        }
    };

    const handleFileAttachment = (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(event.target.files || []);
        setAttachments(prev => [...prev, ...files]);
    };

    const handleImageAttachment = (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(event.target.files || []);
        setAttachments(prev => [...prev, ...files]);
    };

    const removeAttachment = (index: number) => {
        setAttachments(prev => prev.filter((_, i) => i !== index));
    };

    const handleSubmit = async () => {
        if (!query.trim() && attachments.length === 0) return;

        setIsProcessing(true);

        try {
            // Determine intent
            const q = query.toLowerCase();

            // Check if it's a stock symbol
            const isSymbol = popularSymbols.some(s => s.toLowerCase() === q);
            if (isSymbol) {
                onSymbolChange?.(query.toUpperCase());
            }
            // Check if it's an analysis request
            else if (q.includes('analyze') || q.includes('analysis') || q.includes('pattern')) {
                onAnalyze?.();
            }
            // Otherwise, open chat
            else {
                onOpenChat?.(query);
            }

            // Clear after submit
            setQuery('');
            setAttachments([]);
            setShowSuggestions(false);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <Box
            ref={anchorEl}
            sx={{
                position: 'relative',
                width: '100%',
                maxWidth: 600,
                mx: 'auto',
            }}
        >
            <Paper
                elevation={0}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 2.5,
                    py: 1.25,
                    borderRadius: 4,
                    border: `2px solid ${alpha(theme.palette.primary.main, 0.15)}`,
                    backgroundColor: alpha(theme.palette.background.paper, 0.95),
                    backdropFilter: 'blur(20px)',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    position: 'relative',
                    overflow: 'hidden',
                    '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, transparent 100%)`,
                        opacity: 0,
                        transition: 'opacity 0.3s',
                    },
                    '&:hover': {
                        borderColor: alpha(theme.palette.primary.main, 0.3),
                        boxShadow: `0 4px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
                        transform: 'translateY(-1px)',
                        '&::before': {
                            opacity: 1,
                        },
                    },
                    '&:focus-within': {
                        borderColor: theme.palette.primary.main,
                        boxShadow: `0 4px 32px ${alpha(theme.palette.primary.main, 0.25)}`,
                        transform: 'translateY(-2px)',
                        '&::before': {
                            opacity: 1,
                        },
                    },
                }}
            >
                <AutoAwesome
                    sx={{
                        color: theme.palette.primary.main,
                        mr: 2,
                        fontSize: 24,
                        filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
                    }}
                />

                {/* Current Symbol Badge */}
                <Chip
                    label={currentSymbol}
                    size="medium"
                    sx={{
                        mr: 2,
                        backgroundColor: alpha(theme.palette.primary.main, 0.1),
                        color: theme.palette.primary.main,
                        fontWeight: 'bold',
                        fontSize: '0.875rem',
                        height: 28,
                        '& .MuiChip-label': {
                            px: 2,
                        },
                    }}
                />

                <InputBase
                    ref={inputRef}
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value);
                        setShowSuggestions(true);
                    }}
                    onFocus={() => setShowSuggestions(true)}
                    onKeyPress={handleKeyPress}
                    placeholder={`Search stocks or ask about ${currentSymbol}...`}
                    sx={{
                        flex: 1,
                        fontSize: '1rem',
                        fontWeight: 500,
                        '& input': {
                            padding: '6px 0',
                            '&::placeholder': {
                                opacity: 0.7,
                                fontWeight: 400,
                            },
                        },
                    }}
                />

                {/* Attachments */}
                {attachments.length > 0 && (
                    <Stack direction="row" spacing={0.5} sx={{ mx: 1 }}>
                        {attachments.map((file, index) => (
                            <Chip
                                key={index}
                                label={file.name.length > 15 ? `${file.name.substring(0, 15)}...` : file.name}
                                size="small"
                                icon={file.type.startsWith('image/') ? <ImageIcon /> : <Description />}
                                onDelete={() => removeAttachment(index)}
                                sx={{ maxWidth: 120 }}
                            />
                        ))}
                    </Stack>
                )}

                {/* Action Buttons */}
                <Stack direction="row" spacing={0.5} alignItems="center">
                    {/* Voice Input */}
                    <Tooltip title={isListening ? "Stop listening" : "Voice input"}>
                        <IconButton
                            size="small"
                            onClick={handleVoiceInput}
                            sx={{
                                color: isListening ? theme.palette.error.main : theme.palette.text.secondary,
                                animation: isListening ? 'pulse 1.5s infinite' : 'none',
                                '@keyframes pulse': {
                                    '0%': { opacity: 1 },
                                    '50%': { opacity: 0.5 },
                                    '100%': { opacity: 1 },
                                },
                            }}
                        >
                            {isListening ? <MicIcon /> : <MicOffIcon />}
                        </IconButton>
                    </Tooltip>

                    {/* Image Upload */}
                    <Tooltip title="Upload image">
                        <IconButton size="small" onClick={() => imageInputRef.current?.click()}>
                            <PhotoCamera />
                        </IconButton>
                    </Tooltip>
                    <input
                        ref={imageInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={handleImageAttachment}
                        multiple
                    />

                    {/* File Attachment */}
                    <Tooltip title="Attach file">
                        <IconButton size="small" onClick={() => fileInputRef.current?.click()}>
                            <AttachFileIcon />
                        </IconButton>
                    </Tooltip>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,.csv,.xlsx,.xls,.txt,.json"
                        style={{ display: 'none' }}
                        onChange={handleFileAttachment}
                        multiple
                    />

                    <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />

                    {/* Submit */}
                    <Tooltip title="Send">
                        <span>
                            <IconButton
                                size="small"
                                onClick={handleSubmit}
                                disabled={!query.trim() && attachments.length === 0}
                                sx={{
                                    color: theme.palette.primary.main,
                                    '&:disabled': {
                                        color: theme.palette.text.disabled,
                                    },
                                }}
                            >
                                {isProcessing ? <CircularProgress size={20} /> : <SendIcon />}
                            </IconButton>
                        </span>
                    </Tooltip>
                </Stack>
            </Paper>

            {/* Suggestions Dropdown */}
            <Popper
                open={showSuggestions && suggestions.length > 0}
                anchorEl={anchorEl.current}
                placement="bottom-start"
                style={{ width: anchorEl.current?.offsetWidth, zIndex: 1300 }}
            >
                <ClickAwayListener onClickAway={() => setShowSuggestions(false)}>
                    <Paper
                        elevation={8}
                        sx={{
                            mt: 1,
                            maxHeight: 300,
                            overflow: 'auto',
                            borderRadius: 2,
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                        }}
                    >
                        <List dense>
                            {suggestions.map((suggestion, index) => (
                                <ListItem
                                    key={index}
                                    button
                                    onClick={suggestion.action}
                                    sx={{
                                        '&:hover': {
                                            backgroundColor: alpha(theme.palette.primary.main, 0.08),
                                        },
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 40 }}>
                                        {suggestion.icon}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={suggestion.text}
                                        secondary={suggestion.description}
                                        primaryTypographyProps={{
                                            fontWeight: 500,
                                            fontSize: '0.9rem',
                                        }}
                                        secondaryTypographyProps={{
                                            fontSize: '0.8rem',
                                        }}
                                    />
                                </ListItem>
                            ))}
                        </List>
                    </Paper>
                </ClickAwayListener>
            </Popper>
        </Box>
    );
}; 