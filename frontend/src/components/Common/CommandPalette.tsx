import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Dialog,
    DialogContent,
    TextField,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    ListItemSecondaryAction,
    Chip,
    Box,
    Typography,
    IconButton,
    InputAdornment,
    Divider,
    useTheme,
    alpha,
    Fade,
    Zoom,
} from '@mui/material';
import {
    Search,
    Close,
    TrendingUp,
    ShowChart,
    Analytics,
    AccountBalance,
    Psychology,
    Scanner,
    Timeline,
    Settings,
    Mic,
    Speed,
    AutoAwesome,
    KeyboardReturn,
    KeyboardCommandKey,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useHotkeys } from 'react-hotkeys-hook';

interface Command {
    id: string;
    title: string;
    description?: string;
    icon: React.ReactNode;
    category: 'navigation' | 'action' | 'ai' | 'analysis' | 'trading';
    shortcut?: string;
    action: () => void;
    keywords?: string[];
}

interface CommandPaletteProps {
    open?: boolean;
    onClose?: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ open: externalOpen, onClose: externalOnClose }) => {
    const theme = useTheme();
    const navigate = useNavigate();
    const [internalOpen, setInternalOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [isListening, setIsListening] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const recognitionRef = useRef<any>(null);

    // Use external control if provided, otherwise use internal state
    const open = externalOpen !== undefined ? externalOpen : internalOpen;
    const handleClose = () => {
        if (externalOnClose) {
            externalOnClose();
        } else {
            setInternalOpen(false);
        }
        setQuery('');
        setSelectedIndex(0);
    };

    // Global keyboard shortcut
    useHotkeys('cmd+k, ctrl+k', (e) => {
        e.preventDefault();
        if (externalOpen === undefined) {
            setInternalOpen(true);
        }
    }, { enableOnFormTags: true });

    // Define all available commands
    const commands: Command[] = [
        // Navigation
        {
            id: 'nav-dashboard',
            title: 'Go to Dashboard',
            description: 'View your trading command center',
            icon: <TrendingUp />,
            category: 'navigation',
            shortcut: '⌘D',
            action: () => {
                navigate('/dashboard');
                handleClose();
            },
            keywords: ['home', 'main', 'overview'],
        },
        {
            id: 'nav-trading',
            title: 'Live Trading',
            description: 'Open advanced trading interface',
            icon: <ShowChart />,
            category: 'navigation',
            shortcut: '⌘T',
            action: () => {
                navigate('/trading');
                handleClose();
            },
            keywords: ['trade', 'chart', 'buy', 'sell'],
        },
        {
            id: 'nav-analytics',
            title: 'Market Analytics',
            description: 'Analyze market trends and patterns',
            icon: <Analytics />,
            category: 'navigation',
            shortcut: '⌘A',
            action: () => {
                navigate('/analytics');
                handleClose();
            },
            keywords: ['analysis', 'market', 'trends'],
        },
        {
            id: 'nav-portfolio',
            title: 'Portfolio Lab',
            description: 'Manage and optimize your portfolio',
            icon: <AccountBalance />,
            category: 'navigation',
            shortcut: '⌘P',
            action: () => {
                navigate('/portfolio');
                handleClose();
            },
            keywords: ['holdings', 'positions', 'balance'],
        },

        // AI Actions
        {
            id: 'ai-chat',
            title: 'Open AI Assistant',
            description: 'Chat with your AI trading assistant',
            icon: <Psychology />,
            category: 'ai',
            shortcut: '⌘/',
            action: () => {
                // Trigger AI chat open
                window.dispatchEvent(new CustomEvent('openAIChat'));
                handleClose();
            },
            keywords: ['chat', 'ask', 'help', 'assistant'],
        },
        {
            id: 'ai-analyze',
            title: 'AI Chart Analysis',
            description: 'Upload a chart for AI pattern recognition',
            icon: <AutoAwesome />,
            category: 'ai',
            action: () => {
                window.dispatchEvent(new CustomEvent('openAIChat', {
                    detail: { message: 'I want to analyze a chart pattern' }
                }));
                handleClose();
            },
            keywords: ['pattern', 'recognition', 'scan'],
        },
        {
            id: 'ai-voice',
            title: 'Voice Command',
            description: 'Start voice command mode',
            icon: <Mic />,
            category: 'ai',
            shortcut: 'Space',
            action: () => {
                window.dispatchEvent(new CustomEvent('startVoiceCommand'));
                handleClose();
            },
            keywords: ['speak', 'voice', 'microphone'],
        },

        // Analysis Actions
        {
            id: 'scan-patterns',
            title: 'Scan Chart Patterns',
            description: 'Find patterns across multiple stocks',
            icon: <Scanner />,
            category: 'analysis',
            action: () => {
                navigate('/scanner');
                handleClose();
            },
            keywords: ['find', 'search', 'detect'],
        },
        {
            id: 'run-backtest',
            title: 'Run Backtest',
            description: 'Test your trading strategy',
            icon: <Timeline />,
            category: 'analysis',
            action: () => {
                navigate('/backtest');
                handleClose();
            },
            keywords: ['test', 'strategy', 'historical'],
        },

        // Trading Actions
        {
            id: 'quick-trade',
            title: 'Quick Trade',
            description: 'Open quick trade panel',
            icon: <Speed />,
            category: 'trading',
            shortcut: '⌘Q',
            action: () => {
                window.dispatchEvent(new CustomEvent('openQuickTrade'));
                handleClose();
            },
            keywords: ['buy', 'sell', 'order', 'fast'],
        },

        // Settings
        {
            id: 'settings',
            title: 'Settings',
            description: 'Configure your preferences',
            icon: <Settings />,
            category: 'navigation',
            shortcut: '⌘,',
            action: () => {
                navigate('/settings');
                handleClose();
            },
            keywords: ['preferences', 'config', 'setup'],
        },
    ];

    // Filter commands based on search
    const filteredCommands = commands.filter(cmd => {
        const searchLower = query.toLowerCase();
        return (
            cmd.title.toLowerCase().includes(searchLower) ||
            cmd.description?.toLowerCase().includes(searchLower) ||
            cmd.keywords?.some(k => k.toLowerCase().includes(searchLower))
        );
    });

    // Group commands by category
    const groupedCommands = filteredCommands.reduce((acc, cmd) => {
        if (!acc[cmd.category]) acc[cmd.category] = [];
        acc[cmd.category].push(cmd);
        return acc;
    }, {} as Record<string, Command[]>);

    // Category labels
    const categoryLabels = {
        navigation: 'Navigation',
        action: 'Actions',
        ai: 'AI Features',
        analysis: 'Analysis',
        trading: 'Trading',
    };

    // Handle keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!open) return;

            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    setSelectedIndex(prev =>
                        prev < filteredCommands.length - 1 ? prev + 1 : 0
                    );
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    setSelectedIndex(prev =>
                        prev > 0 ? prev - 1 : filteredCommands.length - 1
                    );
                    break;
                case 'Enter':
                    e.preventDefault();
                    if (filteredCommands[selectedIndex]) {
                        filteredCommands[selectedIndex].action();
                    }
                    break;
                case 'Escape':
                    e.preventDefault();
                    handleClose();
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [open, selectedIndex, filteredCommands, handleClose]);

    // Reset selection when search changes
    useEffect(() => {
        setSelectedIndex(0);
    }, [query]);

    // Focus input when opened
    useEffect(() => {
        if (open) {
            setTimeout(() => inputRef.current?.focus(), 100);
        }
    }, [open]);

    // Voice input setup
    useEffect(() => {
        if ('webkitSpeechRecognition' in window && open) {
            const recognition = new (window as any).webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onresult = (event: any) => {
                const transcript = Array.from(event.results)
                    .map((result: any) => result[0])
                    .map((result: any) => result.transcript)
                    .join('');

                setQuery(transcript);
            };

            recognition.onerror = () => {
                setIsListening(false);
            };

            recognition.onend = () => {
                setIsListening(false);
            };

            recognitionRef.current = recognition;
        }
    }, [open]);

    const toggleVoice = () => {
        if (!recognitionRef.current) return;

        if (isListening) {
            recognitionRef.current.stop();
        } else {
            recognitionRef.current.start();
            setIsListening(true);
        }
    };

    const getCategoryIcon = (category: string) => {
        switch (category) {
            case 'navigation': return '📍';
            case 'ai': return '🤖';
            case 'analysis': return '📊';
            case 'trading': return '💹';
            default: return '⚡';
        }
    };

    return (
        <Dialog
            open={open}
            onClose={handleClose}
            maxWidth="sm"
            fullWidth
            PaperProps={{
                sx: {
                    backgroundColor: alpha(theme.palette.background.paper, 0.95),
                    backdropFilter: 'blur(20px)',
                    borderRadius: 3,
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    overflow: 'hidden',
                },
            }}
        >
            <Box sx={{ p: 2, pb: 0 }}>
                <TextField
                    ref={inputRef}
                    fullWidth
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Type a command or search..."
                    variant="outlined"
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <Search sx={{ color: theme.palette.text.secondary }} />
                            </InputAdornment>
                        ),
                        endAdornment: (
                            <InputAdornment position="end">
                                <IconButton
                                    size="small"
                                    onClick={toggleVoice}
                                    color={isListening ? 'error' : 'default'}
                                    sx={{
                                        animation: isListening ? 'pulse 1.5s infinite' : 'none',
                                        '@keyframes pulse': {
                                            '0%': { transform: 'scale(1)' },
                                            '50%': { transform: 'scale(1.1)' },
                                            '100%': { transform: 'scale(1)' },
                                        },
                                    }}
                                >
                                    <Mic fontSize="small" />
                                </IconButton>
                                <IconButton size="small" onClick={handleClose}>
                                    <Close fontSize="small" />
                                </IconButton>
                            </InputAdornment>
                        ),
                        sx: {
                            borderRadius: 2,
                            backgroundColor: alpha(theme.palette.background.default, 0.5),
                            '& fieldset': { border: 'none' },
                        },
                    }}
                    autoComplete="off"
                />
            </Box>

            <DialogContent sx={{ p: 2, pt: 1 }}>
                <AnimatePresence mode="wait">
                    {Object.entries(groupedCommands).map(([category, cmds], categoryIndex) => (
                        <motion.div
                            key={category}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ delay: categoryIndex * 0.05 }}
                        >
                            <Typography
                                variant="caption"
                                sx={{
                                    color: theme.palette.text.secondary,
                                    fontWeight: 600,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 1,
                                    mt: categoryIndex > 0 ? 2 : 0,
                                    mb: 1,
                                }}
                            >
                                <span>{getCategoryIcon(category)}</span>
                                {categoryLabels[category as keyof typeof categoryLabels]}
                            </Typography>

                            <List dense sx={{ p: 0 }}>
                                {cmds.map((cmd, index) => {
                                    const globalIndex = filteredCommands.indexOf(cmd);
                                    const isSelected = globalIndex === selectedIndex;

                                    return (
                                        <ListItem
                                            key={cmd.id}
                                            button
                                            onClick={cmd.action}
                                            selected={isSelected}
                                            sx={{
                                                borderRadius: 2,
                                                mb: 0.5,
                                                backgroundColor: isSelected
                                                    ? alpha(theme.palette.primary.main, 0.1)
                                                    : 'transparent',
                                                '&:hover': {
                                                    backgroundColor: alpha(theme.palette.primary.main, 0.05),
                                                },
                                                transition: 'all 0.2s ease',
                                                transform: isSelected ? 'scale(1.02)' : 'scale(1)',
                                            }}
                                        >
                                            <ListItemIcon sx={{ minWidth: 40 }}>
                                                {cmd.icon}
                                            </ListItemIcon>
                                            <ListItemText
                                                primary={cmd.title}
                                                secondary={cmd.description}
                                                primaryTypographyProps={{
                                                    fontWeight: isSelected ? 600 : 500,
                                                }}
                                            />
                                            {cmd.shortcut && (
                                                <ListItemSecondaryAction>
                                                    <Chip
                                                        label={cmd.shortcut}
                                                        size="small"
                                                        sx={{
                                                            backgroundColor: alpha(theme.palette.background.default, 0.5),
                                                            fontFamily: 'monospace',
                                                            fontSize: '0.75rem',
                                                        }}
                                                    />
                                                </ListItemSecondaryAction>
                                            )}
                                        </ListItem>
                                    );
                                })}
                            </List>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {filteredCommands.length === 0 && (
                    <Box
                        sx={{
                            textAlign: 'center',
                            py: 4,
                            color: theme.palette.text.secondary,
                        }}
                    >
                        <Typography variant="body2">
                            No commands found for "{query}"
                        </Typography>
                        <Typography variant="caption" sx={{ mt: 1 }}>
                            Try searching for something else or use voice input
                        </Typography>
                    </Box>
                )}
            </DialogContent>

            <Box
                sx={{
                    p: 2,
                    pt: 0,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
            >
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                        <KeyboardCommandKey fontSize="small" sx={{ verticalAlign: 'middle' }} /> K
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        ↑↓ Navigate
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        <KeyboardReturn fontSize="small" sx={{ verticalAlign: 'middle' }} /> Select
                    </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                    ESC to close
                </Typography>
            </Box>
        </Dialog>
    );
};

// Global hook to open command palette
export const useCommandPalette = () => {
    const [open, setOpen] = useState(false);

    useHotkeys('cmd+k, ctrl+k', (e) => {
        e.preventDefault();
        setOpen(true);
    });

    return {
        open,
        setOpen,
        openCommandPalette: () => setOpen(true),
        closeCommandPalette: () => setOpen(false),
    };
}; 