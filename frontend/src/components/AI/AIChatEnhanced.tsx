import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Box,
    Paper,
    TextField,
    IconButton,
    Typography,
    List,
    ListItem,
    Avatar,
    Chip,
    Button,
    CircularProgress,
    Divider,
    Tabs,
    Tab,
    Grid,
    Card,
    CardContent,
    Tooltip,
    Fade,
    Zoom,
    Alert,
    Snackbar,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Switch,
    FormControlLabel,
    Slider,
    Badge,
    Drawer,
    AppBar,
    Toolbar,
    useTheme,
    useMediaQuery,
} from '@mui/material';
import {
    Send as SendIcon,
    AttachFile as AttachFileIcon,
    Mic as MicIcon,
    MicOff as MicOffIcon,
    Image as ImageIcon,
    InsertChart as ChartIcon,
    Assessment as AssessmentIcon,
    TrendingUp as TrendingUpIcon,
    AccountBalance as PortfolioIcon,
    Psychology as AIIcon,
    VolumeUp as VolumeUpIcon,
    VolumeOff as VolumeOffIcon,
    Fullscreen as FullscreenIcon,
    FullscreenExit as FullscreenExitIcon,
    Settings as SettingsIcon,
    Download as DownloadIcon,
    Share as ShareIcon,
    History as HistoryIcon,
    Clear as ClearIcon,
    PhotoCamera as CameraIcon,
    BarChart as BacktestIcon,
    ShowChart as PatternIcon,
    Timeline as TimelineIcon,
    Speed as SpeedIcon,
    Lightbulb as SuggestionIcon,
    Warning as WarningIcon,
    CheckCircle as SuccessIcon,
    Error as ErrorIcon,
    Info as InfoIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { format } from 'date-fns';
import { Line, Bar, Pie, Radar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    RadialLinearScale,
    Title,
    Tooltip as ChartTooltip,
    Legend,
    Filler,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    RadialLinearScale,
    Title,
    ChartTooltip,
    Legend,
    Filler
);

interface Message {
    id: string;
    type: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    attachments?: Attachment[];
    charts?: string[];
    visualizations?: Visualization[];
    tradingSignals?: TradingSignal[];
    confidence?: number;
    sources?: string[];
    audioUrl?: string;
    analysisType?: string[];
    detectedPatterns?: string[];
    riskMetrics?: RiskMetrics;
}

interface Attachment {
    id: string;
    name: string;
    type: string;
    size: number;
    preview?: string;
    analysis?: any;
}

interface Visualization {
    type: string;
    data: any;
}

interface TradingSignal {
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    reasoning: string;
    entryPrice?: number;
    targetPrice?: number;
    stopLoss?: number;
}

interface RiskMetrics {
    marketVolatility: string;
    correlationRisk: string;
    concentrationRisk: string;
    recommendations: string[];
}

interface PortfolioHolding {
    symbol: string;
    shares: number;
    value: number;
}

interface BacktestStrategy {
    shortWindow: number;
    longWindow: number;
    stopLoss?: number;
    takeProfit?: number;
}

interface ChatSettings {
    voiceEnabled: boolean;
    autoPlayResponses: boolean;
    streamingEnabled: boolean;
    chartTheme: 'light' | 'dark';
    language: string;
    speechRate: number;
}

const defaultSettings: ChatSettings = {
    voiceEnabled: false,
    autoPlayResponses: false,
    streamingEnabled: true,
    chartTheme: 'dark',
    language: 'en',
    speechRate: 1.0,
};

export const AIChatEnhanced: React.FC = () => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

    // State management
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [attachments, setAttachments] = useState<Attachment[]>([]);
    const [sessionId] = useState(() => `session-${Date.now()}`);
    const [activeTab, setActiveTab] = useState(0);
    const [isRecording, setIsRecording] = useState(false);
    const [settings, setSettings] = useState<ChatSettings>(defaultSettings);
    const [showSettings, setShowSettings] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
    const [showHistory, setShowHistory] = useState(false);
    const [notification, setNotification] = useState<{ message: string; severity: 'success' | 'error' | 'warning' | 'info' } | null>(null);

    // Refs
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const audioRef = useRef<HTMLAudioElement>(null);
    const recognitionRef = useRef<any>(null);
    const wsRef = useRef<WebSocket | null>(null);

    // WebSocket connection for streaming
    useEffect(() => {
        if (settings.streamingEnabled) {
            const ws = new WebSocket(`ws://localhost:8000/api/v1/ai-chat/ws/${sessionId}`);

            ws.onopen = () => {
                console.log('WebSocket connected');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleStreamingResponse(data);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                showNotification('Connection error. Falling back to regular mode.', 'warning');
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
            };

            wsRef.current = ws;

            return () => {
                ws.close();
            };
        }
    }, [sessionId, settings.streamingEnabled]);

    // Speech recognition setup
    useEffect(() => {
        if ('webkitSpeechRecognition' in window && settings.voiceEnabled) {
            const recognition = new (window as any).webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = settings.language;

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
                showNotification('Speech recognition error. Please try again.', 'error');
            };

            recognition.onend = () => {
                setIsRecording(false);
            };

            recognitionRef.current = recognition;
        }
    }, [settings.voiceEnabled, settings.language]);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Dropzone configuration
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop: handleFileDrop,
        accept: {
            'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
            'text/csv': ['.csv'],
            'application/pdf': ['.pdf'],
            'application/vnd.ms-excel': ['.xls', '.xlsx'],
            'audio/*': ['.mp3', '.wav', '.m4a'],
        },
        maxSize: 100 * 1024 * 1024, // 100MB
    });

    // Handlers
    async function handleFileDrop(acceptedFiles: File[]) {
        const newAttachments: Attachment[] = [];

        for (const file of acceptedFiles) {
            const attachment: Attachment = {
                id: `file-${Date.now()}-${Math.random()}`,
                name: file.name,
                type: file.type,
                size: file.size,
            };

            // Generate preview for images
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    attachment.preview = e.target?.result as string;
                    setAttachments((prev) => [...prev, attachment]);
                };
                reader.readAsDataURL(file);
            } else {
                newAttachments.push(attachment);
            }

            // Analyze file immediately
            if (file.type.startsWith('image/')) {
                await analyzeImage(file);
            } else if (file.type === 'text/csv' || file.type.includes('excel')) {
                await analyzeData(file);
            }
        }

        setAttachments((prev) => [...prev, ...newAttachments]);
    }

    async function analyzeImage(file: File) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            const response = await fetch('/api/v1/ai-chat/analyze/image', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const analysis = await response.json();
                showNotification('Image analyzed successfully!', 'success');

                // Add analysis results to the chat
                const systemMessage: Message = {
                    id: `msg-${Date.now()}`,
                    type: 'system',
                    content: `Chart Analysis Complete:\n- Patterns: ${analysis.detected_patterns.join(', ')}\n- Trend: ${analysis.trend_direction}\n- Confidence: ${(analysis.confidence_scores.pattern_detection * 100).toFixed(0)}%`,
                    timestamp: new Date(),
                    detectedPatterns: analysis.detected_patterns,
                };

                setMessages((prev) => [...prev, systemMessage]);
            }
        } catch (error) {
            console.error('Error analyzing image:', error);
            showNotification('Failed to analyze image', 'error');
        }
    }

    async function analyzeData(file: File) {
        // Similar implementation for CSV/Excel analysis
        showNotification('Data file uploaded. Analysis in progress...', 'info');
    }

    async function handleSend() {
        if (!input.trim() && attachments.length === 0) return;

        const userMessage: Message = {
            id: `msg-${Date.now()}`,
            type: 'user',
            content: input,
            timestamp: new Date(),
            attachments: [...attachments],
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setAttachments([]);
        setIsLoading(true);

        try {
            if (settings.streamingEnabled && wsRef.current?.readyState === WebSocket.OPEN) {
                // Send via WebSocket for streaming
                wsRef.current.send(JSON.stringify({
                    message: input,
                    attachments: attachments,
                    settings: settings,
                }));
            } else {
                // Regular HTTP request
                const response = await sendChatMessage(input, attachments);
                handleResponse(response);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            showNotification('Failed to send message', 'error');
        } finally {
            setIsLoading(false);
        }
    }

    async function sendChatMessage(message: string, attachments: Attachment[]) {
        const formData = new FormData();
        formData.append('message', message);
        formData.append('session_id', sessionId);
        formData.append('voice_enabled', settings.voiceEnabled.toString());

        // Add files if any
        for (const attachment of attachments) {
            // In real implementation, you'd append actual file objects
            formData.append('files', new Blob([]), attachment.name);
        }

        const response = await fetch('/api/v1/ai-chat/multimodal', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        return response.json();
    }

    function handleResponse(response: any) {
        const assistantMessage: Message = {
            id: `msg-${Date.now()}`,
            type: 'assistant',
            content: response.message,
            timestamp: new Date(),
            charts: response.charts,
            visualizations: response.visualizations,
            tradingSignals: response.trading_signals,
            confidence: response.confidence,
            sources: response.sources,
            audioUrl: response.audio_url,
            analysisType: response.analysis_type,
            detectedPatterns: response.detected_patterns,
            riskMetrics: response.risk_metrics,
        };

        setMessages((prev) => [...prev, assistantMessage]);

        // Play audio if enabled
        if (settings.autoPlayResponses && response.audio_url && audioRef.current) {
            audioRef.current.src = response.audio_url;
            audioRef.current.play();
        }

        // Show trading signals as notifications
        if (response.trading_signals?.length > 0) {
            const signal = response.trading_signals[0];
            showNotification(
                `${signal.type} Signal: ${signal.symbol} - ${signal.reasoning}`,
                signal.type === 'BUY' ? 'success' : 'warning'
            );
        }
    }

    function handleStreamingResponse(data: any) {
        if (data.done) {
            handleResponse(data.data);
        } else {
            // Update the last assistant message with streaming content
            setMessages((prev) => {
                const lastMessage = prev[prev.length - 1];
                if (lastMessage?.type === 'assistant') {
                    return [
                        ...prev.slice(0, -1),
                        { ...lastMessage, content: data.content },
                    ];
                } else {
                    return [
                        ...prev,
                        {
                            id: `msg-${Date.now()}`,
                            type: 'assistant',
                            content: data.content,
                            timestamp: new Date(),
                        },
                    ];
                }
            });
        }
    }

    function toggleRecording() {
        if (!recognitionRef.current) return;

        if (isRecording) {
            recognitionRef.current.stop();
        } else {
            recognitionRef.current.start();
            setIsRecording(true);
        }
    }

    async function generateChart(symbol: string, period: string = '1mo') {
        setIsLoading(true);
        try {
            const response = await fetch('/api/v1/ai-chat/generate/chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol,
                    period,
                    indicators: ['SMA20', 'SMA50', 'BB'],
                    session_id: sessionId,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                const chartMessage: Message = {
                    id: `msg-${Date.now()}`,
                    type: 'assistant',
                    content: `Here's the ${period} chart for ${symbol}:`,
                    timestamp: new Date(),
                    charts: [data.chart],
                };
                setMessages((prev) => [...prev, chartMessage]);
            }
        } catch (error) {
            console.error('Error generating chart:', error);
            showNotification('Failed to generate chart', 'error');
        } finally {
            setIsLoading(false);
        }
    }

    async function analyzePortfolio() {
        // Example portfolio - in real app, get from user data
        const holdings: PortfolioHolding[] = [
            { symbol: 'AAPL', shares: 100, value: 17500 },
            { symbol: 'GOOGL', shares: 50, value: 7000 },
            { symbol: 'MSFT', shares: 75, value: 28500 },
        ];

        setIsLoading(true);
        try {
            const response = await fetch('/api/v1/ai-chat/analyze/portfolio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    holdings,
                    session_id: sessionId,
                }),
            });

            if (response.ok) {
                const analysis = await response.json();
                const portfolioMessage: Message = {
                    id: `msg-${Date.now()}`,
                    type: 'assistant',
                    content: `Portfolio Analysis:\n\nTotal Value: $${analysis.statistics.total_value.toLocaleString()}\nExpected Return: ${(analysis.statistics.expected_return * 100).toFixed(2)}%\nVolatility: ${(analysis.statistics.volatility * 100).toFixed(2)}%\nSharpe Ratio: ${analysis.statistics.sharpe_ratio.toFixed(2)}\n\nSuggestions:\n${analysis.suggestions.join('\n')}`,
                    timestamp: new Date(),
                    visualizations: [
                        {
                            type: 'portfolio_allocation',
                            data: analysis.current_allocation,
                        },
                    ],
                };
                setMessages((prev) => [...prev, portfolioMessage]);
            }
        } catch (error) {
            console.error('Error analyzing portfolio:', error);
            showNotification('Failed to analyze portfolio', 'error');
        } finally {
            setIsLoading(false);
        }
    }

    async function runBacktest() {
        const strategy: BacktestStrategy = {
            shortWindow: 20,
            longWindow: 50,
            stopLoss: 0.02,
            takeProfit: 0.05,
        };

        setIsLoading(true);
        try {
            const response = await fetch('/api/v1/ai-chat/backtest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    strategy,
                    symbol: 'SPY',
                    period: '1y',
                    session_id: sessionId,
                }),
            });

            if (response.ok) {
                const results = await response.json();
                const backtestMessage: Message = {
                    id: `msg-${Date.now()}`,
                    type: 'assistant',
                    content: `Backtest Results:\n\nTotal Return: ${(results.summary.total_return * 100).toFixed(2)}%\nSharpe Ratio: ${results.summary.sharpe_ratio.toFixed(2)}\nMax Drawdown: ${(results.summary.max_drawdown * 100).toFixed(2)}%\nWin Rate: ${(results.summary.win_rate * 100).toFixed(2)}%`,
                    timestamp: new Date(),
                    charts: [results.chart],
                };
                setMessages((prev) => [...prev, backtestMessage]);
            }
        } catch (error) {
            console.error('Error running backtest:', error);
            showNotification('Failed to run backtest', 'error');
        } finally {
            setIsLoading(false);
        }
    }

    function showNotification(message: string, severity: 'success' | 'error' | 'warning' | 'info') {
        setNotification({ message, severity });
    }

    function downloadConversation() {
        const content = messages
            .map((msg) => `[${format(msg.timestamp, 'yyyy-MM-dd HH:mm:ss')}] ${msg.type.toUpperCase()}: ${msg.content}`)
            .join('\n\n');

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-${sessionId}-${format(new Date(), 'yyyyMMdd-HHmmss')}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    function shareConversation() {
        // In real app, generate shareable link
        navigator.clipboard.writeText(window.location.href);
        showNotification('Conversation link copied to clipboard!', 'success');
    }

    // Render functions
    const renderMessage = (message: Message) => {
        const isUser = message.type === 'user';
        const isSystem = message.type === 'system';

        return (
            <ListItem
                key={message.id}
                sx={{
                    flexDirection: isUser ? 'row-reverse' : 'row',
                    gap: 2,
                    mb: 2,
                }}
            >
                <Avatar
                    sx={{
                        bgcolor: isUser ? theme.palette.primary.main : isSystem ? theme.palette.warning.main : theme.palette.secondary.main,
                    }}
                >
                    {isUser ? 'U' : isSystem ? 'S' : <AIIcon />}
                </Avatar>

                <Paper
                    elevation={1}
                    sx={{
                        p: 2,
                        maxWidth: '70%',
                        bgcolor: isUser ? theme.palette.primary.dark : theme.palette.background.paper,
                        color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
                    }}
                >
                    {/* Message content */}
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

                    {/* Attachments */}
                    {message.attachments && message.attachments.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                            {message.attachments.map((attachment) => (
                                <Chip
                                    key={attachment.id}
                                    label={attachment.name}
                                    icon={<AttachFileIcon />}
                                    size="small"
                                    sx={{ mr: 1, mb: 1 }}
                                />
                            ))}
                        </Box>
                    )}

                    {/* Charts */}
                    {message.charts && message.charts.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                            {message.charts.map((chart, index) => (
                                <Box key={index} sx={{ mb: 2 }}>
                                    <img
                                        src={chart}
                                        alt={`Chart ${index + 1}`}
                                        style={{ width: '100%', borderRadius: 8 }}
                                    />
                                </Box>
                            ))}
                        </Box>
                    )}

                    {/* Trading Signals */}
                    {message.tradingSignals && message.tradingSignals.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                            {message.tradingSignals.map((signal, index) => (
                                <Card key={index} sx={{ mb: 1 }}>
                                    <CardContent>
                                        <Box display="flex" alignItems="center" gap={1}>
                                            <Chip
                                                label={signal.type}
                                                color={signal.type === 'BUY' ? 'success' : signal.type === 'SELL' ? 'error' : 'default'}
                                                size="small"
                                            />
                                            <Typography variant="subtitle2">{signal.symbol}</Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                Confidence: {(signal.confidence * 100).toFixed(0)}%
                                            </Typography>
                                        </Box>
                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                            {signal.reasoning}
                                        </Typography>
                                        {signal.entryPrice && (
                                            <Box sx={{ mt: 1 }}>
                                                <Typography variant="caption">
                                                    Entry: ${signal.entryPrice.toFixed(2)} |
                                                    Target: ${signal.targetPrice?.toFixed(2)} |
                                                    Stop: ${signal.stopLoss?.toFixed(2)}
                                                </Typography>
                                            </Box>
                                        )}
                                    </CardContent>
                                </Card>
                            ))}
                        </Box>
                    )}

                    {/* Metadata */}
                    <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Typography variant="caption" color="text.secondary">
                            {format(message.timestamp, 'HH:mm')}
                        </Typography>

                        {message.confidence && (
                            <Chip
                                label={`${(message.confidence * 100).toFixed(0)}% confident`}
                                size="small"
                                variant="outlined"
                            />
                        )}

                        {message.sources && message.sources.length > 0 && (
                            <Tooltip title={message.sources.join(', ')}>
                                <Chip
                                    label={`${message.sources.length} sources`}
                                    size="small"
                                    variant="outlined"
                                />
                            </Tooltip>
                        )}

                        {message.audioUrl && (
                            <IconButton
                                size="small"
                                onClick={() => {
                                    if (audioRef.current) {
                                        audioRef.current.src = message.audioUrl!;
                                        audioRef.current.play();
                                    }
                                }}
                            >
                                <VolumeUpIcon fontSize="small" />
                            </IconButton>
                        )}
                    </Box>
                </Paper>
            </ListItem>
        );
    };

    const renderQuickActions = () => (
        <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button
                variant="outlined"
                size="small"
                startIcon={<ChartIcon />}
                onClick={() => generateChart('SPY', '1d')}
            >
                SPY Chart
            </Button>
            <Button
                variant="outlined"
                size="small"
                startIcon={<PortfolioIcon />}
                onClick={analyzePortfolio}
            >
                Analyze Portfolio
            </Button>
            <Button
                variant="outlined"
                size="small"
                startIcon={<BacktestIcon />}
                onClick={runBacktest}
            >
                Run Backtest
            </Button>
            <Button
                variant="outlined"
                size="small"
                startIcon={<PatternIcon />}
                onClick={() => setInput('What chart patterns do you see in AAPL?')}
            >
                Pattern Analysis
            </Button>
        </Box>
    );

    const renderSettings = () => (
        <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="sm" fullWidth>
            <DialogTitle>Chat Settings</DialogTitle>
            <DialogContent>
                <Box sx={{ pt: 2 }}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={settings.voiceEnabled}
                                onChange={(e) => setSettings({ ...settings, voiceEnabled: e.target.checked })}
                            />
                        }
                        label="Enable Voice Input/Output"
                    />

                    <FormControlLabel
                        control={
                            <Switch
                                checked={settings.autoPlayResponses}
                                onChange={(e) => setSettings({ ...settings, autoPlayResponses: e.target.checked })}
                            />
                        }
                        label="Auto-play Voice Responses"
                    />

                    <FormControlLabel
                        control={
                            <Switch
                                checked={settings.streamingEnabled}
                                onChange={(e) => setSettings({ ...settings, streamingEnabled: e.target.checked })}
                            />
                        }
                        label="Enable Streaming Responses"
                    />

                    <FormControl fullWidth sx={{ mt: 2 }}>
                        <InputLabel>Chart Theme</InputLabel>
                        <Select
                            value={settings.chartTheme}
                            onChange={(e) => setSettings({ ...settings, chartTheme: e.target.value as 'light' | 'dark' })}
                        >
                            <MenuItem value="light">Light</MenuItem>
                            <MenuItem value="dark">Dark</MenuItem>
                        </Select>
                    </FormControl>

                    <FormControl fullWidth sx={{ mt: 2 }}>
                        <InputLabel>Language</InputLabel>
                        <Select
                            value={settings.language}
                            onChange={(e) => setSettings({ ...settings, language: e.target.value })}
                        >
                            <MenuItem value="en">English</MenuItem>
                            <MenuItem value="es">Spanish</MenuItem>
                            <MenuItem value="fr">French</MenuItem>
                            <MenuItem value="de">German</MenuItem>
                            <MenuItem value="zh">Chinese</MenuItem>
                            <MenuItem value="ja">Japanese</MenuItem>
                        </Select>
                    </FormControl>

                    <Box sx={{ mt: 2 }}>
                        <Typography gutterBottom>Speech Rate</Typography>
                        <Slider
                            value={settings.speechRate}
                            onChange={(e, value) => setSettings({ ...settings, speechRate: value as number })}
                            min={0.5}
                            max={2}
                            step={0.1}
                            marks
                            valueLabelDisplay="auto"
                        />
                    </Box>
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={() => setShowSettings(false)}>Close</Button>
                <Button onClick={() => setSettings(defaultSettings)} color="secondary">
                    Reset to Defaults
                </Button>
            </DialogActions>
        </Dialog>
    );

    return (
        <Box
            sx={{
                height: isFullscreen ? '100vh' : '600px',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
            }}
        >
            {/* Header */}
            <AppBar position="static" color="default" elevation={1}>
                <Toolbar variant="dense">
                    <AIIcon sx={{ mr: 1 }} />
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        AI Trading Assistant
                    </Typography>

                    <Tooltip title="Download Conversation">
                        <IconButton size="small" onClick={downloadConversation}>
                            <DownloadIcon />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title="Share Conversation">
                        <IconButton size="small" onClick={shareConversation}>
                            <ShareIcon />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title="Settings">
                        <IconButton size="small" onClick={() => setShowSettings(true)}>
                            <SettingsIcon />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
                        <IconButton size="small" onClick={() => setIsFullscreen(!isFullscreen)}>
                            {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
                        </IconButton>
                    </Tooltip>
                </Toolbar>
            </AppBar>

            {/* Quick Actions */}
            <Box sx={{ p: 2, bgcolor: 'background.default' }}>
                {renderQuickActions()}
            </Box>

            {/* Messages Area */}
            <Box
                sx={{
                    flexGrow: 1,
                    overflow: 'auto',
                    p: 2,
                    bgcolor: 'background.default',
                }}
                {...getRootProps()}
            >
                <input {...getInputProps()} />

                {isDragActive && (
                    <Box
                        sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            bgcolor: 'rgba(0, 0, 0, 0.5)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            zIndex: 1000,
                        }}
                    >
                        <Paper sx={{ p: 4 }}>
                            <Typography variant="h6">Drop files here to analyze</Typography>
                        </Paper>
                    </Box>
                )}

                <List>
                    {messages.map(renderMessage)}
                    {isLoading && (
                        <ListItem>
                            <Avatar sx={{ bgcolor: theme.palette.secondary.main }}>
                                <AIIcon />
                            </Avatar>
                            <Box sx={{ ml: 2 }}>
                                <CircularProgress size={20} />
                                <Typography variant="body2" sx={{ ml: 2, display: 'inline' }}>
                                    Analyzing...
                                </Typography>
                            </Box>
                        </ListItem>
                    )}
                </List>
                <div ref={messagesEndRef} />
            </Box>

            {/* Attachments Preview */}
            {attachments.length > 0 && (
                <Box sx={{ p: 1, bgcolor: 'background.paper', borderTop: 1, borderColor: 'divider' }}>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {attachments.map((attachment) => (
                            <Chip
                                key={attachment.id}
                                label={attachment.name}
                                onDelete={() => setAttachments(attachments.filter((a) => a.id !== attachment.id))}
                                icon={
                                    attachment.type.startsWith('image/') ? <ImageIcon /> :
                                        attachment.type.includes('csv') || attachment.type.includes('excel') ? <AssessmentIcon /> :
                                            <AttachFileIcon />
                                }
                            />
                        ))}
                    </Box>
                </Box>
            )}

            {/* Input Area */}
            <Box
                sx={{
                    p: 2,
                    bgcolor: 'background.paper',
                    borderTop: 1,
                    borderColor: 'divider',
                }}
            >
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
                    <IconButton
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isLoading}
                    >
                        <AttachFileIcon />
                    </IconButton>

                    <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        hidden
                        onChange={(e) => {
                            if (e.target.files) {
                                handleFileDrop(Array.from(e.target.files));
                            }
                        }}
                    />

                    {settings.voiceEnabled && (
                        <IconButton
                            onClick={toggleRecording}
                            disabled={isLoading}
                            color={isRecording ? 'error' : 'default'}
                        >
                            {isRecording ? <MicIcon /> : <MicOffIcon />}
                        </IconButton>
                    )}

                    <TextField
                        fullWidth
                        multiline
                        maxRows={4}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        placeholder="Ask me anything about trading, upload charts for analysis, or request portfolio optimization..."
                        disabled={isLoading || isRecording}
                        variant="outlined"
                        size="small"
                    />

                    <IconButton
                        onClick={handleSend}
                        disabled={isLoading || (!input.trim() && attachments.length === 0)}
                        color="primary"
                    >
                        <SendIcon />
                    </IconButton>
                </Box>

                {/* Suggested prompts */}
                <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    <Chip
                        label="Analyze AAPL chart"
                        size="small"
                        onClick={() => setInput('Can you analyze the AAPL chart and identify any patterns?')}
                    />
                    <Chip
                        label="Market overview"
                        size="small"
                        onClick={() => setInput('Give me a market overview for today')}
                    />
                    <Chip
                        label="Options strategy"
                        size="small"
                        onClick={() => setInput('What options strategy would you recommend for SPY?')}
                    />
                    <Chip
                        label="Risk analysis"
                        size="small"
                        onClick={() => setInput('Analyze my portfolio risk')}
                    />
                </Box>
            </Box>

            {/* Hidden audio element */}
            <audio ref={audioRef} hidden />

            {/* Settings Dialog */}
            {renderSettings()}

            {/* Notifications */}
            <Snackbar
                open={!!notification}
                autoHideDuration={6000}
                onClose={() => setNotification(null)}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                {notification && (
                    <Alert
                        onClose={() => setNotification(null)}
                        severity={notification.severity}
                        sx={{ width: '100%' }}
                    >
                        {notification.message}
                    </Alert>
                )}
            </Snackbar>
        </Box>
    );
}; 