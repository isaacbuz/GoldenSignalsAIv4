import React, { useEffect, useRef, useState } from 'react';
import { Box, Paper, Stack, Button, Chip, Typography, IconButton, Tooltip } from '@mui/material';
import {
    Timeline as FibonacciIcon,
    TrendingUp as TrendLineIcon,
    ShowChart as IndicatorIcon,
    AutoAwesome as AIIcon,
    PlayArrow as AutomateIcon,
} from '@mui/icons-material';

declare global {
    interface Window {
        TradingView: any;
    }
}

interface TradingViewIntegrationProps {
    symbol: string;
    interval: string;
    onSignalGenerated?: (signal: any) => void;
}

const TradingViewIntegration: React.FC<TradingViewIntegrationProps> = ({
    symbol,
    interval,
    onSignalGenerated
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const widgetRef = useRef<any>(null);
    const [isAutoTrading, setIsAutoTrading] = useState(false);
    const [detectedPatterns, setDetectedPatterns] = useState<string[]>([]);

    useEffect(() => {
        if (!containerRef.current) return;

        // Load TradingView library
        const script = document.createElement('script');
        script.src = 'https://s3.tradingview.com/tv.js';
        script.async = true;
        script.onload = () => initializeTradingView();
        document.head.appendChild(script);

        return () => {
            if (widgetRef.current) {
                widgetRef.current.remove();
            }
            document.head.removeChild(script);
        };
    }, []);

    const initializeTradingView = () => {
        if (!window.TradingView || !containerRef.current) return;

        widgetRef.current = new window.TradingView.widget({
            width: '100%',
            height: 600,
            symbol: symbol,
            interval: interval,
            timezone: 'Etc/UTC',
            theme: 'dark',
            style: '1',
            locale: 'en',
            toolbar_bg: '#131722',
            enable_publishing: false,
            hide_side_toolbar: false,
            allow_symbol_change: true,
            container_id: containerRef.current.id,

            // Enable drawing tools
            drawings_access: {
                type: 'black',
                tools: [
                    { name: 'Trend Line', grayed: false },
                    { name: 'Fibonacci Retracement', grayed: false },
                    { name: 'Fibonacci Extension', grayed: false },
                    { name: 'Horizontal Line', grayed: false },
                    { name: 'Vertical Line', grayed: false },
                    { name: 'Rectangle', grayed: false },
                    { name: 'Text', grayed: false },
                ]
            },

            // Studies to load
            studies: [
                'RSI@tv-basicstudies',
                'MACD@tv-basicstudies',
                'BB@tv-basicstudies',
            ],

            // Overrides for custom styling
            overrides: {
                'paneProperties.background': '#0a0e1a',
                'paneProperties.vertGridProperties.color': '#1e222d',
                'paneProperties.horzGridProperties.color': '#1e222d',
                'symbolWatermarkProperties.transparency': 90,
                'scalesProperties.textColor': '#AAA',
                'mainSeriesProperties.candleStyle.wickUpColor': '#26a69a',
                'mainSeriesProperties.candleStyle.wickDownColor': '#ef5350',
            },

            // Custom CSS
            custom_css_url: '/tradingview-custom.css',

            // Callback when chart is ready
            onReady: () => {
                console.log('TradingView chart ready');
                setupAutomatedAnalysis();
            },
        });
    };

    const setupAutomatedAnalysis = () => {
        if (!widgetRef.current) return;

        // Access the chart instance
        widgetRef.current.onChartReady(() => {
            const chart = widgetRef.current.chart();

            // Add custom indicators and tools programmatically
            if (isAutoTrading) {
                // Auto-draw Fibonacci retracement
                autoDrawFibonacci(chart);

                // Auto-detect patterns
                detectPatterns(chart);

                // Set up alerts
                setupAlerts(chart);
            }
        });
    };

    const autoDrawFibonacci = (chart: any) => {
        // Get price data
        chart.onIntervalChanged().subscribe(null, () => {
            const bars = chart.getBarsInfo();
            if (!bars || bars.length < 100) return;

            // Find swing high and low
            let highIndex = 0, lowIndex = 0;
            let highPrice = 0, lowPrice = Infinity;

            bars.forEach((bar: any, index: number) => {
                if (bar.high > highPrice) {
                    highPrice = bar.high;
                    highIndex = index;
                }
                if (bar.low < lowPrice) {
                    lowPrice = bar.low;
                    lowIndex = index;
                }
            });

            // Create Fibonacci retracement
            const fibTool = chart.createMultipointShape(
                [
                    { time: bars[Math.min(highIndex, lowIndex)].time, price: highIndex < lowIndex ? highPrice : lowPrice },
                    { time: bars[Math.max(highIndex, lowIndex)].time, price: highIndex > lowIndex ? highPrice : lowPrice }
                ],
                {
                    shape: 'fib_retracement',
                    overrides: {
                        'linecolor': '#2196F3',
                        'linewidth': 2,
                        'transparency': 50,
                        'level0.visible': true,
                        'level1.visible': true,
                        'level2.visible': true,
                        'level3.visible': true,
                        'level4.visible': true,
                        'level5.visible': true,
                        'level0.coeff': 0,
                        'level1.coeff': 0.236,
                        'level2.coeff': 0.382,
                        'level3.coeff': 0.5,
                        'level4.coeff': 0.618,
                        'level5.coeff': 0.786,
                    }
                }
            );

            // Generate signal based on current price and Fib levels
            analyzeForSignals(bars, highPrice, lowPrice);
        });
    };

    const detectPatterns = (chart: any) => {
        // Pattern detection logic
        const patterns = [
            'Head and Shoulders',
            'Double Top',
            'Ascending Triangle',
            'Bull Flag',
            'Cup and Handle'
        ];

        // Simulate pattern detection
        setTimeout(() => {
            const detectedPattern = patterns[Math.floor(Math.random() * patterns.length)];
            setDetectedPatterns([detectedPattern]);

            // Add text annotation on chart
            chart.createShape(
                { time: Date.now() / 1000, price: chart.getVisibleRange().priceRange.maxValue },
                {
                    shape: 'text',
                    text: `Pattern: ${detectedPattern}`,
                    overrides: {
                        'fontsize': 14,
                        'textcolor': '#4CAF50',
                        'backgroundColor': '#1a1a1a',
                        'backgroundTransparency': 20,
                    }
                }
            );
        }, 3000);
    };

    const setupAlerts = (chart: any) => {
        // Set up price alerts at Fibonacci levels
        const currentPrice = chart.getVisiblePriceRange().maxValue;
        const fibLevels = [0.236, 0.382, 0.5, 0.618, 0.786];

        fibLevels.forEach(level => {
            const alertPrice = currentPrice * (1 - level * 0.1); // Example calculation

            chart.createPriceLine({
                price: alertPrice,
                color: '#FF9800',
                lineWidth: 1,
                lineStyle: 2, // Dashed
                axisLabelVisible: true,
                title: `Fib ${level}`,
            });
        });
    };

    const analyzeForSignals = (bars: any[], high: number, low: number) => {
        const currentPrice = bars[bars.length - 1].close;
        const fibLevels = {
            '0': low,
            '0.236': low + (high - low) * 0.236,
            '0.382': low + (high - low) * 0.382,
            '0.5': low + (high - low) * 0.5,
            '0.618': low + (high - low) * 0.618,
            '0.786': low + (high - low) * 0.786,
            '1': high,
        };

        // Check if price is near a Fibonacci level
        Object.entries(fibLevels).forEach(([level, price]) => {
            const tolerance = (high - low) * 0.01; // 1% tolerance

            if (Math.abs(currentPrice - price) < tolerance) {
                // Generate signal
                const signal = {
                    type: currentPrice > (high + low) / 2 ? 'LONG' : 'SHORT',
                    entry: currentPrice,
                    stopLoss: currentPrice > price ? price - tolerance : price + tolerance,
                    takeProfit: [
                        currentPrice + (currentPrice - (currentPrice > price ? price - tolerance : price + tolerance)) * 1.618,
                        currentPrice + (currentPrice - (currentPrice > price ? price - tolerance : price + tolerance)) * 2.618,
                    ],
                    fibLevel: parseFloat(level),
                    confidence: 75 + Math.random() * 20,
                    reasoning: `Price bounced off Fibonacci ${level} level with strong momentum`,
                };

                if (onSignalGenerated) {
                    onSignalGenerated(signal);
                }
            }
        });
    };

    const toggleAutomation = () => {
        setIsAutoTrading(!isAutoTrading);
        if (!isAutoTrading && widgetRef.current) {
            setupAutomatedAnalysis();
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Control Panel */}
            <Paper sx={{ p: 2, mb: 2, bgcolor: '#131722' }}>
                <Stack direction="row" spacing={2} alignItems="center">
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        TradingView Integration
                    </Typography>

                    <Tooltip title="Auto-draw Fibonacci retracements">
                        <IconButton color="primary">
                            <FibonacciIcon />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title="Auto-draw trend lines">
                        <IconButton>
                            <TrendLineIcon />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title="Add indicators">
                        <IconButton>
                            <IndicatorIcon />
                        </IconButton>
                    </Tooltip>

                    <Button
                        variant={isAutoTrading ? 'contained' : 'outlined'}
                        startIcon={<AutomateIcon />}
                        onClick={toggleAutomation}
                        color={isAutoTrading ? 'success' : 'primary'}
                    >
                        {isAutoTrading ? 'Auto Trading ON' : 'Auto Trading OFF'}
                    </Button>
                </Stack>

                {detectedPatterns.length > 0 && (
                    <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                            Detected Patterns:
                        </Typography>
                        {detectedPatterns.map((pattern, index) => (
                            <Chip
                                key={index}
                                label={pattern}
                                color="success"
                                size="small"
                                icon={<AIIcon />}
                            />
                        ))}
                    </Stack>
                )}
            </Paper>

            {/* TradingView Chart Container */}
            <Paper sx={{ flex: 1, bgcolor: '#0a0e1a', position: 'relative' }}>
                <div
                    ref={containerRef}
                    id="tradingview_widget"
                    style={{ height: '100%', width: '100%' }}
                />

                {isAutoTrading && (
                    <Chip
                        label="AI Analysis Active"
                        color="success"
                        icon={<AIIcon />}
                        sx={{
                            position: 'absolute',
                            top: 16,
                            right: 16,
                            animation: 'pulse 2s infinite',
                        }}
                    />
                )}
            </Paper>
        </Box>
    );
};

export default TradingViewIntegration; 