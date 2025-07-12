import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';

interface MiniChartProps {
    symbol: string;
    data?: number[];
    height?: number;
    width?: number;
}

export const MiniChart: React.FC<MiniChartProps> = ({
    symbol,
    data = [100, 102, 98, 105, 103, 107, 104],
    height = 60,
    width = 120,
}) => {
    const theme = useTheme();

    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;

    return (
        <Box sx={{ width, height }}>
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
                {symbol}
            </Typography>
            <svg width={width} height={height - 20} viewBox={`0 0 ${width} ${height - 20}`}>
                <path
                    d={data.map((value, index) => {
                        const x = (index / (data.length - 1)) * width;
                        const y = ((max - value) / range) * (height - 20);
                        return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke={theme.palette.primary.main}
                    strokeWidth="1.5"
                />
            </svg>
        </Box>
    );
}; 