import React from 'react';
import { Card, CardContent, Typography, Box, useTheme, alpha, Avatar, Stack } from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';

export interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon: React.ReactNode;
  color?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  trend = 'neutral',
  icon,
  color,
}) => {
  const theme = useTheme();
  const trendColor =
    trend === 'up'
      ? theme.palette.success.main
      : trend === 'down'
      ? theme.palette.error.main
      : theme.palette.text.secondary;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography variant="body2" color="text.secondary">{title}</Typography>
            <Typography variant="h5" sx={{ fontWeight: 'bold', my: 1 }}>{value}</Typography>
            {change && (
              <Stack direction="row" alignItems="center" spacing={0.5}>
                {trend === 'up' ? <TrendingUp sx={{ color: trendColor, fontSize: '1rem' }} /> : <TrendingDown sx={{ color: trendColor, fontSize: '1rem' }} />}
                <Typography variant="body2" sx={{ color: trendColor }}>{change}</Typography>
              </Stack>
            )}
          </Box>
          <Avatar sx={{ bgcolor: alpha(color || theme.palette.primary.main, 0.1), color: color || theme.palette.primary.main }}>{icon}</Avatar>
        </Stack>
      </CardContent>
    </Card>
  );
}; 