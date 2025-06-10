import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemText, ListItemAvatar, Avatar, useTheme, alpha, Box } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import { useSignals } from '../../store';

export const SignalStream: React.FC = () => {
  const { signals } = useSignals();
  const theme = useTheme();

  return (
    <Card sx={{ height: 440 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Signal Stream</Typography>
        <List dense>
          {signals.slice(0, 5).map((signal, index) => (
            <ListItem key={index} divider>
              <ListItemAvatar>
                <Avatar sx={{ 
                  bgcolor: signal.signal_type === 'BUY' ? alpha(theme.palette.success.main, 0.1) : alpha(theme.palette.error.main, 0.1),
                  color: signal.signal_type === 'BUY' ? 'success.main' : 'error.main'
                }}>
                  {signal.signal_type === 'BUY' ? <TrendingUp /> : <TrendingDown />}
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={`${signal.symbol} - ${signal.signal_type}`}
                secondary={`Confidence: ${Math.round(signal.confidence * 100)}%`}
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
}; 