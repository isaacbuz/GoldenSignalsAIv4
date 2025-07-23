import React, { useState, useEffect } from 'react';
import { List, ListItem, ListItemText, Typography, Box, Chip, Stack, useTheme } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';

import { Signal } from '../../../types/signals';
import logger from '../../services/logger';


export const RealTimeFeed: React.FC = () => {
  const theme = useTheme();
  const [signals, setSignals] = useState<Signal[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000/ws');
    socket.onopen = () => logger.info('WebSocket connected');
    socket.onmessage = (event) => {
      const newSignal = JSON.parse(event.data);
      setSignals(prev => [newSignal, ...prev.slice(0, 49)]);
    };
    socket.onclose = () => logger.info('WebSocket disconnected');
    setWs(socket);

    return () => socket.close();
  }, []);

  return (
    <Box sx={{ height: '100%', overflowY: 'auto', background: theme.palette.background.paper, p: 2 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Real-Time Signal Feed</Typography>
      <AnimatePresence>
        <List>
          {signals.map((signal, index) => (
            <motion.div key={signal.id} initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: index * 0.1 }}>
              <ListItem>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Chip label={signal.action} color={signal.action === 'BUY' ? 'success' : 'error'} />
                  <ListItemText primary={signal.symbol} secondary={signal.reasoning} />
                  <Typography>{(signal.confidence * 100).toFixed(0)}%</Typography>
                </Stack>
              </ListItem>
            </motion.div>
          ))}
        </List>
      </AnimatePresence>
    </Box>
  );
};

export default RealTimeFeed;
