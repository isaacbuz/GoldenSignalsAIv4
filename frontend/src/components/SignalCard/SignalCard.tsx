import React, { useState } from 'react';
import { Card, CardContent, Typography, Box, IconButton, Chip, Collapse, Stack, Divider, LinearProgress, useTheme } from '@mui/material';
import { ExpandMore, TrendingUp, TrendingDown, Psychology, Speed, Analytics } from '@mui/icons-material';
import { motion } from 'framer-motion';

import { Signal } from '../../../types/signals';

interface SignalCardProps {
  signal: Signal;
}

export const SignalCard: React.FC<SignalCardProps> = ({ signal }) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);

  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY': return 'success';
      case 'SELL': return 'error';
      default: return 'warning';
    }
  };

  const handleExpand = () => setExpanded(!expanded);

  return (
    <motion.div whileHover={{ scale: 1.02 }} transition={{ duration: 0.2 }}>
      <Card sx={{ background: theme.palette.background.paper, border: `1px solid ${theme.palette.primary.light}` }}>
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">{signal.symbol}</Typography>
            <Chip label={signal.action} color={getActionColor(signal.action)} icon={signal.action === 'BUY' ? <TrendingUp /> : <TrendingDown />} />
          </Stack>
          <Box sx={{ my: 1 }}>
            <LinearProgress variant="determinate" value={signal.confidence * 100} color="primary" />
            <Typography variant="caption">Confidence: {(signal.confidence * 100).toFixed(0)}%</Typography>
          </Box>
          <Typography variant="body2">{signal.reasoning}</Typography>
          <IconButton onClick={handleExpand} sx={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>
            <ExpandMore />
          </IconButton>
          <Collapse in={expanded}>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2">Agent Votes</Typography>
            <Stack direction="row" spacing={1}>
              {signal.agentVotes?.map((vote, index) => (
                <Chip key={index} label={vote.agent} icon={<Psychology />} size="small" />
              ))}
            </Stack>
            <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
              <Box>
                <Speed />
                <Typography>RSI: {signal.indicators?.rsi}</Typography>
              </Box>
              <Box>
                <Analytics />
                <Typography>MACD: {signal.indicators?.macd}</Typography>
    </Box>
            </Stack>
          </Collapse>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default SignalCard;