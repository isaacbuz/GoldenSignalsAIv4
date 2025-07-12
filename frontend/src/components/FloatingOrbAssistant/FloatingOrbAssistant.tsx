import React, { useState } from 'react';
import { Fab, Tooltip, Zoom, useTheme } from '@mui/material';
import { Psychology } from '@mui/icons-material';
import { motion } from 'framer-motion';

export const FloatingOrbAssistant: React.FC<{ onClick: () => void }> = ({ onClick }) => {
  const theme = useTheme();
  const [hovered, setHovered] = useState(false);

  return (
    <Tooltip title="Ask AI Prophet" placement="left">
      <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ duration: 0.5 }} whileHover={{ scale: 1.2 }}>
        <Fab color="primary" onClick={onClick} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)} sx={{ position: 'fixed', bottom: 20, right: 20, background: 'linear-gradient(45deg, #FFD700, #FFA500)', boxShadow: '0 0 20px rgba(255,215,0,0.5)' }}>
          <Psychology />
          <Zoom in={hovered}><Typography sx={{ ml: 1 }}>AI</Typography></Zoom>
        </Fab>
      </motion.div>
    </Tooltip>
  );
};
