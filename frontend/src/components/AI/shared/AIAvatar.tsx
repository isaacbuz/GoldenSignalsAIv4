import React from 'react';
import { Avatar, Box, keyframes } from '@mui/material';
import { AutoAwesome, Psychology, SmartToy } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(255, 215, 0, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 215, 0, 0);
  }
`;

const rotate = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

const StyledAvatar = styled(Avatar)<{ isActive?: boolean; size?: string }>(
  ({ theme, isActive, size }) => ({
    width: size === 'small' ? 32 : size === 'large' ? 64 : 48,
    height: size === 'small' ? 32 : size === 'large' ? 64 : 48,
    background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
    color: '#0A0E27',
    ...(isActive && {
      animation: `${pulse} 2s infinite`,
    }),
  })
);

const ProcessingRing = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: -4,
  left: -4,
  right: -4,
  bottom: -4,
  borderRadius: '50%',
  border: '2px solid transparent',
  borderTopColor: '#FFD700',
  animation: `${rotate} 1s linear infinite`,
}));

interface AIAvatarProps {
  variant?: 'sparkle' | 'brain' | 'robot';
  size?: 'small' | 'medium' | 'large';
  isActive?: boolean;
  isProcessing?: boolean;
  name?: string;
}

const AIAvatar: React.FC<AIAvatarProps> = ({
  variant = 'sparkle',
  size = 'medium',
  isActive = false,
  isProcessing = false,
  name,
}) => {
  const getIcon = () => {
    const iconSize = size === 'small' ? 20 : size === 'large' ? 40 : 28;
    switch (variant) {
      case 'brain':
        return <Psychology sx={{ fontSize: iconSize }} />;
      case 'robot':
        return <SmartToy sx={{ fontSize: iconSize }} />;
      default:
        return <AutoAwesome sx={{ fontSize: iconSize }} />;
    }
  };

  return (
    <Box sx={{ position: 'relative', display: 'inline-flex' }}>
      <StyledAvatar isActive={isActive} size={size}>
        {name ? name.charAt(0).toUpperCase() : getIcon()}
      </StyledAvatar>
      {isProcessing && <ProcessingRing />}
    </Box>
  );
};

export default AIAvatar;
