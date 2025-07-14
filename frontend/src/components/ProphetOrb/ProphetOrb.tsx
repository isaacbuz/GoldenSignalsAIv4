import React from 'react';

export interface ProphetOrbProps {
  insights: string;
}

export const ProphetOrb: React.FC<ProphetOrbProps> = ({ insights }) => {
  return (
    <div style={{
      position: 'absolute',
      top: 20,
      right: 20,
      width: 60,
      height: 60,
      borderRadius: '50%',
      background: 'gold',
      animation: 'pulse 2s infinite',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'black',
      fontSize: 12,
      textAlign: 'center'
    }}>
      {insights || 'AI Thinking...'}
    </div>
  );
};
// Add @keyframes pulse in CSS