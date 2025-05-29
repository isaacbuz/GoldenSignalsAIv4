import React from 'react';
import { useTickerContext } from '@/context/TickerContext';

interface WatchlistItemProps {
  symbol: string;
}

const WatchlistItem: React.FC<WatchlistItemProps> = React.memo(({ symbol }) => {
  const { selected, setSelected } = useTickerContext();
  const isActive = selected === symbol;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      setSelected(symbol);
    }
  };

  return (
    <div
      className={`cursor-pointer px-3 py-2 rounded-lg font-inter font-semibold transition-all duration-200 select-none \
        ${isActive ? 'bg-[#273549] text-white shadow-neon' : 'text-accentBlue hover:bg-[#273549] hover:shadow-neon'}
      `}
      onClick={() => setSelected(symbol)}
      aria-selected={isActive}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      role="option"
    >
      {symbol}
    </div>
  );
});

export default WatchlistItem;
