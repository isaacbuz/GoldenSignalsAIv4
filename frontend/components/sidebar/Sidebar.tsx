import LogoHeader from './LogoHeader';
import TickerSearch from './TickerSearch';
import TimeRangeSelector from './TimeRangeSelector';
import WatchlistItem from './WatchlistItem';
import { useTickerContext } from '@/context/TickerContext';

export default function Sidebar() {
  const { tickers } = useTickerContext();

  return (
    <aside className="w-64 bg-bgDark h-full p-4 rounded-xl shadow-neon font-sans border-r border-borderSoft flex flex-col gap-y-4">
      <LogoHeader />
      <TickerSearch />
      <TimeRangeSelector />
      <h2 className="text-xl font-bold text-accentGreen mb-1 tracking-tight font-sans">Watchlist</h2>
      <div className="flex-1 overflow-y-auto max-h-[300px] space-y-3 pr-1">
        {tickers.map((symbol: string) => (
          <WatchlistItem key={symbol} symbol={symbol} />
        ))}
      </div>
    </aside>
  );
}
