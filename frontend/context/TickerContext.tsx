import { createContext, useContext, useState } from 'react';

const TickerContext = createContext<any>(null);

export const TickerProvider = ({ children }: { children: React.ReactNode }) => {
  const [tickers, setTickers] = useState<string[]>(['AAPL']);
  const [selected, setSelected] = useState<string>('AAPL');

  const addTicker = (symbol: string) => {
    if (!tickers.includes(symbol)) {
      setTickers([...tickers, symbol]);
    }
    setSelected(symbol);
  };

  return (
    <TickerContext.Provider value={{ tickers, selected, addTicker, setSelected }}>
      {children}
    </TickerContext.Provider>
  );
};

export const useTickerContext = () => useContext(TickerContext);
