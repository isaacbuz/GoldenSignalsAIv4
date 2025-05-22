import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  ReactNode,
} from "react";

// Types for price ticks and signals
export type Tick = { timestamp: number; price: number };
export type Signal = {
  id: string;
  symbol: string;
  type: "buy" | "sell" | "hold";
  confidence: number;
  timestamp: number;
  source?: string; // optional: RSI, LSTM, etc.
};

interface WebSocketState {
  ticks: Record<string, Tick[]>;
  signals: Record<string, Signal[]>;
}

const WebSocketContext = createContext<WebSocketState>({
  ticks: {},
  signals: {},
});

export const useWebSocket = () => useContext(WebSocketContext);

export const WebSocketProvider: React.FC<{ url: string; children: ReactNode }> = ({
  url,
  children,
}) => {
  const [ticks, setTicks] = useState<Record<string, Tick[]>>({});
  const [signals, setSignals] = useState<Record<string, Signal[]>>({});
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const now = Date.now();

      if (msg.type === "tick") {
        setTicks((prev) => {
          const history = prev[msg.symbol] || [];
          return {
            ...prev,
            [msg.symbol]: [...history, { timestamp: now, price: msg.price }].slice(-500),
          };
        });
      }

      if (msg.type === "signal") {
        setSignals((prev) => {
          const history = prev[msg.symbol] || [];
          return {
            ...prev,
            [msg.symbol]: [...history, msg].slice(-200),
          };
        });

        // Optional: trigger global UI reaction
        document.dispatchEvent(new CustomEvent("signal:new", { detail: msg }));
      }
    };

    ws.onclose = () => {
      console.warn("WebSocket closed, attempting reconnect...");
      setTimeout(() => window.location.reload(), 3000); // naive retry
    };

    return () => ws.close();
  }, [url]);

  return (
    <WebSocketContext.Provider value={{ ticks, signals }}>
      {children}
    </WebSocketContext.Provider>
  );
};
