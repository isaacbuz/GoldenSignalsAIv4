import { useEffect, useRef, useState } from "react";

export interface AISignal {
  id: string;
  symbol: string;
  action: string;
  confidence: number;
  reason: string;
  time: string;
}

export function useSignalWebSocket(url: string) {
  const [signals, setSignals] = useState<AISignal[]>([]);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(url);
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    ws.current.onerror = () => setConnected(false);
    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (Array.isArray(data)) {
          setSignals(data);
        } else if (data && data.id) {
          setSignals((prev) => [data, ...prev]);
        }
      } catch {}
    };
    return () => {
      ws.current?.close();
    };
  }, [url]);

  return { signals, connected };
}
