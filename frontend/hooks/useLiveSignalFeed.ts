import { useEffect, useState } from 'react';
import io from 'socket.io-client';

export type Signal = {
  name?: string;
  source?: string;
  signal: string;
  confidence?: number;
  explanation?: string;
};

export type LiveSignalBundle = {
  symbol: string;
  signals: Signal[];
};

export const useLiveSignalFeed = () => {
  const [signalData, setSignalData] = useState<LiveSignalBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState<boolean>(false);

  useEffect(() => {
    const socket = io({ path: "/api/socket", reconnection: true, reconnectionAttempts: 5 });
    socket.on("connect", () => {
      setConnected(true);
      setError(null);
    });
    socket.on("disconnect", () => {
      setConnected(false);
    });
    socket.on("connect_error", (err: any) => {
      setError("Connection error: " + (err?.message || "Unknown error"));
      setConnected(false);
    });
    socket.on("ai-signal", (data: LiveSignalBundle) => {
      setSignalData(data);
      setError(null);
    });
    return () => { socket.disconnect(); }
  }, []);

  return { signalData, error, connected };
};
