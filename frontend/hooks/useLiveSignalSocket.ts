import { useEffect, useState } from 'react';
import io from 'socket.io-client';

export function useLiveSignalSocket() {
  const [signal, setSignal] = useState<any>(null);

  useEffect(() => {
    const socket = io({ path: '/api/socket' });
    socket.on('signal', setSignal);
    return () => { socket.disconnect(); }
  }, []);

  return signal;
}
