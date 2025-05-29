import { useEffect, useState } from 'react';

export function useSignalPolling(ticker: string) {
  const [alert, setAlert] = useState<any>(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch(`/api/signal/check?ticker=${ticker}`);
      const data = await res.json();
      if (data?.triggered) setAlert(data.signal);
    }, 15000);

    return () => clearInterval(interval);
  }, [ticker]);

  return alert;
}
