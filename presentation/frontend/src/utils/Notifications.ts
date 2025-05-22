// src/utils/Notifications.ts

export function requestNotificationPermission() {
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
}

export function sendSignalNotification({
  symbol,
  type,
  confidence,
  price,
}: {
  symbol: string;
  type: string;
  confidence: number;
  price?: number;
}) {
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification(`${symbol} ${type.toUpperCase()} Signal`, {
      body: `Confidence: ${Math.round(confidence * 100)}%` + (price ? ` @ $${price}` : ""),
      icon: "/favicon.ico", // optional: put trading logo here
    });

    const audio = new Audio("/signal.mp3");
    audio.volume = 0.3;
    audio.play().catch(() => {});
  }
}
