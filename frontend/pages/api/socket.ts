import { Server } from 'socket.io';

let io: any;

export default function handler(req: any, res: any) {
  if (!io) {
    io = new Server(res.socket.server, {
      path: "/api/socket",
    });

    io.on("connection", (socket: any) => {
      console.log("User connected");
      // Emit mock ai-signal events every 5 seconds (LiveSignalBundle format)
      setInterval(() => {
        socket.emit("ai-signal", {
          symbol: "AAPL",
          signals: [
            { name: "RSI", signal: "buy", confidence: 75, explanation: "RSI < 30" },
            { name: "MACD", signal: "buy", confidence: 80, explanation: "MACD crossover" },
            { source: "TV_AI_Signals_V3", signal: "buy", confidence: 90, explanation: "AI V3 bullish" }
          ]
        });
      }, 5000);
    });

    res.socket.server.io = io;
  }
  res.end();
}
