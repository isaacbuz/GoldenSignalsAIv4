import asyncio
import websocket

class WebSocketClient:
  def __init__(self, url):
    self.url = url

  async def connect(self, symbol):
    ws = websocket.WebSocketApp(self.url, on_message=self.on_message)
    ws.run_forever()

  def on_message(self, ws, message):
    # Parse and broadcast to frontend
    pass
# Usage: client = WebSocketClient('ws://example.com')
# asyncio.run(client.connect('AAPL')) 