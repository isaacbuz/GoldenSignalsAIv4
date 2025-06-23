import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useWebSocket } from '../useWebSocket'

// Mock WebSocket class
class MockWebSocket {
  url: string
  readyState: number
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null

  constructor(url: string) {
    this.url = url
    this.readyState = WebSocket.CONNECTING

    // Simulate connection opening
    setTimeout(() => {
      this.readyState = WebSocket.OPEN
      if (this.onopen) {
        this.onopen(new Event('open'))
      }
    }, 10)
  }

  send(data: string) {
    if (this.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
  }

  close() {
    this.readyState = WebSocket.CLOSED
    if (this.onclose) {
      this.onclose(new CloseEvent('close'))
    }
  }
}

describe('useWebSocket', () => {
  let mockWebSocket: MockWebSocket

  beforeEach(() => {
    // Replace global WebSocket with mock using vi.stubGlobal
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('connects to WebSocket on mount', async () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })
  })

  it('handles incoming messages', async () => {
    const onMessage = vi.fn()
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws', { onMessage })
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    // Simulate incoming message
    const mockData = { type: 'signal', data: { symbol: 'AAPL', action: 'BUY' } }
    act(() => {
      if (result.current.ws?.onmessage) {
        result.current.ws.onmessage(
          new MessageEvent('message', {
            data: JSON.stringify(mockData)
          })
        )
      }
    })

    expect(onMessage).toHaveBeenCalledWith(mockData)
    expect(result.current.lastMessage).toEqual(mockData)
  })

  it('sends messages correctly', async () => {
    const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    const sendSpy = vi.spyOn(result.current.ws!, 'send')
    const message = { type: 'subscribe', symbol: 'AAPL' }

    act(() => {
      result.current.sendMessage(message)
    })

    expect(sendSpy).toHaveBeenCalledWith(JSON.stringify(message))
  })

  it('handles connection errors', async () => {
    const onError = vi.fn()
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws', { onError })
    )

    await waitFor(() => {
      expect(result.current.ws).toBeDefined()
    })

    // Simulate error
    act(() => {
      if (result.current.ws?.onerror) {
        result.current.ws.onerror(new Event('error'))
      }
    })

    expect(onError).toHaveBeenCalled()
    expect(result.current.error).toBeDefined()
  })

  it('reconnects after disconnect', async () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws', {
        reconnect: true,
        reconnectInterval: 100
      })
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    // Simulate disconnect
    act(() => {
      result.current.ws?.close()
    })

    expect(result.current.isConnected).toBe(false)

    // Wait for reconnection
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    }, { timeout: 500 })
  })

  it('cleans up on unmount', async () => {
    const { result, unmount } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws')
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    const closeSpy = vi.spyOn(result.current.ws!, 'close')

    unmount()

    expect(closeSpy).toHaveBeenCalled()
  })
})
