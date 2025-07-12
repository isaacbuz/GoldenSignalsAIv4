/**
 * Debug utility to disable WebSocket connections
 * Can be used in browser console for testing
 */

export function disableWebSocket() {
    // Override WebSocket constructor
    (window as any).WebSocket = class MockWebSocket {
        readyState = 0; // CONNECTING
        url: string;

        constructor(url: string) {
            this.url = url;
            console.log(`[MockWebSocket] Blocked connection to: ${url}`);

            // Simulate connection failure after a delay
            setTimeout(() => {
                this.readyState = 3; // CLOSED
                if (this.onclose) {
                    this.onclose(new CloseEvent('close'));
                }
            }, 100);
        }

        send() {
            console.log('[MockWebSocket] Send blocked');
        }

        close() {
            this.readyState = 3;
            console.log('[MockWebSocket] Connection closed');
        }

        onopen: ((event: Event) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
    };

    console.log('✅ WebSocket disabled. Refresh the page to apply changes.');

    // Also set localStorage flag
    localStorage.setItem('DISABLE_WEBSOCKET', 'true');
}

// Make it available globally in development
if (process.env.NODE_ENV === 'development') {
    (window as any).disableWebSocket = disableWebSocket;
}

export function enableWebSocket() {
    localStorage.removeItem('DISABLE_WEBSOCKET');
    console.log('✅ WebSocket enabled. Refresh the page to apply changes.');
} 