/**
 * MCP Client for Frontend
 * Handles communication with MCP tools and Golden Eye backend
 */

import { API_BASE_URL } from '../../config/api.config';
import logger from '../logger';


export interface MCPToolResult {
  success: boolean;
  data: any;
  error?: string;
  execution_time?: number;
}

export interface GoldenEyeQueryParams {
  query: string;
  symbol: string;
  timeframe?: string;
  context?: Record<string, any>;
}

export interface MCPEvent {
  type: 'thinking' | 'text' | 'tool_execution' | 'agent_consultation' | 'chart_action' | 'error' | 'complete';
  message?: string;
  content?: string;
  tool?: string;
  agent?: string;
  result?: any;
  action?: any;
  llm?: string;
  agents?: string[];
  intent?: string;
}

export interface ConsensusResult {
  consensus: {
    action: string;
    confidence: number;
  };
  individual_signals: any[];
  participating_agents: number;
}

export interface PredictionResult {
  prediction: number[];
  confidence_bands: {
    upper: number[];
    lower: number[];
  };
  supporting_factors: string[];
  risk_score: number;
}

export class MCPClient {
  private ws: WebSocket | null = null;
  private eventSource: EventSource | null = null;
  private pendingRequests: Map<string, (result: any) => void> = new Map();
  private clientId: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor() {
    this.clientId = this.generateClientId();
  }

  /**
   * Execute an MCP tool directly
   */
  async executeMCPTool(toolName: string, params: Record<string, any>): Promise<MCPToolResult> {
    const response = await fetch(`${API_BASE_URL}/api/v1/mcp/execute/${toolName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      throw new Error(`MCP tool execution failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Execute a specific agent
   */
  async executeAgent(agentName: string, params: any): Promise<any> {
    return this.executeMCPTool(`analyze_with_${agentName.toLowerCase()}`, params);
  }

  /**
   * Get consensus from multiple agents
   */
  async getConsensus(symbol: string, agents: string[]): Promise<ConsensusResult> {
    const response = await fetch(`${API_BASE_URL}/api/v1/golden-eye/agents/consensus`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`
      },
      body: JSON.stringify({
        symbol,
        agents,
        timeframe: '1h',
        voting_method: 'weighted'
      })
    });

    if (!response.ok) {
      throw new Error(`Consensus request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get AI prediction
   */
  async getPrediction(symbol: string, horizon: number = 24): Promise<PredictionResult> {
    const response = await fetch(`${API_BASE_URL}/api/v1/golden-eye/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`
      },
      body: JSON.stringify({
        symbol,
        horizon,
        use_ensemble: true
      })
    });

    if (!response.ok) {
      throw new Error(`Prediction request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Stream Golden Eye query using Server-Sent Events
   */
  async *streamGoldenEyeQuery(params: GoldenEyeQueryParams): AsyncIterableIterator<MCPEvent> {
    // Close any existing connection
    if (this.eventSource) {
      this.eventSource.close();
    }

    const url = new URL(`${API_BASE_URL}/api/v1/golden-eye/query/stream`);

    return new Promise<AsyncIterableIterator<MCPEvent>>((resolve, reject) => {
      fetch(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify(params)
      }).then(response => {
        if (!response.ok) {
          throw new Error(`Query failed: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        async function* readStream(): AsyncIterableIterator<MCPEvent> {
          try {
            while (true) {
              const { done, value } = await reader.read();

              if (done) {
                break;
              }

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const data = line.slice(6);
                  if (data.trim()) {
                    try {
                      const event = JSON.parse(data);
                      yield event;
                    } catch (e) {
                      logger.error('Failed to parse SSE data:', e);
                    }
                  }
                }
              }
            }
          } finally {
            reader.releaseLock();
          }
        }

        resolve(readStream());
      }).catch(reject);
    });
  }

  /**
   * Connect to WebSocket for real-time communication
   */
  async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/v1/golden-eye/ws/${this.clientId}`;

      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        logger.info('Golden Eye WebSocket connected');
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onerror = (error) => {
        logger.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onclose = () => {
        logger.info('WebSocket disconnected');
        this.handleReconnect();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleWebSocketMessage(data);
        } catch (e) {
          logger.error('Failed to parse WebSocket message:', e);
        }
      };
    });
  }

  /**
   * Send query via WebSocket
   */
  async sendWebSocketQuery(params: GoldenEyeQueryParams): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      await this.connectWebSocket();
    }

    this.ws!.send(JSON.stringify({
      type: 'query',
      ...params
    }));
  }

  /**
   * Get agent status
   */
  async getAgentStatus(agents?: string[]): Promise<Record<string, any>> {
    const response = await fetch(`${API_BASE_URL}/api/v1/golden-eye/agents/discover`, {
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    });

    if (!response.ok) {
      throw new Error(`Agent status request failed: ${response.statusText}`);
    }

    const data = await response.json();

    // Transform to status format
    const status: Record<string, any> = {};
    for (const [agentName, agentInfo] of Object.entries(data.agents)) {
      status[agentName] = {
        available: true,
        type: (agentInfo as any).type,
        last_update: new Date().toISOString()
      };
    }

    return status;
  }

  /**
   * Discover available MCP tools
   */
  async discoverTools(toolType?: string): Promise<any[]> {
    const url = new URL(`${API_BASE_URL}/api/v1/golden-eye/tools/discover`);
    if (toolType) {
      url.searchParams.append('tool_type', toolType);
    }

    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    });

    if (!response.ok) {
      throw new Error(`Tool discovery failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.tools;
  }

  /**
   * Execute a workflow
   */
  async executeWorkflow(workflowName: string, symbol: string, parameters?: any): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/v1/golden-eye/workflow/${workflowName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`
      },
      body: JSON.stringify({
        symbol,
        parameters
      })
    });

    if (!response.ok) {
      throw new Error(`Workflow execution failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  private handleWebSocketMessage(data: any): void {
    // Handle different message types
    if (data.type === 'pong') {
      // Heartbeat response
      return;
    }

    // Check if this is a response to a pending request
    if (data.requestId && this.pendingRequests.has(data.requestId)) {
      const resolver = this.pendingRequests.get(data.requestId);
      this.pendingRequests.delete(data.requestId);
      resolver!(data);
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      logger.info(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      setTimeout(() => {
        this.connectWebSocket().catch(console.error);
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getAuthToken(): string {
    // Get auth token from localStorage or session
    return localStorage.getItem('auth_token') || '';
  }

  /**
   * Send heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Every 30 seconds
  }
}

// Export singleton instance
export const mcpClient = new MCPClient();
