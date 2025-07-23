/**
 * useAgentWebSocket Hook
 * Manages real-time WebSocket connection for agent signals and workflow updates
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import logger from '../services/logger';
import {
  AgentSignal,
  WorkflowUpdate,
  AgentStatus,
  AgentWebSocketMessage,
  WorkflowStage
} from '../types/agent.types';

interface UseAgentWebSocketOptions {
  enabled?: boolean;
  onSignal?: (signal: AgentSignal) => void;
  onWorkflowUpdate?: (update: WorkflowUpdate) => void;
  onStatusUpdate?: (status: AgentStatus) => void;
  onError?: (error: Error) => void;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

interface WebSocketState {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  reconnectCount: number;
  lastError: Error | null;
}

export function useAgentWebSocket(symbol: string, options: UseAgentWebSocketOptions = {}) {
  const {
    enabled = true,
    onSignal,
    onWorkflowUpdate,
    onStatusUpdate,
    onError,
    reconnectAttempts = 5,
    reconnectDelay = 3000
  } = options;

  // State
  const [agents, setAgents] = useState<Record<string, AgentSignal>>({});
  const [workflowProgress, setWorkflowProgress] = useState<WorkflowUpdate | null>(null);
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [wsState, setWsState] = useState<WebSocketState>({
    isConnected: false,
    connectionStatus: 'disconnected',
    reconnectCount: 0,
    lastError: null
  });

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Get WebSocket URL
  const getWsUrl = useCallback(() => {
    const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const wsUrl = baseUrl.replace(/^http/, 'ws');
    // Use the v2 signals endpoint which exists in the backend
    return `${wsUrl}/ws/v2/signals/${symbol}`;
  }, [symbol]);

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: AgentWebSocketMessage = JSON.parse(event.data);

      switch (message.type) {
        case 'agent_signal':
          const signal = message.data as AgentSignal;
          setAgents(prev => ({
            ...prev,
            [signal.agent_name]: signal
          }));
          onSignal?.(signal);
          break;

        case 'workflow_update':
          const update = message.data as WorkflowUpdate;
          setWorkflowProgress(update);
          onWorkflowUpdate?.(update);
          break;

        case 'agent_status':
          const status = message.data as AgentStatus;
          setAgentStatuses(prev => ({
            ...prev,
            [status.name]: status
          }));
          onStatusUpdate?.(status);
          break;

        case 'error':
          const error = new Error(message.data.message);
          setWsState(prev => ({ ...prev, lastError: error }));
          onError?.(error);
          break;
      }
    } catch (error) {
      logger.error('Failed to parse WebSocket message:', error);
      const err = error instanceof Error ? error : new Error('Failed to parse message');
      onError?.(err);
    }
  }, [onSignal, onWorkflowUpdate, onStatusUpdate, onError]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setWsState(prev => ({ ...prev, connectionStatus: 'connecting' }));

    try {
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        logger.info('Agent WebSocket connected');
        setWsState({
          isConnected: true,
          connectionStatus: 'connected',
          reconnectCount: 0,
          lastError: null
        });

        // Start ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000); // Ping every 30 seconds
      };

      ws.onmessage = handleMessage;

      ws.onerror = (event) => {
        const error = new Error('WebSocket error occurred');
        logger.error('Agent WebSocket error:', event);
        setWsState(prev => ({
          ...prev,
          lastError: error,
          connectionStatus: 'error'
        }));
        onError?.(error);
      };

      ws.onclose = (event) => {
        logger.info('Agent WebSocket closed:', event.code, event.reason);

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        setWsState(prev => ({
          ...prev,
          isConnected: false,
          connectionStatus: 'disconnected'
        }));

        // Attempt reconnection if not manually closed
        if (enabled && event.code !== 1000 && wsState.reconnectCount < reconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            setWsState(prev => ({
              ...prev,
              reconnectCount: prev.reconnectCount + 1
            }));
            connect();
          }, reconnectDelay);
        }
      };

    } catch (error) {
      const err = error instanceof Error ? error : new Error('Failed to connect');
      logger.error('Failed to create WebSocket:', err);
      setWsState(prev => ({
        ...prev,
        lastError: err,
        connectionStatus: 'error'
      }));
      onError?.(err);
    }
  }, [enabled, getWsUrl, handleMessage, onError, reconnectAttempts, reconnectDelay, wsState.reconnectCount]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }

    setWsState({
      isConnected: false,
      connectionStatus: 'disconnected',
      reconnectCount: 0,
      lastError: null
    });
  }, []);

  // Send message to WebSocket
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  // Subscribe to specific agent updates
  const subscribeToAgent = useCallback((agentName: string) => {
    return sendMessage({
      type: 'subscribe',
      agent: agentName
    });
  }, [sendMessage]);

  // Unsubscribe from specific agent updates
  const unsubscribeFromAgent = useCallback((agentName: string) => {
    return sendMessage({
      type: 'unsubscribe',
      agent: agentName
    });
  }, [sendMessage]);

  // Effect to manage connection lifecycle
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, symbol]); // Only reconnect on symbol change

  // Get latest signals for each agent
  const getLatestSignals = useCallback(() => {
    return Object.values(agents).sort((a, b) =>
      new Date(b.metadata.timestamp).getTime() - new Date(a.metadata.timestamp).getTime()
    );
  }, [agents]);

  // Get workflow completion percentage
  const getWorkflowCompletion = useCallback(() => {
    return workflowProgress?.progress || 0;
  }, [workflowProgress]);

  // Check if workflow is in progress
  const isWorkflowActive = useCallback(() => {
    return workflowProgress && workflowProgress.stage !== 'complete';
  }, [workflowProgress]);

  return {
    // Connection state
    isConnected: wsState.isConnected,
    connectionStatus: wsState.connectionStatus,
    reconnectCount: wsState.reconnectCount,
    lastError: wsState.lastError,

    // Data
    agents,
    workflowProgress,
    agentStatuses,

    // Derived data
    latestSignals: getLatestSignals(),
    workflowCompletion: getWorkflowCompletion(),
    isWorkflowActive: isWorkflowActive(),

    // Actions
    connect,
    disconnect,
    sendMessage,
    subscribeToAgent,
    unsubscribeFromAgent,

    // Utilities
    clearAgents: () => setAgents({}),
    clearWorkflow: () => setWorkflowProgress(null)
  };
}
