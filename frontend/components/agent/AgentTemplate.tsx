import React, { useEffect } from 'react';
import { useAgentRegistry } from './AgentOrchestrator';

/**
 * AgentTemplate is a base pattern for agentic UI components.
 * Each agent registers itself and exposes an API (e.g., send/receive messages, refresh, etc).
 */
export function AgentTemplate({ name, children, onMessage }: { name: string; children?: React.ReactNode; onMessage?: (msg: any) => void }) {
  const { registerAgent } = useAgentRegistry();

  useEffect(() => {
    // Register this agent with its API
    registerAgent(name, { onMessage });
    // Optionally: unregister on unmount
    return () => registerAgent(name, null);
    // eslint-disable-next-line
  }, [name, onMessage]);

  return <div>{children}</div>;
}
