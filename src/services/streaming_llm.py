"""
Streaming LLM Service
Provides real-time streaming responses from AI providers for better UX
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import openai
from anthropic import AsyncAnthropic
from fastapi import Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Represents a chunk of streamed content"""

    content: str
    provider: str
    model: str
    timestamp: datetime
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class StreamingLLMService:
    """
    Service for streaming LLM responses with support for multiple providers
    """

    def __init__(self):
        # Initialize clients with async support
        self.openai_client = openai.AsyncOpenAI()
        self.anthropic_client = AsyncAnthropic()

        # Streaming configuration
        self.chunk_size = 10  # Characters to buffer before yielding
        self.stream_timeout = 60  # Seconds

        # Metrics
        self.streaming_metrics = {
            "total_streams": 0,
            "active_streams": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0,
        }

    async def stream_market_analysis(
        self, symbol: str, provider: str = "openai", include_metadata: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Stream market analysis with real-time updates
        """
        self.streaming_metrics["total_streams"] += 1
        self.streaming_metrics["active_streams"] += 1

        try:
            prompt = f"""Analyze {symbol} with the following structure:
1. Current Market Conditions
2. Technical Analysis
3. Key Support/Resistance Levels
4. Trading Recommendation
5. Risk Factors

Provide detailed analysis with specific price levels and reasoning."""

            if provider == "openai":
                async for chunk in self._stream_openai(prompt, model="gpt-4-turbo-preview"):
                    yield chunk
            elif provider == "anthropic":
                async for chunk in self._stream_anthropic(prompt, model="claude-3-opus-20240229"):
                    yield chunk
            elif provider == "grok":
                async for chunk in self._stream_grok(prompt, symbol=symbol):
                    yield chunk
            else:
                yield json.dumps({"error": f"Unknown provider: {provider}"})

        finally:
            self.streaming_metrics["active_streams"] -= 1

    async def stream_trading_decision(
        self, symbol: str, context: Dict[str, Any], provider: str = "openai"
    ) -> AsyncGenerator[str, None]:
        """
        Stream trading decision analysis with progressive detail
        """
        prompt = f"""Based on the following context for {symbol}, provide a trading decision:

Market Context:
{json.dumps(context, indent=2)}

Structure your response as:
1. Quick Decision (BUY/SELL/HOLD)
2. Confidence Level
3. Detailed Reasoning
4. Entry/Exit Points
5. Risk Management

Stream the response progressively, starting with the decision."""

        if provider == "openai":
            async for chunk in self._stream_openai_with_tools(prompt, context):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt):
                yield chunk
        else:
            yield json.dumps({"error": f"Provider {provider} not supported for trading decisions"})

    async def _stream_openai(
        self, prompt: str, model: str = "gpt-4-turbo-preview", temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream from OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional trading analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            buffer = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    buffer += content

                    # Yield complete words/sentences for better UX
                    if len(buffer) >= self.chunk_size or content.endswith((".", "!", "?", "\n")):
                        yield json.dumps(
                            {
                                "type": "content",
                                "content": buffer,
                                "provider": "openai",
                                "model": model,
                            }
                        ) + "\n"
                        buffer = ""

                # Check for usage stats in final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    self.streaming_metrics["total_tokens"] += chunk.usage.total_tokens

            # Yield any remaining buffer
            if buffer:
                yield json.dumps(
                    {"type": "content", "content": buffer, "provider": "openai", "model": model}
                ) + "\n"

            # Final message
            yield json.dumps({"type": "done", "provider": "openai", "model": model}) + "\n"

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield json.dumps({"type": "error", "error": str(e), "provider": "openai"}) + "\n"

    async def _stream_openai_with_tools(
        self, prompt: str, context: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream from OpenAI with tool usage"""
        try:
            from services.mcp_tools import get_mcp_tools_for_openai

            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading analyst with access to market tools.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=get_mcp_tools_for_openai(),
                stream=True,
            )

            current_tool_call = None
            async for chunk in response:
                # Handle tool calls
                if chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function.name:
                            current_tool_call = tool_call.function.name
                            yield json.dumps(
                                {"type": "tool_start", "tool": current_tool_call}
                            ) + "\n"

                        if tool_call.function.arguments:
                            # Stream tool arguments as they come
                            yield json.dumps(
                                {"type": "tool_args", "content": tool_call.function.arguments}
                            ) + "\n"

                # Handle regular content
                if chunk.choices[0].delta.content:
                    yield json.dumps(
                        {"type": "content", "content": chunk.choices[0].delta.content}
                    ) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            logger.error(f"OpenAI tools streaming error: {e}")
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    async def _stream_anthropic(
        self, prompt: str, model: str = "claude-3-opus-20240229"
    ) -> AsyncGenerator[str, None]:
        """Stream from Anthropic"""
        try:
            async with self.anthropic_client.messages.stream(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7,
            ) as stream:
                buffer = ""
                async for chunk in stream:
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        content = chunk.delta.text
                        buffer += content

                        if len(buffer) >= self.chunk_size:
                            yield json.dumps(
                                {
                                    "type": "content",
                                    "content": buffer,
                                    "provider": "anthropic",
                                    "model": model,
                                }
                            ) + "\n"
                            buffer = ""

                # Yield remaining buffer
                if buffer:
                    yield json.dumps(
                        {
                            "type": "content",
                            "content": buffer,
                            "provider": "anthropic",
                            "model": model,
                        }
                    ) + "\n"

                yield json.dumps({"type": "done", "provider": "anthropic", "model": model}) + "\n"

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield json.dumps({"type": "error", "error": str(e), "provider": "anthropic"}) + "\n"

    async def _stream_grok(self, prompt: str, symbol: str) -> AsyncGenerator[str, None]:
        """Stream from Grok with real-time capabilities"""
        try:
            # Grok uses OpenAI-compatible API
            import os

            headers = {
                "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
                "Content-Type": "application/json",
            }

            data = {
                "model": "grok-4",
                "messages": [
                    {"role": "system", "content": f"Analyze {symbol} with real-time market data."},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "temperature": 0.7,
            }

            for attempt in range(3):
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=self.stream_timeout,
                    ) as response:
                        if response.status_code == 200:
                            buffer = ""
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    if line == "data: [DONE]":
                                        break

                                    try:
                                        chunk_data = json.loads(line[6:])
                                        if chunk_data["choices"][0]["delta"].get("content"):
                                            content = chunk_data["choices"][0]["delta"]["content"]
                                            buffer += content

                                            if len(buffer) >= self.chunk_size:
                                                yield json.dumps(
                                                    {
                                                        "type": "content",
                                                        "content": buffer,
                                                        "provider": "grok",
                                                        "model": "grok-4",
                                                    }
                                                ) + "\n"
                                                buffer = ""
                                    except json.JSONDecodeError:
                                        continue

                            if buffer:
                                yield json.dumps(
                                    {
                                        "type": "content",
                                        "content": buffer,
                                        "provider": "grok",
                                        "model": "grok-4",
                                    }
                                ) + "\n"

                            yield json.dumps(
                                {"type": "done", "provider": "grok", "model": "grok-4"}
                            ) + "\n"

                            return  # Successful stream, exit function

                        else:
                            error_content = await response.aread()
                            error_msg = error_content.decode()
                            logger.error(f"Grok API error {response.status_code}: {error_msg}")

                            try:
                                error_json = json.loads(error_msg)
                                if (
                                    response.status_code == 500
                                    and error_json.get("error", {}).get("message") == "Overloaded"
                                ):
                                    delay = 2**attempt
                                    logger.info(f"Overloaded, retrying after {delay} seconds...")
                                    await asyncio.sleep(delay)
                                    continue
                            except json.JSONDecodeError:
                                pass

                            # If not retryable, yield error
                            yield json.dumps(
                                {"type": "error", "error": error_msg, "provider": "grok"}
                            ) + "\n"
                            return

            # If all retries fail
            yield json.dumps(
                {"type": "error", "error": "Max retries exceeded for Grok API", "provider": "grok"}
            ) + "\n"

        except Exception as e:
            logger.error(f"Grok streaming error: {e}")
            yield json.dumps({"type": "error", "error": str(e), "provider": "grok"}) + "\n"

    async def parallel_stream_analysis(
        self, symbol: str, providers: List[str] = ["openai", "anthropic", "grok"]
    ) -> AsyncGenerator[str, None]:
        """
        Stream analysis from multiple providers in parallel
        """
        # Create queues for each provider
        queues = {provider: asyncio.Queue() for provider in providers}
        tasks = []

        # Start streaming from each provider
        for provider in providers:
            task = asyncio.create_task(self._stream_to_queue(symbol, provider, queues[provider]))
            tasks.append(task)

        # Merge streams
        try:
            while True:
                # Check each queue for data
                for provider, queue in queues.items():
                    try:
                        chunk = queue.get_nowait()
                        if chunk is None:  # Provider finished
                            providers.remove(provider)
                        else:
                            yield chunk
                    except asyncio.QueueEmpty:
                        continue

                if not providers:  # All providers finished
                    break

                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _stream_to_queue(self, symbol: str, provider: str, queue: asyncio.Queue):
        """Helper to stream provider output to queue"""
        try:
            async for chunk in self.stream_market_analysis(symbol, provider):
                await queue.put(chunk)
        finally:
            await queue.put(None)  # Signal completion

    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        return {
            **self.streaming_metrics,
            "avg_chunk_size": self.chunk_size,
            "providers_available": ["openai", "anthropic", "grok"],
        }


# Singleton instance
streaming_service = StreamingLLMService()


# FastAPI streaming endpoint helper
def create_streaming_response(
    generator: AsyncGenerator[str, None], media_type: str = "text/event-stream"
) -> StreamingResponse:
    """Create a FastAPI streaming response"""
    return StreamingResponse(
        generator,
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
