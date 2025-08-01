"""
RAG Query MCP Server
Provides unified access to all RAG services
Issue #191: MCP-2: Build RAG Query MCP Server
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import time
import io

# Import our RAG components
import sys
sys.path.append('..')
from agents.rag.historical_market_context_rag import HistoricalMarketContextRAG
from agents.rag.real_time_sentiment_analyzer import RealTimeSentimentAnalyzer
from agents.rag.options_flow_intelligence_rag import OptionsFlowIntelligenceRAG
from agents.rag.technical_pattern_success_rag import TechnicalPatternSuccessRAG
from agents.rag.risk_event_prediction_rag import RiskEventPredictionRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService(Enum):
    """Available RAG services"""
    HISTORICAL_CONTEXT = "historical_context"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    OPTIONS_FLOW = "options_flow"
    TECHNICAL_PATTERNS = "technical_patterns"
    RISK_PREDICTION = "risk_prediction"
    ALL = "all"


class QueryMode(Enum):
    """Query execution modes"""
    FAST = "fast"  # Return first available result
    COMPREHENSIVE = "comprehensive"  # Query all relevant services
    CONSENSUS = "consensus"  # Get consensus from multiple services
    STREAMING = "streaming"  # Stream results as they arrive


@dataclass
class RAGQuery:
    """Structured RAG query"""
    id: str
    query: str
    services: List[RAGService]
    mode: QueryMode
    context: Dict[str, Any]
    filters: Dict[str, Any]
    timestamp: datetime
    timeout_ms: int = 5000

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'services': [s.value for s in self.services],
            'mode': self.mode.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RAGResult:
    """Result from a RAG service"""
    service: RAGService
    query_id: str
    content: Any
    confidence: float
    metadata: Dict[str, Any]
    latency_ms: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'service': self.service.value,
            'query_id': self.query_id,
            'content': self.content,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'latency_ms': self.latency_ms,
            'timestamp': self.timestamp.isoformat()
        }


class QueryCache:
    """LRU cache for RAG queries"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired"""
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl:
                self.access_times[key] = time.time()
                return self.cache[key]
            else:
                del self.cache[key]
                del self.access_times[key]
        return None

    def set(self, key: str, value: Dict[str, Any]):
        """Cache a result"""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()


class RAGQueryMCP:
    """
    MCP Server providing unified access to all RAG services
    Handles query routing, caching, and result aggregation
    """

    def __init__(self):
        self.app = FastAPI(title="RAG Query MCP Server")

        # Initialize RAG services
        self.services = {
            RAGService.HISTORICAL_CONTEXT: HistoricalMarketContextRAG(),
            RAGService.SENTIMENT_ANALYSIS: RealTimeSentimentAnalyzer(),
            RAGService.OPTIONS_FLOW: OptionsFlowIntelligenceRAG(),
            RAGService.TECHNICAL_PATTERNS: TechnicalPatternSuccessRAG(),
            RAGService.RISK_PREDICTION: RiskEventPredictionRAG()
        }

        # Query cache
        self.cache = QueryCache()

        # Active queries
        self.active_queries: Dict[str, RAGQuery] = {}

        # Metrics
        self.metrics = defaultdict(lambda: {
            'queries': 0,
            'hits': 0,
            'avg_latency_ms': 0,
            'errors': 0
        })

        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "service": "RAG Query MCP",
                "status": "active",
                "available_services": [s.value for s in RAGService],
                "metrics": dict(self.metrics)
            }

        @self.app.get("/tools")
        async def list_tools():
            """List available RAG query tools"""
            return {
                "tools": [
                    {
                        "name": "query",
                        "description": "Execute a RAG query across one or more services",
                        "parameters": {
                            "query": "string (required)",
                            "services": "array[string] (optional, default: all)",
                            "mode": "string (fast/comprehensive/consensus/streaming)",
                            "context": "object (optional context data)",
                            "filters": "object (optional filters)",
                            "timeout_ms": "integer (optional, default: 5000)"
                        }
                    },
                    {
                        "name": "search_similar",
                        "description": "Find similar queries and their results",
                        "parameters": {
                            "query": "string",
                            "limit": "integer (default: 10)",
                            "threshold": "number (similarity threshold 0-1)"
                        }
                    },
                    {
                        "name": "explain",
                        "description": "Get explanation of RAG results",
                        "parameters": {
                            "query_id": "string",
                            "service": "string (optional)"
                        }
                    },
                    {
                        "name": "feedback",
                        "description": "Provide feedback on query results",
                        "parameters": {
                            "query_id": "string",
                            "rating": "number (1-5)",
                            "comments": "string (optional)"
                        }
                    }
                ]
            }

        @self.app.post("/call")
        async def call_tool(request: Dict[str, Any], background_tasks: BackgroundTasks):
            """Execute a RAG tool"""
            tool_name = request.get("tool")
            params = request.get("parameters", {})

            try:
                if tool_name == "query":
                    return await self._execute_query(params)
                elif tool_name == "search_similar":
                    return await self._search_similar(params)
                elif tool_name == "explain":
                    return await self._explain_results(params)
                elif tool_name == "feedback":
                    return await self._process_feedback(params)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return {"error": str(e), "tool": tool_name}

        @self.app.get("/stream/{query_id}")
        async def stream_results(query_id: str):
            """Stream query results as they arrive"""
            if query_id not in self.active_queries:
                raise HTTPException(status_code=404, detail="Query not found")

            return StreamingResponse(
                self._stream_query_results(query_id),
                media_type="text/event-stream"
            )

        @self.app.get("/services")
        async def list_services():
            """List all available RAG services and their status"""
            service_status = {}

            for service_type, service in self.services.items():
                try:
                    # Simple health check
                    status = "active" if hasattr(service, 'retrieve') else "inactive"
                    service_status[service_type.value] = {
                        "status": status,
                        "metrics": self.metrics.get(service_type.value, {}),
                        "capabilities": self._get_service_capabilities(service_type)
                    }
                except Exception as e:
                    service_status[service_type.value] = {
                        "status": "error",
                        "error": str(e)
                    }

            return service_status

    async def _execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG query"""
        import uuid

        query_text = params.get('query')
        if not query_text:
            raise ValueError("Query is required")

        # Parse services
        service_names = params.get('services', ['all'])
        if 'all' in service_names:
            services = list(RAGService)
            services.remove(RAGService.ALL)
        else:
            services = [RAGService(name) for name in service_names]

        # Create query object
        query = RAGQuery(
            id=str(uuid.uuid4()),
            query=query_text,
            services=services,
            mode=QueryMode(params.get('mode', 'comprehensive')),
            context=params.get('context', {}),
            filters=params.get('filters', {}),
            timestamp=datetime.now(),
            timeout_ms=params.get('timeout_ms', 5000)
        )

        # Check cache
        cache_key = self._get_cache_key(query)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query.id}")
            return cached_result

        # Store active query
        self.active_queries[query.id] = query

        try:
            # Execute based on mode
            if query.mode == QueryMode.FAST:
                result = await self._execute_fast_query(query)
            elif query.mode == QueryMode.COMPREHENSIVE:
                result = await self._execute_comprehensive_query(query)
            elif query.mode == QueryMode.CONSENSUS:
                result = await self._execute_consensus_query(query)
            elif query.mode == QueryMode.STREAMING:
                # For streaming, return immediately with query ID
                return {
                    "query_id": query.id,
                    "status": "streaming",
                    "stream_url": f"/stream/{query.id}"
                }

            # Cache result
            self.cache.set(cache_key, result)

            return result

        finally:
            # Clean up
            if query.id in self.active_queries:
                del self.active_queries[query.id]

    async def _execute_fast_query(self, query: RAGQuery) -> Dict[str, Any]:
        """Execute query and return first available result"""
        start_time = time.time()

        # Create tasks for all services
        tasks = []
        for service_type in query.services:
            if service_type in self.services:
                task = asyncio.create_task(
                    self._query_service(service_type, query)
                )
                tasks.append((service_type, task))

        # Wait for first result
        for service_type, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=query.timeout_ms/1000)
                if result and result.confidence > 0.5:
                    # Cancel remaining tasks
                    for _, other_task in tasks:
                        if not other_task.done():
                            other_task.cancel()

                    return {
                        "query_id": query.id,
                        "mode": "fast",
                        "results": [result.to_dict()],
                        "total_latency_ms": (time.time() - start_time) * 1000
                    }
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in fast query for {service_type}: {e}")

        return {
            "query_id": query.id,
            "mode": "fast",
            "results": [],
            "error": "No results available within timeout"
        }

    async def _execute_comprehensive_query(self, query: RAGQuery) -> Dict[str, Any]:
        """Execute query across all specified services"""
        start_time = time.time()
        results = []

        # Query all services in parallel
        tasks = []
        for service_type in query.services:
            if service_type in self.services:
                task = asyncio.create_task(
                    self._query_service(service_type, query)
                )
                tasks.append((service_type, task))

        # Collect all results
        for service_type, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=query.timeout_ms/1000)
                if result:
                    results.append(result.to_dict())
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {service_type}")
            except Exception as e:
                logger.error(f"Error querying {service_type}: {e}")
                self.metrics[service_type.value]['errors'] += 1

        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            "query_id": query.id,
            "mode": "comprehensive",
            "results": results,
            "services_queried": len(tasks),
            "services_responded": len(results),
            "total_latency_ms": (time.time() - start_time) * 1000
        }

    async def _execute_consensus_query(self, query: RAGQuery) -> Dict[str, Any]:
        """Execute query and get consensus from multiple services"""
        # First get comprehensive results
        comprehensive_result = await self._execute_comprehensive_query(query)
        results = comprehensive_result['results']

        if not results:
            return comprehensive_result

        # Calculate consensus
        consensus = self._calculate_consensus(results)

        return {
            "query_id": query.id,
            "mode": "consensus",
            "consensus": consensus,
            "individual_results": results,
            "total_latency_ms": comprehensive_result['total_latency_ms']
        }

    async def _query_service(self, service_type: RAGService, query: RAGQuery) -> Optional[RAGResult]:
        """Query a specific RAG service"""
        start_time = time.time()
        service = self.services.get(service_type)

        if not service:
            return None

        try:
            # Update metrics
            self.metrics[service_type.value]['queries'] += 1

            # Execute query based on service type
            if service_type == RAGService.HISTORICAL_CONTEXT:
                result = await service.retrieve(
                    query.query,
                    symbol=query.context.get('symbol'),
                    lookback_days=query.filters.get('lookback_days', 365)
                )
            elif service_type == RAGService.SENTIMENT_ANALYSIS:
                result = await service.analyze_sentiment(
                    symbols=query.context.get('symbols', []),
                    keywords=[query.query]
                )
            elif service_type == RAGService.OPTIONS_FLOW:
                result = await service.analyze_flow(
                    symbol=query.context.get('symbol', 'SPY')
                )
            elif service_type == RAGService.TECHNICAL_PATTERNS:
                result = await service.find_patterns(
                    symbol=query.context.get('symbol', 'SPY'),
                    pattern_type=query.filters.get('pattern_type')
                )
            elif service_type == RAGService.RISK_PREDICTION:
                result = await service.predict_risks(
                    market_data=query.context.get('market_data', {})
                )
            else:
                return None

            # Create RAGResult
            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            self.metrics[service_type.value]['hits'] += 1
            avg_latency = self.metrics[service_type.value]['avg_latency_ms']
            hits = self.metrics[service_type.value]['hits']
            self.metrics[service_type.value]['avg_latency_ms'] = (
                (avg_latency * (hits - 1) + latency_ms) / hits
            )

            return RAGResult(
                service=service_type,
                query_id=query.id,
                content=result,
                confidence=result.get('confidence', 0.8) if isinstance(result, dict) else 0.8,
                metadata={
                    'query': query.query,
                    'context': query.context
                },
                latency_ms=latency_ms,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error querying {service_type}: {e}")
            self.metrics[service_type.value]['errors'] += 1
            return None

    async def _stream_query_results(self, query_id: str) -> AsyncGenerator[str, None]:
        """Stream query results as they arrive"""
        query = self.active_queries.get(query_id)
        if not query:
            yield f"data: {json.dumps({'error': 'Query not found'})}\n\n"
            return

        # Query services and stream results
        tasks = []
        for service_type in query.services:
            if service_type in self.services:
                task = asyncio.create_task(
                    self._query_service(service_type, query)
                )
                tasks.append((service_type, task))

        # Stream results as they complete
        for service_type, task in tasks:
            try:
                result = await task
                if result:
                    yield f"data: {json.dumps(result.to_dict())}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'service': service_type.value})}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'event': 'complete', 'query_id': query_id})}\n\n"

    async def _search_similar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for similar queries in cache"""
        query_text = params.get('query')
        limit = params.get('limit', 10)
        threshold = params.get('threshold', 0.7)

        if not query_text:
            raise ValueError("Query is required")

        # Simple similarity search in cache
        # In production, would use vector similarity
        similar_results = []

        for key, cached in self.cache.cache.items():
            if 'results' in cached and cached['results']:
                # Simple text similarity (would use embeddings in production)
                similarity = self._calculate_text_similarity(
                    query_text,
                    cached['results'][0].get('metadata', {}).get('query', '')
                )

                if similarity >= threshold:
                    similar_results.append({
                        'query': cached['results'][0].get('metadata', {}).get('query', ''),
                        'similarity': similarity,
                        'results': cached['results'][:3]  # Top 3 results
                    })

        # Sort by similarity
        similar_results.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            "query": query_text,
            "similar_queries": similar_results[:limit],
            "total_found": len(similar_results)
        }

    async def _explain_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Provide explanation for query results"""
        query_id = params.get('query_id')
        service = params.get('service')

        # In production, would retrieve actual query results
        # For now, return generic explanation
        return {
            "query_id": query_id,
            "service": service,
            "explanation": {
                "methodology": "The RAG service uses semantic search to find relevant context",
                "confidence_factors": [
                    "Similarity score between query and retrieved documents",
                    "Recency of the information",
                    "Source reliability"
                ],
                "limitations": [
                    "Limited to historical data available",
                    "May not capture real-time events"
                ]
            }
        }

    async def _process_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback on query results"""
        query_id = params.get('query_id')
        rating = params.get('rating')
        comments = params.get('comments', '')

        if not all([query_id, rating]):
            raise ValueError("Query ID and rating are required")

        # In production, would store feedback for model improvement
        logger.info(f"Feedback for query {query_id}: rating={rating}, comments={comments}")

        return {
            "status": "feedback_recorded",
            "query_id": query_id,
            "rating": rating
        }

    def _get_cache_key(self, query: RAGQuery) -> str:
        """Generate cache key for query"""
        # Include query text, services, and key filters
        key_parts = [
            query.query,
            ','.join(sorted([s.value for s in query.services])),
            json.dumps(query.filters, sort_keys=True)
        ]
        return ':'.join(key_parts)

    def _calculate_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus from multiple RAG results"""
        if not results:
            return {"consensus": None, "confidence": 0}

        # Weight by confidence
        total_confidence = sum(r['confidence'] for r in results)

        if total_confidence == 0:
            return {"consensus": None, "confidence": 0}

        # For now, return highest confidence result as consensus
        # In production, would do more sophisticated aggregation
        best_result = max(results, key=lambda x: x['confidence'])

        return {
            "consensus": best_result['content'],
            "confidence": best_result['confidence'],
            "agreement_score": self._calculate_agreement(results),
            "sources": len(results)
        }

    def _calculate_agreement(self, results: List[Dict[str, Any]]) -> float:
        """Calculate agreement score between results"""
        if len(results) < 2:
            return 1.0

        # Simple agreement based on confidence variance
        confidences = [r['confidence'] for r in results]
        variance = np.var(confidences)

        # Lower variance = higher agreement
        return max(0, 1 - variance)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        # In production, would use embeddings
        # For now, use simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _get_service_capabilities(self, service_type: RAGService) -> List[str]:
        """Get capabilities of a specific service"""
        capabilities_map = {
            RAGService.HISTORICAL_CONTEXT: [
                "Historical pattern matching",
                "Similar market conditions",
                "Past performance analysis"
            ],
            RAGService.SENTIMENT_ANALYSIS: [
                "Real-time sentiment scoring",
                "Multi-source aggregation",
                "Trend detection"
            ],
            RAGService.OPTIONS_FLOW: [
                "Smart money detection",
                "Unusual activity alerts",
                "Flow direction analysis"
            ],
            RAGService.TECHNICAL_PATTERNS: [
                "Pattern recognition",
                "Success rate calculation",
                "Optimal parameters"
            ],
            RAGService.RISK_PREDICTION: [
                "Risk event forecasting",
                "Volatility prediction",
                "Crash detection"
            ]
        }

        return capabilities_map.get(service_type, [])


# Demo function
async def demo_rag_query_mcp():
    """Demonstrate RAG Query MCP functionality"""
    import uvicorn

    logger.info("Starting RAG Query MCP Server demo...")

    # Create server
    server = RAGQueryMCP()

    # Run server
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=8192,
        log_level="info"
    )

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(demo_rag_query_mcp())
