"""
Vector Memory Service - Advanced RAG Implementation
Uses embeddings and vector search for intelligent trading memory
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# For advanced vector operations
import faiss
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of trading memories"""

    TRADE_SETUP = "trade_setup"
    MARKET_CONDITION = "market_condition"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_EVENT = "risk_event"
    STRATEGY_OUTCOME = "strategy_outcome"
    AI_DECISION = "ai_decision"


@dataclass
class TradingMemory:
    """Structure for a trading memory"""

    id: str
    type: MemoryType
    symbol: str
    timestamp: datetime
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None


class VectorMemoryService:
    """
    Advanced vector-based memory system for trading intelligence
    Supports multiple vector stores and intelligent retrieval
    """

    def __init__(
        self,
        collection_name: str = "trading_memories",
        persist_directory: str = "./chroma_db",
        use_faiss: bool = True,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_faiss = use_faiss

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        )

        # Initialize vector stores
        self._init_vector_stores()

        # Memory cache for fast access
        self.memory_cache: Dict[str, TradingMemory] = {}

        # Performance metrics
        self.metrics = {
            "total_memories": 0,
            "queries_processed": 0,
            "avg_query_time": 0.0,
            "cache_hits": 0,
        }

    def _init_vector_stores(self):
        """Initialize vector storage backends"""
        # Chroma for persistent storage
        self.chroma_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        # FAISS for high-performance similarity search
        if self.use_faiss:
            self.faiss_index = None
            self.faiss_id_map = {}
            self._init_faiss_index()

    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            # Load existing index if available
            index_path = f"{self.persist_directory}/faiss_index.pkl"
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    data = pickle.load(f)
                    self.faiss_index = data["index"]
                    self.faiss_id_map = data["id_map"]
                logger.info(f"Loaded FAISS index with {len(self.faiss_id_map)} vectors")
            else:
                # Create new index (384 dimensions for text-embedding-3-small)
                self.faiss_index = faiss.IndexFlatL2(384)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.use_faiss = False

    def _save_faiss_index(self):
        """Persist FAISS index to disk"""
        if self.use_faiss and self.faiss_index:
            try:
                index_path = f"{self.persist_directory}/faiss_index.pkl"
                os.makedirs(self.persist_directory, exist_ok=True)
                with open(index_path, "wb") as f:
                    pickle.dump({"index": self.faiss_index, "id_map": self.faiss_id_map}, f)
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")

    async def add_memory(
        self, memory_type: MemoryType, symbol: str, content: str, metadata: Dict[str, Any]
    ) -> TradingMemory:
        """Add a new trading memory"""
        try:
            # Generate unique ID
            memory_id = self._generate_memory_id(symbol, content)

            # Create memory object
            memory = TradingMemory(
                id=memory_id,
                type=memory_type,
                symbol=symbol,
                timestamp=datetime.now(),
                content=content,
                metadata=metadata,
            )

            # Generate embedding
            embedding = await self._generate_embedding(content)
            memory.embedding = embedding

            # Store in Chroma
            doc = Document(
                page_content=content,
                metadata={
                    "id": memory_id,
                    "type": memory_type.value,
                    "symbol": symbol,
                    "timestamp": memory.timestamp.isoformat(),
                    **metadata,
                },
            )
            self.chroma_store.add_documents([doc])

            # Add to FAISS if enabled
            if self.use_faiss and embedding:
                self._add_to_faiss(memory_id, embedding)

            # Update cache
            self.memory_cache[memory_id] = memory
            self.metrics["total_memories"] += 1

            logger.info(f"Added memory {memory_id} for {symbol}")
            return memory

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    async def search_memories(
        self,
        query: str,
        symbol: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        min_relevance: float = 0.7,
    ) -> List[TradingMemory]:
        """Search for relevant memories using vector similarity"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Build metadata filter
            filter_dict = {}
            if symbol:
                filter_dict["symbol"] = symbol
            if memory_type:
                filter_dict["type"] = memory_type.value

            # Search in Chroma
            results = self.chroma_store.similarity_search_with_score(
                query,
                k=limit * 2,  # Get extra results for filtering
                filter=filter_dict if filter_dict else None,
            )

            # Convert to TradingMemory objects
            memories = []
            for doc, score in results:
                # Convert distance to similarity score (0-1)
                relevance = 1 / (1 + score)

                if relevance >= min_relevance:
                    memory = self._doc_to_memory(doc, relevance)
                    memories.append(memory)

            # Sort by relevance and limit
            memories.sort(key=lambda m: m.relevance_score, reverse=True)
            memories = memories[:limit]

            # Update metrics
            query_time = asyncio.get_event_loop().time() - start_time
            self._update_query_metrics(query_time)

            logger.info(f"Found {len(memories)} relevant memories for query")
            return memories

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def get_similar_trades(
        self, trade_setup: Dict[str, Any], limit: int = 5
    ) -> List[Tuple[TradingMemory, Dict[str, Any]]]:
        """Find similar historical trades with outcomes"""
        # Create a descriptive query from trade setup
        query_parts = [
            f"Symbol: {trade_setup.get('symbol', 'N/A')}",
            f"Action: {trade_setup.get('action', 'N/A')}",
            f"Market conditions: {trade_setup.get('market_conditions', {})}",
            f"Technical indicators: {trade_setup.get('indicators', {})}",
            f"Risk factors: {trade_setup.get('risk_factors', [])}",
        ]
        query = " ".join(query_parts)

        # Search for similar setups
        similar_memories = await self.search_memories(
            query=query, memory_type=MemoryType.TRADE_SETUP, limit=limit, min_relevance=0.6
        )

        # Get outcomes for similar trades
        results = []
        for memory in similar_memories:
            # Look for corresponding outcome
            outcome_query = f"Trade outcome for {memory.id}"
            outcomes = await self.search_memories(
                query=outcome_query, memory_type=MemoryType.STRATEGY_OUTCOME, limit=1
            )

            outcome = outcomes[0].metadata if outcomes else {"status": "unknown"}
            results.append((memory, outcome))

        return results

    async def analyze_pattern_success(
        self, pattern_name: str, timeframe: str = "30d"
    ) -> Dict[str, Any]:
        """Analyze historical success rate of a pattern"""
        # Search for all instances of this pattern
        pattern_memories = await self.search_memories(
            query=f"Pattern: {pattern_name}", memory_type=MemoryType.PATTERN_RECOGNITION, limit=100
        )

        # Analyze outcomes
        total_trades = len(pattern_memories)
        successful_trades = 0
        total_return = 0.0

        for memory in pattern_memories:
            outcome = memory.metadata.get("outcome", {})
            if outcome.get("profitable", False):
                successful_trades += 1
                total_return += outcome.get("return_pct", 0)

        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0

        return {
            "pattern": pattern_name,
            "total_occurrences": total_trades,
            "success_rate": success_rate,
            "average_return": avg_return,
            "timeframe": timeframe,
            "confidence": min(0.95, success_rate + 0.1) if total_trades >= 10 else 0.5,
        }

    async def get_market_context(self, symbol: str, lookback_days: int = 7) -> Dict[str, Any]:
        """Get historical market context for a symbol"""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        # Search for recent market conditions
        market_memories = await self.search_memories(
            query=f"Market conditions for {symbol}",
            symbol=symbol,
            memory_type=MemoryType.MARKET_CONDITION,
            limit=20,
        )

        # Filter by date and aggregate
        recent_memories = [m for m in market_memories if m.timestamp >= cutoff_date]

        if not recent_memories:
            return {"status": "no_recent_data"}

        # Aggregate market conditions
        conditions = {
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "volatility_events": 0,
            "key_levels": [],
            "dominant_trend": None,
        }

        for memory in recent_memories:
            sentiment = memory.metadata.get("sentiment", "neutral").lower()
            if "bullish" in sentiment:
                conditions["bullish_count"] += 1
            elif "bearish" in sentiment:
                conditions["bearish_count"] += 1
            else:
                conditions["neutral_count"] += 1

            if memory.metadata.get("high_volatility", False):
                conditions["volatility_events"] += 1

            # Collect key levels
            if "support" in memory.metadata:
                conditions["key_levels"].append(
                    {"type": "support", "level": memory.metadata["support"]}
                )
            if "resistance" in memory.metadata:
                conditions["key_levels"].append(
                    {"type": "resistance", "level": memory.metadata["resistance"]}
                )

        # Determine dominant trend
        total = len(recent_memories)
        if conditions["bullish_count"] / total > 0.6:
            conditions["dominant_trend"] = "bullish"
        elif conditions["bearish_count"] / total > 0.6:
            conditions["dominant_trend"] = "bearish"
        else:
            conditions["dominant_trend"] = "neutral"

        return conditions

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            # Use async if available, otherwise run in thread
            embedding = await asyncio.to_thread(self.embeddings.embed_query, text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def _add_to_faiss(self, memory_id: str, embedding: List[float]):
        """Add embedding to FAISS index"""
        if not self.use_faiss or not embedding:
            return

        try:
            # Convert to numpy array
            vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

            # Add to index
            idx = self.faiss_index.ntotal
            self.faiss_index.add(vec)
            self.faiss_id_map[idx] = memory_id

            # Save periodically
            if idx % 100 == 0:
                self._save_faiss_index()

        except Exception as e:
            logger.error(f"Failed to add to FAISS: {e}")

    def _generate_memory_id(self, symbol: str, content: str) -> str:
        """Generate unique memory ID"""
        data = f"{symbol}:{content}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _doc_to_memory(self, doc: Document, relevance: float) -> TradingMemory:
        """Convert Langchain Document to TradingMemory"""
        metadata = doc.metadata

        # Check cache first
        memory_id = metadata.get("id", "")
        if memory_id in self.memory_cache:
            memory = self.memory_cache[memory_id]
            memory.relevance_score = relevance
            self.metrics["cache_hits"] += 1
            return memory

        # Create from document
        return TradingMemory(
            id=memory_id,
            type=MemoryType(metadata.get("type", MemoryType.TRADE_SETUP.value)),
            symbol=metadata.get("symbol", ""),
            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
            content=doc.page_content,
            metadata={
                k: v for k, v in metadata.items() if k not in ["id", "type", "symbol", "timestamp"]
            },
            relevance_score=relevance,
        )

    def _update_query_metrics(self, query_time: float):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1

        # Update rolling average
        n = self.metrics["queries_processed"]
        avg = self.metrics["avg_query_time"]
        self.metrics["avg_query_time"] = (avg * (n - 1) + query_time) / n

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "cache_size": len(self.memory_cache),
            "faiss_vectors": self.faiss_index.ntotal if self.use_faiss and self.faiss_index else 0,
        }

    async def clear_old_memories(self, days: int = 90):
        """Clear memories older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)

        # This would need to be implemented based on your vector store's capabilities
        logger.info(f"Clearing memories older than {days} days")

        # Clear cache of old entries
        self.memory_cache = {
            k: v for k, v in self.memory_cache.items() if v.timestamp >= cutoff_date
        }


# Singleton instance
vector_memory = VectorMemoryService()


# High-level API functions
async def remember_trading_decision(
    symbol: str,
    decision: Dict[str, Any],
    market_context: Dict[str, Any],
    agents_involved: List[str],
) -> TradingMemory:
    """Remember a trading decision for future reference"""
    content = f"""
    Trading Decision for {symbol}:
    Action: {decision.get('action', 'HOLD')}
    Confidence: {decision.get('confidence', 0):.2%}
    Reasoning: {decision.get('reasoning', 'N/A')}
    Market Context: {json.dumps(market_context, indent=2)}
    Agents: {', '.join(agents_involved)}
    """

    metadata = {
        "decision": decision,
        "market_context": market_context,
        "agents": agents_involved,
        "timestamp": datetime.now().isoformat(),
    }

    return await vector_memory.add_memory(MemoryType.AI_DECISION, symbol, content, metadata)


async def find_similar_market_conditions(
    current_conditions: Dict[str, Any], symbol: Optional[str] = None, limit: int = 5
) -> List[Tuple[TradingMemory, Dict[str, Any]]]:
    """Find similar historical market conditions"""
    query = f"""
    Market conditions:
    Trend: {current_conditions.get('trend', 'unknown')}
    Volatility: {current_conditions.get('volatility', 'normal')}
    Volume: {current_conditions.get('volume_profile', 'average')}
    Key indicators: {current_conditions.get('indicators', {})}
    """

    memories = await vector_memory.search_memories(
        query=query, symbol=symbol, memory_type=MemoryType.MARKET_CONDITION, limit=limit
    )

    # Return memories with their outcomes
    results = []
    for memory in memories:
        results.append((memory, memory.metadata))

    return results
