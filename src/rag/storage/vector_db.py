"""
Vector Database Integration for RAG System
Supports multiple vector database backends
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors with metadata"""
        pass
    
    @abstractmethod
    async def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by ID"""
        pass

class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = {}
        self.metadata = {}
        
    async def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors to memory store"""
        ids = []
        for vector, meta in zip(vectors, metadata):
            vec_id = f"vec_{len(self.vectors)}_{datetime.now().timestamp()}"
            self.vectors[vec_id] = vector
            self.metadata[vec_id] = meta
            ids.append(vec_id)
        return ids
    
    async def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Cosine similarity search"""
        similarities = []
        
        for vec_id, vector in self.vectors.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((self.metadata[vec_id], similarity, vec_id))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(meta, score) for meta, score, _ in similarities[:k]]
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors from store"""
        for vec_id in ids:
            if vec_id in self.vectors:
                del self.vectors[vec_id]
                del self.metadata[vec_id]
        return True

class ChromaDBStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "goldensignals"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
    async def initialize(self):
        """Initialize ChromaDB client"""
        try:
            import chromadb
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except ImportError:
            logger.warning("ChromaDB not installed, falling back to in-memory store")
            raise
    
    async def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors to ChromaDB"""
        ids = [f"vec_{i}_{datetime.now().timestamp()}" for i in range(len(vectors))]
        
        self.collection.add(
            embeddings=[v.tolist() for v in vectors],
            metadatas=metadata,
            ids=ids
        )
        
        return ids
    
    async def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k
        )
        
        pairs = []
        if results['metadatas'] and results['distances']:
            for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
                similarity = 1 - dist  # Convert distance to similarity
                pairs.append((meta, similarity))
        
        return pairs
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete from ChromaDB"""
        self.collection.delete(ids=ids)
        return True

class VectorDBManager:
    """Manages vector database operations"""
    
    def __init__(self, store_type: str = "memory"):
        self.store_type = store_type
        self.store = None
        
    async def initialize(self):
        """Initialize vector store"""
        if self.store_type == "chromadb":
            try:
                self.store = ChromaDBStore()
                await self.store.initialize()
            except:
                logger.warning("ChromaDB initialization failed, using in-memory store")
                self.store = InMemoryVectorStore()
        else:
            self.store = InMemoryVectorStore()
            
        logger.info(f"Vector store initialized: {type(self.store).__name__}")
    
    async def index_documents(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Index documents with embeddings"""
        metadata = []
        for doc in documents:
            meta = {
                "text": doc.get("text", ""),
                "timestamp": doc.get("timestamp", datetime.now()).isoformat(),
                "type": doc.get("type", "general"),
                "source": doc.get("source", "unknown")
            }
            metadata.append(meta)
        
        return await self.store.add_vectors(embeddings, metadata)
    
    async def similarity_search(self, query_embedding: np.ndarray, k: int = 5, filter_type: Optional[str] = None):
        """Search for similar documents"""
        results = await self.store.search(query_embedding, k * 2)  # Get more results for filtering
        
        if filter_type:
            filtered = [(meta, score) for meta, score in results if meta.get("type") == filter_type]
            return filtered[:k]
        
        return results[:k]
