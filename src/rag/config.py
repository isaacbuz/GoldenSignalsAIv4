"""RAG System Configuration"""

RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_dimension": 384,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "retrieval_k": 5,
    "rerank_enabled": True,
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hour
}

DOCUMENT_TYPES = {
    "market_data": {"weight": 1.0, "fields": ["price", "volume", "timestamp"]},
    "news": {"weight": 0.8, "fields": ["title", "content", "sentiment", "timestamp"]},
    "signals": {"weight": 1.2, "fields": ["signal_type", "confidence", "outcome", "timestamp"]},
    "patterns": {"weight": 1.5, "fields": ["pattern_type", "accuracy", "occurrences", "timestamp"]},
}
