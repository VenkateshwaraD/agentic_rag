"""Dependency factories — single construction point for swappable services.

Author: venkateshwara doijode
"""

from functools import lru_cache

from app.core.config import get_settings


@lru_cache
def get_embedder():
    """Return the cached embedding client built from current settings."""
    from app.retrieval.embeddings import build_embedder

    return build_embedder(get_settings())


@lru_cache
def get_reranker():
    """Return the cached cross-encoder reranker configured by settings."""
    from app.retrieval.reranker import Reranker

    return Reranker(get_settings().reranker_model)


@lru_cache
def get_opensearch():
    """Return the cached OpenSearch-backed lexical store."""
    from app.retrieval.lexical import LexicalStore

    return LexicalStore(get_settings())


@lru_cache
def get_vectorstore():
    """Return the cached pgvector-backed dense vector store."""
    from app.retrieval.vectorstore import VectorStore

    return VectorStore(get_settings())


@lru_cache
def get_pii_redactor():
    """Return the cached PII redactor used by guardrails and ingestion."""
    from app.security.pii import build_redactor

    return build_redactor()


@lru_cache
def get_audit():
    """Return the cached audit-trail writer."""
    from app.observability.audit import AuditWriter

    return AuditWriter(get_settings())


@lru_cache
def get_hybrid():
    """Assemble and cache the hybrid retriever from embedder, stores, and reranker."""
    from app.retrieval.hybrid import HybridRetriever

    return HybridRetriever(
        embedder=get_embedder(),
        vectors=get_vectorstore(),
        lexical=get_opensearch(),
        reranker=get_reranker(),
        settings=get_settings(),
    )


@lru_cache
def get_graph():
    """Build and cache the compiled LangGraph agent graph."""
    from app.agent.graph import build_graph

    return build_graph()
