"""Dependency factories — single construction point for swappable services."""

from functools import lru_cache

from app.core.config import get_settings


@lru_cache
def get_embedder():
    from app.retrieval.embeddings import build_embedder

    return build_embedder(get_settings())


@lru_cache
def get_reranker():
    from app.retrieval.reranker import Reranker

    return Reranker(get_settings().reranker_model)


@lru_cache
def get_opensearch():
    from app.retrieval.lexical import LexicalStore

    return LexicalStore(get_settings())


@lru_cache
def get_vectorstore():
    from app.retrieval.vectorstore import VectorStore

    return VectorStore(get_settings())


@lru_cache
def get_pii_redactor():
    from app.security.pii import build_redactor

    return build_redactor()


@lru_cache
def get_audit():
    from app.observability.audit import AuditWriter

    return AuditWriter(get_settings())


@lru_cache
def get_hybrid():
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
    from app.agent.graph import build_graph

    return build_graph()
