"""OpenSearch BM25 lexical store — one index per collection, ACL term
filters applied as query-time filter clauses (pre-filter, not post).

Mirrors the pgvector schema: chunk_id/doc_type/dates/content_hash/
parent_id/embed metadata are indexed here too so both stores return
identical fields and deletion/consistency checks stay symmetric. The
``entitlement_wildcard`` flag is derived from the canonical
``is_wildcard_entitlement`` rule — identical semantics to pgvector.

Author: venkateshwara doijode
"""

from typing import Any

from opensearchpy import OpenSearch, helpers

from app.core.config import Settings
from app.core.logging import get_logger
from app.data.ingest.chunking import Chunk
from app.retrieval.freshness import is_expired, os_freshness_filter
from app.security.authz import ACLFilter, is_wildcard_entitlement

log = get_logger(__name__)

_MAPPINGS = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "client_id": {"type": "keyword"},
            "date": {"type": "keyword"},
            "owner": {"type": "keyword"},
            "doc_version": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "page_number": {"type": "integer"},
            "effective_date": {"type": "keyword"},
            "expiry_date": {"type": "keyword"},
            "section": {"type": "keyword"},
            "doc_title": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "content_hash": {"type": "keyword"},
            "entitlement_roles": {"type": "keyword"},
            "entitlement_wildcard": {"type": "boolean"},
            "embed_model": {"type": "keyword"},
            "embed_version": {"type": "keyword"},
            "embed_dim": {"type": "integer"},
            "text": {"type": "text"},
        }
    }
}

_FIELDS = [
    "doc_id",
    "chunk_id",
    "section",
    "doc_title",
    "text",
    "client_id",
    "source",
    "doc_version",
    "doc_type",
    "page_number",
    "effective_date",
    "expiry_date",
    "parent_id",
    "content_hash",
]


class LexicalStore:
    """OpenSearch-backed BM25 store with per-collection indexes and ACL filters."""

    def __init__(self, settings: Settings):
        """Create the OpenSearch client and record the expiry mode."""
        self._client = OpenSearch(hosts=[settings.opensearch_url], timeout=30, max_retries=2)
        self._expiry_mode = settings.doc_expiry_mode

    @staticmethod
    def index_name(collection: str) -> str:
        """Map a collection name to its OpenSearch index name."""
        return f"rag_{collection}"

    def ensure_index(self, collection: str) -> None:
        """Create the collection's index with the field mappings if missing."""
        name = self.index_name(collection)
        if not self._client.indices.exists(index=name):
            self._client.indices.create(index=name, body=_MAPPINGS)
            log.info("lexical.index_created", index=name)

    def clear(self, collection: str) -> None:
        """Delete the collection's index entirely (used on rebuild)."""
        name = self.index_name(collection)
        if self._client.indices.exists(index=name):
            self._client.indices.delete(index=name)

    @staticmethod
    def _source(c: Chunk, ordinal: int) -> dict[str, Any]:
        """Build the indexed document body for one chunk."""
        return {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "client_id": c.client_id,
            "date": c.date,
            "owner": c.owner,
            "doc_version": c.doc_version,
            "doc_type": c.doc_type,
            "page_number": c.page_number,
            "effective_date": c.effective_date,
            "expiry_date": c.expiry_date,
            "section": c.section,
            "doc_title": c.doc_title,
            "parent_id": c.parent_id,
            "content_hash": c.content_hash,
            "entitlement_roles": c.entitlements.get("roles", []),
            "entitlement_wildcard": is_wildcard_entitlement(c.entitlements),
            "embed_model": c.embed_model,
            "embed_version": c.embed_version,
            "embed_dim": c.embed_dim,
            "text": c.text,
        }

    def bulk_index(self, collection: str, chunks: list[Chunk]) -> int:
        """Bulk-index chunks keyed by chunk_id (idempotent); returns count indexed."""
        name = self.index_name(collection)
        actions = [
            {
                "_index": name,
                "_id": c.chunk_id or f"{c.doc_id}:{i}",
                "_source": self._source(c, i),
            }
            for i, c in enumerate(chunks)
        ]
        ok, _ = helpers.bulk(self._client, actions, refresh=True)
        return ok

    def _filters(self, acl: ACLFilter | None, focus_client: str | None) -> list[dict]:
        """Assemble the query-time filter list: ACL clause + freshness clause."""
        filters = [acl.os_filter(focus_client)] if acl else []
        fresh = os_freshness_filter(self._expiry_mode)
        if fresh:
            filters.append(fresh)
        return filters

    def _hit(self, h: dict) -> dict[str, Any]:
        """Project an OpenSearch hit into the shared result dict shape."""
        src = h["_source"]
        row = {k: src.get(k) for k in _FIELDS}
        row["score"] = h["_score"]
        if self._expiry_mode == "down_rank" and is_expired(row):
            row["stale"] = True
        return row

    def search(
        self,
        query: str,
        collection: str,
        acl: ACLFilter | None,
        top_k: int,
        focus_client: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run an ACL+freshness-filtered BM25 match query; returns top-k hits.

        The entitlement bool filter is applied as a query-time filter clause
        so unauthorized documents are excluded before scoring — identical
        rule to pgvector's WHERE clause.
        """
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    "filter": self._filters(acl, focus_client),
                }
            },
        }
        resp = self._client.search(index=self.index_name(collection), body=body)
        return [self._hit(h) for h in resp["hits"]["hits"]]

    def delete(self, doc_id: str, collection: str | None = None) -> int:
        """Delete every chunk of a document across collections (propagation path)."""
        indexes = [self.index_name(collection)] if collection else self._all_collection_indexes()
        deleted = 0
        for name in indexes:
            if not self._client.indices.exists(index=name):
                continue
            resp = self._client.delete_by_query(
                index=name,
                body={"query": {"term": {"doc_id": doc_id}}},
                refresh=True,
            )
            deleted += int(resp.get("deleted", 0))
        return deleted

    def fetch_chunk(
        self,
        chunk_id: str,
        collection: str | None = None,
        acl: ACLFilter | None = None,
    ) -> dict[str, Any] | None:
        """Point-fetch a chunk by chunk_id (parent-expansion path).

        ``acl`` applies the caller's entitlement filter as defense-in-depth:
        a corrupt ``parent_id`` must not pull a chunk belonging to a client
        the principal isn't entitled to.
        """
        filters = [acl.os_filter()] if acl else []
        for name in [self.index_name(collection)] if collection else self._all_collection_indexes():
            if not self._client.indices.exists(index=name):
                continue
            resp = self._client.search(
                index=name,
                body={
                    "size": 1,
                    "query": {
                        "bool": {"must": [{"term": {"chunk_id": chunk_id}}], "filter": filters}
                    },
                },
            )
            hits = resp["hits"]["hits"]
            if hits:
                return self._hit(hits[0])
        return None

    def list_chunk_ids(self, collection: str) -> set[str]:
        """Return every chunk_id in the collection index — consistency baseline."""
        name = self.index_name(collection)
        if not self._client.indices.exists(index=name):
            return set()
        resp = self._client.search(
            index=name,
            body={
                "size": 0,
                "aggs": {"ids": {"terms": {"field": "chunk_id", "size": 100_000}}},
            },
        )
        return {b["key"] for b in resp["aggregations"]["ids"]["buckets"] if b["key"]}

    def record_index_meta(self, collection: str, info: dict[str, Any]) -> None:
        """Write a ``_meta`` doc holding embedding-model info for the index."""
        self.ensure_index(collection)
        self._client.index(
            index=self.index_name(collection),
            id="_meta",
            body={"kind": "index_meta", **info},
            refresh=True,
        )

    def index_meta(self, collection: str) -> dict[str, Any] | None:
        """Read the ``_meta`` registry doc for the collection index, if present."""
        name = self.index_name(collection)
        if not self._client.indices.exists(index=name):
            return None
        try:
            doc = self._client.get(index=name, id="_meta")
        except Exception:
            return None
        return doc.get("_source") if doc.get("found") else None

    def _all_collection_indexes(self) -> list[str]:
        """Return index names for every known collection (delete-by-doc path)."""
        from app.core.constants import COLLECTIONS

        return [self.index_name(c) for c in COLLECTIONS]
