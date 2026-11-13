"""OpenSearch BM25 lexical store — one index per collection, ACL term
filters applied as query-time filter clauses (pre-filter, not post)."""

from typing import Any

from opensearchpy import OpenSearch, helpers

from app.core.config import Settings
from app.core.logging import get_logger
from app.data.ingest.chunking import Chunk
from app.security.authz import ACLFilter

log = get_logger(__name__)

_MAPPINGS = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "client_id": {"type": "keyword"},
            "date": {"type": "keyword"},
            "owner": {"type": "keyword"},
            "doc_version": {"type": "keyword"},
            "section": {"type": "keyword"},
            "entitlement_roles": {"type": "keyword"},
            "entitlement_wildcard": {"type": "boolean"},
            "text": {"type": "text"},
        }
    }
}


class LexicalStore:
    def __init__(self, settings: Settings):
        self._client = OpenSearch(hosts=[settings.opensearch_url], timeout=30, max_retries=2)

    @staticmethod
    def index_name(collection: str) -> str:
        return f"rag_{collection}"

    def ensure_index(self, collection: str) -> None:
        name = self.index_name(collection)
        if not self._client.indices.exists(index=name):
            self._client.indices.create(index=name, body=_MAPPINGS)
            log.info("lexical.index_created", index=name)

    def clear(self, collection: str) -> None:
        name = self.index_name(collection)
        if self._client.indices.exists(index=name):
            self._client.indices.delete(index=name)

    def bulk_index(self, collection: str, chunks: list[Chunk]) -> int:
        name = self.index_name(collection)
        actions = [
            {
                "_index": name,
                "_id": f"{c.doc_id}:{i}",
                "_source": {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "client_id": c.client_id,
                    "date": c.date,
                    "owner": c.owner,
                    "doc_version": c.doc_version,
                    "section": c.section,
                    "entitlement_roles": c.entitlements.get("roles", []),
                    "entitlement_wildcard": c.entitlements.get("clients") == "*",
                    "text": c.text,
                },
            }
            for i, c in enumerate(chunks)
        ]
        ok, _ = helpers.bulk(self._client, actions, refresh=True)
        return ok

    def search(
        self,
        query: str,
        collection: str,
        acl: ACLFilter,
        top_k: int,
        focus_client: str | None = None,
    ) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    "filter": [acl.os_filter(focus_client)],
                }
            },
        }
        resp = self._client.search(index=self.index_name(collection), body=body)
        return [
            {
                "doc_id": h["_source"]["doc_id"],
                "section": h["_source"]["section"],
                "text": h["_source"]["text"],
                "client_id": h["_source"]["client_id"],
                "source": h["_source"]["source"],
                "score": h["_score"],
            }
            for h in resp["hits"]["hits"]
        ]
