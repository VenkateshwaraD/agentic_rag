"""pgvector dense store — one `chunks` table, HNSW index, ACL pre-filters
applied in the WHERE clause so unauthorized docs never leave the store.

Also owns the ``index_meta`` registry (per-collection embedding
model/version/dimension) and the store-side of deletion propagation,
parent-child fetches, and pgvector↔OpenSearch consistency listing.

Author: venkateshwara doijode
"""

import json
import re
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import Settings
from app.core.logging import get_logger
from app.data.ingest.chunking import Chunk
from app.retrieval.freshness import is_expired, pg_freshness_where
from app.security.authz import ACLFilter

log = get_logger(__name__)

_SELECT = (
    "doc_id, chunk_id, section, text, client_id, source, doc_version, "
    "doc_type, page_number, effective_date, expiry_date, parent_id, content_hash"
)


class VectorStore:
    """pgvector-backed dense store over the ``chunks`` table with ACL pre-filters."""

    def __init__(self, settings: Settings):
        """Store the rag database URL and expiry mode for later connections."""
        self._url = settings.database_url
        self._expiry_mode = settings.doc_expiry_mode

    def _connect(self) -> psycopg.Connection:
        """Open a connection, ensure the vector extension, and register the codec."""
        conn = psycopg.connect(self._url, autocommit=False)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def ensure_schema(self, dim: int) -> None:
        """Create the chunks + index_meta tables and apply column migrations."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id            SERIAL PRIMARY KEY,
                    doc_id        TEXT NOT NULL,
                    chunk_id      TEXT DEFAULT '',
                    source        TEXT NOT NULL,
                    client_id     TEXT DEFAULT '',
                    date          TEXT DEFAULT '',
                    owner         TEXT DEFAULT '',
                    doc_version   TEXT DEFAULT '1.0',
                    doc_type      TEXT DEFAULT 'generic',
                    page_number   INT,
                    effective_date TEXT DEFAULT '',
                    expiry_date   TEXT DEFAULT '',
                    section       TEXT DEFAULT '',
                    parent_id     TEXT DEFAULT '',
                    content_hash  TEXT DEFAULT '',
                    entitlements  JSONB NOT NULL DEFAULT '{{}}',
                    embed_model   TEXT DEFAULT '',
                    embed_version TEXT DEFAULT '',
                    embed_dim     INT DEFAULT 0,
                    text          TEXT NOT NULL,
                    embedding     vector({dim})
                )"""
            )
            for stmt in [
                "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_client ON chunks(client_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks "
                "USING hnsw (embedding vector_cosine_ops)",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_id TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_type TEXT DEFAULT 'generic'",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number INT",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS effective_date TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS expiry_date TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_id TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embed_model TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embed_version TEXT DEFAULT ''",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embed_dim INT DEFAULT 0",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_source_chunkid "
                "ON chunks(source, chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)",
                """
                CREATE TABLE IF NOT EXISTS index_meta (
                    collection    TEXT PRIMARY KEY,
                    embed_model   TEXT NOT NULL,
                    embed_version TEXT DEFAULT '',
                    embed_dim     INT NOT NULL,
                    provider      TEXT DEFAULT '',
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """,
            ]:
                cur.execute(stmt)
        log.info("vectorstore.schema_ready", dim=dim)

    def record_index_meta(
        self, collection: str, model: str, version: str, dim: int, provider: str
    ) -> None:
        """Upsert the embedding-model metadata recorded for an index."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO index_meta (collection, embed_model, embed_version, embed_dim, provider)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (collection) DO UPDATE SET
                    embed_model = EXCLUDED.embed_model,
                    embed_version = EXCLUDED.embed_version,
                    embed_dim = EXCLUDED.embed_dim,
                    provider = EXCLUDED.provider,
                    updated_at = now()
                """,
                (collection, model, version, dim, provider),
            )

    def index_meta(self) -> list[dict[str, Any]]:
        """Return all per-collection embedding-model registry rows."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT collection, embed_model, embed_version, embed_dim, provider, "
                "updated_at FROM index_meta ORDER BY collection"
            )
            cols = [d.name for d in cur.description or []]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def clear(self) -> None:
        """Truncate the chunks table (used on full rebuild)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE chunks")

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Bulk-insert chunks with embeddings; idempotent on (source, chunk_id).

        Returns rows actually inserted — duplicate chunk_ids are skipped via
        ``ON CONFLICT DO NOTHING``.
        """
        import numpy as np

        rows = [
            (
                c.doc_id,
                c.chunk_id,
                c.source,
                c.client_id,
                c.date,
                c.owner,
                c.doc_version,
                c.doc_type,
                c.page_number,
                c.effective_date,
                c.expiry_date,
                c.section,
                c.parent_id,
                c.content_hash,
                json.dumps(c.entitlements),
                c.embed_model,
                c.embed_version,
                c.embed_dim,
                c.text,
                np.asarray(emb, dtype=np.float32),
            )
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (doc_id, chunk_id, source, client_id, date, owner,
                                    doc_version, doc_type, page_number, effective_date,
                                    expiry_date, section, parent_id, content_hash,
                                    entitlements, embed_model, embed_version, embed_dim,
                                    text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, chunk_id) DO NOTHING""",
                rows,
            )
        return len(rows)

    def search(
        self,
        query_vec: list[float],
        source: str,
        acl: ACLFilter,
        top_k: int,
        focus_client: str | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search over one collection with ACL + freshness filters.

        Translates named params to positional for psycopg and returns the
        top-k rows as dicts including the similarity score and full chunk
        metadata (version, dates, page, chunk_id, parent_id).
        """
        where, params = acl.pg_where(focus_client)
        fresh_where, fresh_params = pg_freshness_where(self._expiry_mode)
        params.update(fresh_params)
        params["src"] = source
        params["k"] = top_k
        sql = f"""
            SELECT {_SELECT},
                   1 - (embedding <=> :qvec) AS score
            FROM chunks
            WHERE source = :src AND {where} {fresh_where}
            ORDER BY embedding <=> :qvec
            LIMIT :k"""
        import numpy as np

        qvec = np.asarray(query_vec, dtype=np.float32)
        order = re.findall(r":(\w+)", sql)
        values = [qvec if n == "qvec" else params[n] for n in order]
        sql_pos = re.sub(r":\w+", "%s", sql)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql_pos, values)
            cols = [d.name for d in cur.description or []]
            rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
        for r in rows:
            if self._expiry_mode == "down_rank" and r.get("expiry_date"):
                r["stale"] = is_expired(r)
        return rows

    def fetch_chunk(self, chunk_id: str, source: str | None = None) -> dict[str, Any] | None:
        """Point-fetch a single chunk by chunk_id (parent-expansion path)."""
        sql = f"SELECT {_SELECT} FROM chunks WHERE chunk_id = :cid"
        params: dict[str, Any] = {"cid": chunk_id}
        if source:
            sql += " AND source = :src"
            params["src"] = source
        sql += " LIMIT 1"
        with self._connect() as conn, conn.cursor() as cur:
            order = re.findall(r":(\w+)", sql)
            values = [params[n] for n in order]
            cur.execute(re.sub(r":\w+", "%s", sql), values)
            cols = [d.name for d in cur.description or []]
            row = cur.fetchone()
        return dict(zip(cols, row, strict=True)) if row else None

    def fetch_by_parent(self, parent_id: str) -> list[dict[str, Any]]:
        """Return all child chunks of a parent section chunk."""
        sql = f"SELECT {_SELECT} FROM chunks WHERE parent_id = :pid ORDER BY chunk_id"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(re.sub(r":\w+", "%s", sql), (parent_id,))
            cols = [d.name for d in cur.description or []]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def delete(self, doc_id: str) -> int:
        """Delete every chunk of a document (propagation path); returns rows removed."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            return cur.rowcount

    def list_documents(self, source: str) -> list[dict[str, Any]]:
        """Return per-document chunk counts + version for a collection."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, MAX(doc_version) AS doc_version, COUNT(*) AS chunks "
                "FROM chunks WHERE source = %s GROUP BY doc_id ORDER BY doc_id",
                (source,),
            )
            return [{"doc_id": r[0], "doc_version": r[1], "chunks": r[2]} for r in cur.fetchall()]

    def list_chunk_ids(self, source: str) -> set[str]:
        """Return every chunk_id in a collection — consistency-check baseline."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM chunks WHERE source = %s AND chunk_id != ''", (source,)
            )
            return {r[0] for r in cur.fetchall()}
