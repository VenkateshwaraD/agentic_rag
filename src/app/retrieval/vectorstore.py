"""pgvector dense store — one `chunks` table, HNSW index, ACL pre-filters
applied in the WHERE clause so unauthorized docs never leave the store."""

import json
import re
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import Settings
from app.core.logging import get_logger
from app.data.ingest.chunking import Chunk
from app.security.authz import ACLFilter

log = get_logger(__name__)


class VectorStore:
    def __init__(self, settings: Settings):
        self._url = settings.database_url

    def _connect(self, dim: int | None = None) -> psycopg.Connection:
        conn = psycopg.connect(self._url, autocommit=False)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def ensure_schema(self, dim: int) -> None:
        with self._connect(dim) as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id            SERIAL PRIMARY KEY,
                    doc_id        TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    client_id     TEXT DEFAULT '',
                    date          TEXT DEFAULT '',
                    owner         TEXT DEFAULT '',
                    doc_version   TEXT DEFAULT '1.0',
                    section       TEXT DEFAULT '',
                    entitlements  JSONB NOT NULL DEFAULT '{}',
                    text          TEXT NOT NULL,
                    embedding     vector({dim})
                )"""
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_client ON chunks(client_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
                USING hnsw (embedding vector_cosine_ops)"""
            )
        log.info("vectorstore.schema_ready", dim=dim)

    def clear(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE chunks")

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        import numpy as np

        rows = [
            (
                c.doc_id,
                c.source,
                c.client_id,
                c.date,
                c.owner,
                c.doc_version,
                c.section,
                json.dumps(c.entitlements),
                c.text,
                np.asarray(emb, dtype=np.float32),
            )
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (doc_id, source, client_id, date, owner,
                                    doc_version, section, entitlements, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
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
        where, params = acl.pg_where(focus_client)
        params["src"] = source
        params["k"] = top_k
        sql = f"""
            SELECT doc_id, section, text, client_id, source,
                   1 - (embedding <=> :qvec) AS score
            FROM chunks
            WHERE source = :src AND {where}
            ORDER BY embedding <=> :qvec
            LIMIT :k"""
        import numpy as np

        qvec = np.asarray(query_vec, dtype=np.float32)
        order = re.findall(r":(\w+)", sql)
        values = [qvec if n == "qvec" else params[n] for n in order]
        sql_pos = re.sub(r":\w+", "%s", sql)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql_pos, values)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
