"""Application settings, loaded from environment / .env via pydantic-settings.

Author: venkateshwara doijode
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings populated from environment variables / .env.

    Covers LLM and embedding providers, datastore URLs, auth fixtures,
    Langfuse tracing, retrieval tuning, ops-tool limits, API bind address,
    and filesystem paths for data/eval directories.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # --- Embeddings ---
    embedding_provider: str = "bedrock"  # "bedrock" | "local"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    local_embed_model: str = "all-MiniLM-L6-v2"
    aws_region: str = "us-east-1"
    aws_profile: str | None = None
    embedding_dim: int = 1024  # Titan v2 default; local provider overrides at runtime

    # --- Datastores ---
    database_url: str = "postgresql://rag:rag@localhost:5432/ragdb"
    ops_database_url: str = "postgresql://rag_readonly:readonly@localhost:5432/opsdb"
    ops_admin_database_url: str = "postgresql://rag:rag@localhost:5432/opsdb"
    opensearch_url: str = "http://localhost:9200"

    # --- Auth (prototype) ---
    api_keys: str = "dev-key-1,dev-key-2"  # dev fixture keys (db/seed/users.json)
    users_file: Path = PROJECT_ROOT / "db" / "seed" / "users.json"
    admin_token: str = "admin-dev-token"

    # --- Langfuse tracing (disabled unless keys set) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # --- Retrieval ---
    dense_top_k: int = 20
    lexical_top_k: int = 20
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0
    rrf_bm25_weight: float = 1.0
    rerank_candidates: int = 20
    final_top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"
    # dense | bm25 | hybrid | hybrid_rerank | auto (classifier decides)
    retrieval_strategy: str = "hybrid_rerank"
    # exclude | down_rank — how expired/stale docs are handled at search time
    doc_expiry_mode: str = "exclude"
    parent_child_enabled: bool = False

    # --- Agent execution budget ---
    agent_max_iterations: int = 6  # hard: retrieve passes through the graph
    agent_max_tool_calls: int = 10  # hard: total ops tool invocations
    agent_max_tokens: int = 20_000  # soft: total LLM tokens (best-effort)
    agent_max_elapsed_ms: int = 60_000  # hard: wall-clock per request
    agent_max_cost_usd: float = 0.25  # soft: estimated LLM spend (best-effort)
    agent_max_retries: int = 2  # bounded rewrite loop
    llm_timeout_s: int = 30

    # --- Ingestion ---
    index_mode: str = "incremental"  # full | incremental
    ingest_max_attempts: int = 3  # per-document retry before dead-letter
    doc_version_policy: str = "latest"  # latest | keep_all

    # --- Ops tools ---
    ops_row_cap: int = 100
    ops_statement_timeout_ms: int = 10_000
    ops_pool_size: int = 5
    ops_pool_max_overflow: int = 5
    ops_pool_timeout_s: int = 10

    # --- Human review ---
    human_review_enabled: bool = True
    review_citation_threshold: float = 0.5

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    request_timeout_s: int = 60
    idempotency_ttl_s: int = 86_400
    stream_enabled: bool = True

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    synthetic_dir: Path = PROJECT_ROOT / "data" / "synthetic"
    real_dir: Path = PROJECT_ROOT / "data" / "real"
    eval_dir: Path = PROJECT_ROOT / "eval"

    @property
    def valid_api_keys(self) -> set[str]:
        """Parse the comma-separated ``api_keys`` value into a set of non-empty keys."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def langfuse_enabled(self) -> bool:
        """Return True only when both Langfuse keys are set, enabling tracing."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Build the ``Settings`` object once and return the cached singleton."""
    return Settings()
