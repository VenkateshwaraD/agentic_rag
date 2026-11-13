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
    rerank_candidates: int = 20
    final_top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Ops tools ---
    ops_row_cap: int = 100
    ops_statement_timeout_ms: int = 10_000

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

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
