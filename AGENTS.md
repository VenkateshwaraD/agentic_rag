# AGENTS.md

Guidelines for AI agents and contributors working in this repository.

Author: venkateshwara doijode

## Project Overview

Compliance-aware agentic RAG client-service copilot: LangGraph agent
(guardrail → route → retrieve → grade → synthesize → guardrail) over hybrid
retrieval (pgvector dense + OpenSearch BM25, RRF fusion, cross-encoder
rerank), with parameterized read-only ops tools, per-client ACL enforcement,
PII redaction, audit logging, and an eval harness.

- Source: `src/app` | Eval: `eval/` | Tests: `tests/` | DB seed/migrations: `db/`
- Entry points: `copilot` CLI (`app.cli:app`), FastAPI (`app.main:app`),
  Streamlit UI (`src/ui/app.py`)

## Commands

```bash
make install          # venv + editable install (.[dev])
make lint             # ruff check + ruff format --check + mypy src
make test-unit        # pytest tests/unit -q
make test-integration # pytest tests/integration -q -m integration (needs Docker)
make dev-up / dev-down
make corpus           # generate synthetic corpus + ops seed + users.json
make index            # index corpus into both stores (--rebuild)
make eval             # run golden-set eval
```

Verify changes with `make lint` and `make test-unit` before committing.

## Code Conventions

- Python 3.12+, type hints on public signatures; ruff (E,F,I,W,UP,B,SIM) +
  ruff-format (line-length 100) + mypy — all must pass.
- Every module gets a docstring ending with `Author: venkateshwara doijode`;
  every function/class gets a docstring describing what it does.
- No comments unless asked; no emojis.
- Imports lazy inside functions where a module is heavy or would create a
  cycle (see `core/dependencies.py`, `retrieval/embeddings.py`).

## Security Rules (non-negotiable)

- **Secrets**: never hardcode real credentials/keys. Dev fixtures
  (`dev-key-*`, `admin-dev-token`, `db/seed/users.json`) are prototype-only —
  `users.json` is generated, never committed; `.env` is gitignored.
- **SQL**: parameterized queries only — never f-string/format user input into
  SQL. `tools/ops` uses fixed templates + `client_clause()` bound params.
- **ACL**: enforce entitlements as a *pre-filter* in the store query
  (`ACLFilter.pg_where` / `os_filter` / `client_clause`), never post-filter.
  Unauthorized rows must be indistinguishable from nonexistent.
- **PII**: redact at ingest (`redact_chunk_text`), in audit payloads
  (`_redact_payload`), in prompt context (`_format_evidence` tool rows), and
  rescan output (`check_output`). No raw PII may reach indexes, logs, audit
  rows, LLM context, or API responses.
- **Ops DB**: read-only only — `rag_readonly` role, session READ ONLY mode,
  statement_timeout, row cap.
- **Guardrails**: input (injection + length) before routing; output (PII
  rescan) before replying; refusal path must never include drafted content.
- API responses expose only: reply, citations, retrieved chunks (already
  redacted), tool-call *metadata* (never raw rows), refusal flag, ids.

## Review Checklist

- No dead code: unused functions/params/schemas (`AuditEventOut`,
  `Principal.is_admin`, `VectorStore._connect(dim)`, `get_graph`) flagged.
- No magic strings duplicated across modules (`_REFUSAL`, client lists —
  keep a single source of truth in `core/constants.py`).
- SQL text that interpolates Python values must be an f-string (see
  `ensure_schema`) — and must still be parameterized for user data.
- API request/response schema must match what handlers return and what
  clients consume (see `AnswerResponse` vs UI `diagnostics`).
- Edge cases: empty collections/client lists → deny-all (`FALSE` clause),
  missing keys → `next()`/KeyError, empty CSV/JSONL → index errors.
- Error handling: swallow only where documented (audit write); dependency
  failures surface as 500 via `AppError` mapping.

## Testing

- Unit tests must not require Docker, network, LLM keys, or spaCy models.
- New behavior → new unit test; verify ACL denial paths and refusal paths
  explicitly (they are the security contract).
