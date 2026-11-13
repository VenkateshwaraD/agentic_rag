# AGENTS.md

Guidelines for AI agents and contributors working in this repository.

Author: venkateshwara doijode

## Project Overview

Compliance-aware agentic RAG client-service copilot. LangGraph **state
machine** (`guardrail_in → route → retrieve → grade → rewrite|synthesize →
validate → guardrail_out`) with bounded budgets (iterations, tool calls,
tokens, elapsed, cost), failure-type-conditional retries, answerability
classification (fully/partial/insufficient/conflicting/unauthorized), and
independent claim-level answer validation with a one-shot re-synthesis
then abstain policy. Retrieval is pluggable (dense / BM25 / hybrid /
hybrid+rerank) with weighted RRF, metadata-aware chunking, freshness and
versioning, identical ACL pre-filters in pgvector and OpenSearch, and
parent-child expansion. Ingestion is a staged pipeline (parse → validate →
normalize → chunk → redact → embed → index) with job tracking, incremental
mode, retries, and a dead-letter queue. Ops tools are strict-schema,
read-only, entitlement-scoped store interfaces. API is versioned at
`/api/v1` with correlation ids, error envelopes, idempotency, SSE
streaming, and an async human-review queue.

- Source: `src/app` | Eval: `eval/` | Tests: `tests/` | Deploy: `deploy/`
- Entry points: `copilot` CLI (`app.cli:app`), FastAPI (`app.main:app`),
  Streamlit UI (`src/ui/app.py`)

## Commands

```bash
make install            # venv + editable install (.[dev])
make lint               # ruff check + ruff format --check + mypy src
make test-unit          # pytest tests/unit -q
make test-integration   # pytest tests/integration -q (needs Postgres/OpenSearch)
make dev-up / dev-down
copilot corpus generate # regenerate synthetic corpus + ops seed + users.json
copilot db init         # ops schema + readonly role + CSV seed + audit table
copilot index [--rebuild] # staged ingestion (incremental unless --rebuild)
copilot docs list|delete|verify|index-info
copilot jobs list|dlq   # ingestion job status + dead letters
copilot eval run --suite pr|dev|holdout|golden [--repeat N --fail-on-leak]
python eval/redaction_eval.py  # PII redaction precision/recall harness
```

Verify changes with `make lint` and `make test-unit` before committing.

## Agent State Machine (rules)

- Nodes stay **independently testable** — each takes state, returns a
  partial-state dict, no cross-node calls. New nodes wire through
  `build_graph()` with a conditional-edge function.
- **Never loop unbounded**: retries go through `grade` → `rewrite` gated by
  `agent_max_retries`; every conditional edge checks `budget.refusal_update`
  and `cancelled` first. Hard budget breaches → refusal; soft breaches →
  record and continue.
- Failure types (`missing_evidence`, `low_relevance`,
  `conflicting_documents`, `incomplete_facts`) drive rewrite strategy —
  grade sets `failure_type` + `answerability`; don't route on ad-hoc flags.
- Validation failures re-synthesize **once** (`resynthesized` flag), then
  abstain. Refusal/abstain paths never carry drafted content.
- Store/LLM/tool errors are captured per-call into `errors` (sanitized) and
  diagnostics — never raise through the graph to the API.

## Ingestion & Retrieval (rules)

- Keep the stage boundary testable: `loaders.parse_document` →
  `validate_document` → `normalize_text` → `chunk_document` →
  `redact_chunk_text` → embed → index. `pipeline.ingest_document` composes
  them and returns `IngestResult` — it never raises.
- `chunk_id` embeds `doc_version` (`{doc_id}:v{ver}:{slug}:{i}`) — required
  for `keep_all` versioning to coexist and `latest` deletes to be clean.
- Incremental mode diffs file-hash vs `document_registry`; changed docs
  propagate deletes BEFORE re-index. `doc_version_policy` governs same-doc_id
  cross-path handling.
- **ACL parity is non-negotiable**: `ACLFilter.pg_where`/`os_filter` and the
  canonical wildcard literal `*` must stay semantically identical in both
  stores; freshness uses `freshness.py` in both query paths.
- Expired docs: `doc_expiry_mode=exclude` (pre-filter) or `down_rank`
  (`stale` flag + fusion penalty). Never silently include expired content.

## API (rules)

- New endpoints mount under `/api/v1`; legacy flat routes stay for compat.
- Errors use the `{"error": {code, message, request_id}}` envelope via
  `AppError.code` — never leak raw exception text to clients.
- `Idempotency-Key` on POST /answer replays safely; `X-Timeout-Ms` tightens
  (never widens) the elapsed budget; `X-Request-Id` propagates to state/logs.

## Security Rules (non-negotiable)

- **Secrets**: never hardcode real credentials. Prod k8s uses
  `ExternalSecret` (AWS SM) — no plaintext `Secret` manifests in prod overlay.
- **SQL**: parameterized only. `tools/ops` additionally asserts
  SELECT-only statements at run time (`assert_readonly_sql`) and verifies
  row entitlements post-query (`assert_entitled_rows`).
- **ACL**: pre-filter in the store query (never post-filter); ops stores
  also gate via `assert_ops_allowed` (role + non-empty client scope).
  Unauthorized rows must be indistinguishable from nonexistent.
- **PII**: redact at ingest, in audit payloads, in prompt context, rescan
  output. Errors/logs/traces carry `type(exc).__name__`, never raw values.
- **Ops DB**: read-only — `rag_readonly` role, READ ONLY session,
  statement_timeout, row cap + pagination. No write path exists by design.
- **Human review**: partial/conflicting/low-support/ops-error/high-impact
  answers enqueue to `review_queue` (`pending_review` flag); reviewers
  resolve via `POST /api/v1/admin/review/{id}/decision`.

## CI/CD (rules)

- GitHub Actions are pinned to commit SHAs (with a `# vX.Y.Z` comment);
  bump deliberately after review.
- Cross-job values go through `needs.<job>.outputs` — never `steps.*`
  across jobs. No `|| true` on deploy steps.
- Deploy gates: rollout-status checks + `deploy/scripts/smoke.sh`; prod
  pins the built SHA and is `workflow_dispatch` only.
- Zero-tolerance: `eval/runner.py --suite pr --fail-on-leak` fails the
  pipeline on any ACL leak.

## Testing

- Unit tests must not require Docker, network, LLM keys, or spaCy models.
- Integration tests (`tests/integration`, marked `integration`) may hit
  Postgres/OpenSearch — they must skip cleanly when services are down.
- Eval suites live in `eval/scenarios/` (pr | dev | holdout); holdout rows
  are never used for prompt/config tuning. Reports record the
  `version_snapshot` (model/embed/prompts/dataset/git) for comparability.
- New behavior → new unit test; verify ACL denial and refusal paths
  explicitly (they are the security contract).
