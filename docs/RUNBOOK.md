# Runbook — Start, Configure, Run, Evaluate

One-page operational guide: everything needed to bring the copilot up,
point it at data, ask it questions, and read the results. For *how it
works inside* see `docs/WORKFLOW.md`; for evaluation theory see
`docs/RAGAS.md`; for the plain-RAG control see `docs/BASELINE.md`.

Author: venkateshwara doijode

---

## 1. Install

```bash
make install            # .venv + editable install + spacy PII model
```

Requires: Python 3.12, Docker (for Postgres + OpenSearch).

## 2. Configure before first run — where keys go

All settings live in `src/app/core/config.py` and load from
**environment variables or `.env` at the repo root** (pydantic
`env_file`). Copy the template and fill it in:

```bash
cp .env.example .env
```

| `.env` variable | purpose | needed for |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude key — LLM drafting/routing/Ragas judge | real LLM answers (empty = deterministic offline mode, fully functional) |
| `ANTHROPIC_MODEL` | model name | defaults `claude-sonnet-4-20250514` |
| `EMBEDDING_PROVIDER` | `bedrock` \| `local` | `local` needs no AWS creds (MiniLM) |
| `AWS_REGION` / `AWS_PROFILE` | Bedrock Titan embeddings | only if `EMBEDDING_PROVIDER=bedrock` |
| `DATABASE_URL` | pgvector store + idempotency/audit/jobs | always |
| `OPS_DATABASE_URL` | read-only ops DB (accounts/trades/tickets) | always |
| `OPENSEARCH_URL` | BM25 store | always |
| `API_KEYS` | comma-separated caller keys (auth) | API/UI access |
| `USERS_FILE` | JSON user fixture (role + client entitlements + key) | auth |
| `ADMIN_TOKEN` | `/admin/*` bearer token | admin endpoints |
| `LANGFUSE_*` | tracing | optional |

**Rule:** a user's entry in `USERS_FILE` (`key` field) must also appear
in `API_KEYS` — both checks happen in `security/authn.py`. Dev fixture:
`dev-key-1` for agents, `dev-key-2` for senior/admin.

**Production:** do not put secrets in `.env` or manifests — prod k8s uses
`ExternalSecret` (AWS Secrets Manager); see `deploy/README.md`.

## 3. Input data — what the agent reads

| data | where | produced by |
|---|---|---|
| policy/agreement/SLA/team docs | `data/synthetic/{agreements,slas,service_teams,ops}/` | `copilot corpus generate` |
| ops rows (accounts, trades, tickets) | `db/seed/*.csv` | `copilot corpus generate` → loaded by `copilot db init` |
| users + entitlements | `db/seed/users.json` | `copilot corpus generate` |
| expected facts (eval authoring aid) | `data/synthetic/facts.json` | `copilot corpus generate` |

**Adding your own document:** drop a `.md` into the right
`data/synthetic/<collection>/` folder with a header block like the
existing files (`doc_id`, `client_id`, `title`, `version`, dates,
`roles`/`clients` entitlement fields — copy an existing file's header),
then `copilot index` (incremental — only changed docs re-index).

## 4. Start the stack

```bash
make dev-up             # Postgres :5432 + OpenSearch :9200 (containers)
copilot corpus generate # docs + ops CSVs + users.json (first time only)
copilot db init         # ops schema + rag_readonly role + CSV seed + audit/jobs tables
copilot index --rebuild # parse → chunk → redact → embed → both stores
copilot docs verify     # sanity: pgvector ↔ OpenSearch parity check
```

## 5. Run the agent

**CLI** (fastest smoke):

```bash
copilot ask "What is the wire fee for acme?" --user u_agent1
```

**API** (`make run-api` → `uvicorn` on :8000):

```bash
curl -X POST http://localhost:8000/api/v1/answer \
  -H "X-API-Key: dev-key-1" -H "X-User-Id: u_agent1" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show open tickets for acme"}'

# streaming variant: POST /api/v1/answer/stream (SSE, PII-sanitized deltas)
# safe retries: add -H "Idempotency-Key: <uuid>"
```

**UI** (`make run-ui` → http://localhost:8501): chat page + eval
dashboard; "Acting as" picker maps to the users fixture.

**Admin** (`-H "Authorization: Bearer $ADMIN_TOKEN"`): `/api/v1/admin/*`
— ingestion jobs, dead-letter retry, review-queue decisions.

## 6. Run with the LLM key vs without

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # (or set in .env)
copilot ask "..."                      # LLM drafting + LLM routing
```

- **Key set** → Claude drafts, routes, and judges (Ragas runs in evals).
- **Key unset** → everything still works: keyword routing + extractive
  offline draft. Fully deterministic — used for unit tests, CI, and the
  committed eval reports. No stubbed scores; the pipeline really runs.
- Same toggle applies to **both** sides of the baseline comparison
  (`docs/BASELINE.md` §4) so it never becomes model-vs-model.

## 7. Where results land

| output | location |
|---|---|
| Answer JSON (reply, citations, answerability, validation, diagnostics) | stdout / API response / `docs/samples/answer_example.json` (committed example) |
| Per-run eval reports | `eval/reports/eval_<ts>.{json,md}` (tagged `eval_<system>_<ts>` for baselines) |
| Baseline comparison | `eval/reports/compare_<ts>.{json,md}` |
| PII redaction scores | `eval/reports/redaction_<ts>.json` |
| Audit trail | `audit_events` table (ragdb) — one row per node decision |
| Human-review queue | `review_queue` table → `POST /api/v1/admin/review/{id}/decision` |
| Ingestion jobs / dead letters | `copilot jobs list` / `copilot jobs dlq` |

## 8. Evaluation

```bash
# golden / scenario suite
copilot eval run --suite pr --fail-on-leak --fail-on-thresholds   # what CI runs
python3 eval/runner.py --suite dev                                # manual suite run
python3 eval/redaction_eval.py                                    # PII precision/recall

# agentic vs plain-RAG baselines (the project's headline comparison)
python3 eval/runner.py --suite pr --system all                    # → compare_<ts>.md
```

Each report carries a `version_snapshot` (model, embedder, prompt
hashes, dataset hash, git SHA) so any two runs are directly comparable.
Reading the numbers: `docs/RAGAS.md` (metrics) and `docs/BASELINE.md`
§7 (comparison semantics — `n/a` = not measured, not zero).

## 9. Verify it works — 60-second checklist

```bash
make lint && make test-unit            # static + unit gates
copilot ask "wire fee for acme" --user u_agent1      # → cited fee answer
copilot ask "accounts for grainmark" --user u_agent1 # → unauthorized refusal (not entitled)
python3 eval/runner.py --suite pr --limit 3 --no-ragas   # smoke eval
```

Expected: a `[DOC-ID §section]`-cited reply, an `unauthorized` verdict
for the out-of-scope client, and `eval/reports/` artifacts with zero
ACL leaks.
