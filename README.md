# Agentic RAG Client-Service Copilot

A compliance-aware agentic RAG copilot that drafts cited replies to client
inquiries (fees, statements, processing cutoffs, service teams) by searching
client agreements, SLAs, and service-team docs — plus read-only parameterized
tools over trade, account, and ticket systems of record.

## Why it exists — the problem

A client-service agent answering "what's the wire fee?" or "did this trade
settle?" digs through client agreements, SLA documents, and ticketing
systems by hand. Basic RAG helps but introduces three failure modes a
regulated service desk can't accept:

- **It leaks.** A naive retriever searches every client's documents — a
  rep asking about *their* client gets an answer sourced from *someone
  else's* agreement.
- **It guesses.** Thin or contradictory evidence still produces a fluent
  answer, and nothing tells the agent the answer is unreliable.
- **It can't be audited.** No citations, no record of what was retrieved,
  no trail when something goes wrong.

## What it does

Runs every question through a controlled LangGraph workflow —
`guardrail_in → route → retrieve → grade → synthesize → validate →
guardrail_out` — that enforces entitlements inside the retrieval query,
drafts only from entitled evidence (docs *and* read-only ops tools),
independently validates each claim against its citation, and abstains or
escalates instead of guessing.

## How it's useful — measurable, not asserted

- **Fewer wrong answers reaching clients:** vs `plain-hybrid-acl` — the
  strictest control (identical retrieval + entitlement enforcement, no
  agent loop) — `key_points_hit` 0.80 vs 0.50 and `refusal_correct`
  1.0 vs 0.67 (`eval/reports/compare_*.md`, Baseline compare page).
  `answerability_correct` is `n/a` for baselines: they emit no verdict
  to score against, same convention as the claim-support metrics.
- **Zero tenant leakage:** `acl_clean` 1.0 vs 0.25 for the ungated
  baseline — enforced as a query pre-filter and gated in CI on every PR.
- **Auditable by default:** every reply carries source citations, every
  request writes an audit trail, and low-confidence/conflicting answers
  auto-enqueue for a human reviewer.
- **Honest when it doesn't know:** abstains on insufficient evidence,
  surfaces conflicting sources with both citations, refuses unauthorized
  or injected requests — so the answer an agent reads can be acted on.

## Architecture

```
Streamlit UI ──► FastAPI /answer ──► authN/Z (user → entitlements)
                                     │ audit row per event ──► Postgres audit_events
                                     ▼
                       LangGraph agent (Langfuse trace per request)
                        guardrail_in → route →
                         ├─ doc tools  : pgvector dense (Titan/Bedrock) + OpenSearch BM25
                         │               → RRF → BGE cross-encoder rerank → top-8
                         └─ ops tools  : get_account / search_trades / search_tickets
                                         (rag_readonly role, fixed parameterized queries)
                        grade (rewrite once) → synthesize cited reply → guardrail_out
```

ACL is enforced as a **pre-filter** in pgvector (`WHERE`), OpenSearch
(`filter` clause), and ops SQL params — unauthorized docs never leave the
store. PII is redacted at ingest (Presidio, irreversible placeholders) and
re-scanned on the output draft.

**Multi-tenancy.** This is a multi-tenant RAG: every document carries
document-level permissions (`client_id` scope for tenant-owned docs,
`"*"` for bank-wide, plus role allowlists), and retrieval is
permission-aware — the caller's entitlement filter is injected into the
vector and lexical queries themselves, so cross-tenant chunks are never
fetched, let alone leaked into context.

### Tech stack

| layer | technology | role |
|---|---|---|
| UI | Streamlit (`src/ui/app.py`) | chat, agent-step view, eval dashboard, baseline compare |
| API | FastAPI + Pydantic (`/api/v1`) | authn/z, SSE streaming, idempotency, review queue |
| Agent workflow | LangGraph (`src/app/agent`) | bounded state machine — route → retrieve → grade → synthesize → validate |
| LLM | Anthropic Claude (LangChain) | routing + reply drafting; deterministic offline path without a key |
| Embeddings | Bedrock Titan v2 (prod) · `all-MiniLM-L6-v2` (local) | dense vectors for chunk + query embedding |
| Vector store | PostgreSQL + pgvector | dense search + ACL pre-filter + doc registry |
| Lexical store | OpenSearch | BM25 + ACL filter, mirrored per collection |
| Reranker | `BAAI/bge-reranker-base` cross-encoder | top-k rerank after RRF fusion |
| Ops data | Postgres (read-only `rag_readonly`) | accounts/trades/tickets, parameterized SELECT-only |
| PII | Presidio + spaCy | ingest, prompt, output, audit redaction |
| Ingestion | staged pipeline (`src/app/data/ingest`) | parse → validate → chunk → redact → embed → index; `.md`/`.txt`/`.csv`/`.pdf`/`.docx` input via pypdf + python-docx |
| Tracing | Langfuse | per-request LLM/step traces |
| Eval | custom harness + Ragas (`eval/`) | scenario suites, gates, agentic-vs-baseline compare |
| Deploy | Docker Compose (dev) · K8s/Kustomize on EKS (prod) | External Secrets + IRSA, HPA/PDB, smoke gates |
| CI | GitHub Actions (SHA-pinned) | lint/type/test + eval leak/threshold gates |
| Tooling | pytest · ruff · mypy | 100+ unit tests offline-capable; lint + typecheck |

## Objective

Production-oriented RAG a regulated client-service org could trust — where
trust is three measurable properties, not a vibe:

1. **Authorization by construction** — client entitlements enforced *inside*
   retrieval and every ops call; unauthorized data is never fetched. Verified
   by eval gates (`acl_clean` + `--fail-on-leak`), not just code review.
2. **Honesty over fluency** — every answer is classified (fully / partially /
   insufficient / conflicting / unauthorized); weak evidence → abstain,
   disagreement → surface both sides, hostile input → refuse.
3. **Provable quality** — claims independently validated against citations;
   deterministic eval suites + a plain-RAG baseline (`eval/compare.py`)
   measure whether the agent loop earns its complexity; a human-review
   queue catches what automation shouldn't decide.

Explicit non-goals today: knowledge-graph retrieval (this is not GraphRAG),
multi-turn conversation, semantic answer caching, and model-tier cost
routing — see `docs/WORKFLOW.md` for what the state machine does cover.

## Quickstart

Operational walkthrough — config, keys, input data, run modes, results:
**`docs/RUNBOOK.md`**.

```bash
make install            # venv + deps + spacy model
cp .env.example .env    # fill ANTHROPIC_API_KEY (+ AWS creds for Bedrock)
make dev-up             # postgres + opensearch
copilot corpus generate # synthetic docs + ops CSVs + facts.json + users.json
copilot db init         # ops schema + readonly role + CSV seed + audit table
copilot index --rebuild # chunk → redact → embed → both stores
copilot ask "What is the wire fee for acme?" --user u_agent1
```

Dev users (`db/seed/users.json`): `u_agent1`/`u_agent2` (agent, 4-client
books), `u_senior` (senior_agent) and `u_admin` (admin) — all 12 clients;
per-user keys `dev-key-*`. Full table + role-vs-scope semantics:
`docs/RUNBOOK.md` §2.

Full local stack (with UI + optional Langfuse):

```bash
docker compose up -d                                            # core
docker compose -f docker-compose.yml \
               -f docker-compose.observability.yml up -d        # + Langfuse :3000
```

API: `make run-api` → `POST /answer` with headers `X-API-Key`, `X-User-Id`.
UI: `make run-ui` → http://localhost:8501 (chat + eval dashboard).

## Evals

```bash
copilot evalset generate     # bootstrap ~85 questions → eval/golden_set.jsonl
# → human review pass, then:
copilot eval run --limit 10  # scores → eval/reports/eval_*.json|md

# system comparison (the headline number) — dev ×3, all 5 systems;
# the 12-row PR suite is a CI gate, too small for quality conclusions
python3 eval/runner.py --suite dev --system all --repeat 3 --no-ragas
```

Metrics: Ragas (faithfulness, answer_relevancy, context_precision/recall,
response_correctness) + custom citation P/R, per-stage retrieval recall
(dense / BM25 / fused / reranked), refusal accuracy, tool selection, and
zero-tolerance ACL-leakage / `must_not_contain` checks.

## Embedding providers

`EMBEDDING_PROVIDER=bedrock` uses Titan Embed v2 (needs AWS creds with the
model enabled in-region). `EMBEDDING_PROVIDER=local` uses a local
sentence-transformers model — handy for offline dev/tests.

## Deployment

- Images: `deploy/docker/` (backend, frontend); CI builds/pushes to ECR.
- K8s: `deploy/k8s/` kustomize — `overlays/staging` includes in-cluster
  Postgres/OpenSearch StatefulSets; `overlays/prod` expects RDS + Amazon
  OpenSearch Service (ConfigMap patch) and a managed/shared Langfuse.
- CI/CD: `.github/workflows/ci.yml` — test → build/push ECR (OIDC) →
  staging auto-deploy → prod via manual `workflow_dispatch`.
- Required GitHub secrets: `AWS_ROLE_ARN`, `EKS_CLUSTER`; app secrets live
  in `copilot-secrets` (template in `deploy/k8s/base/secret.yaml` — wire to
  External Secrets / Sealed Secrets for real use).

## Security notes

- Ops tools have **no free-text SQL path**: fixed parameterized templates,
  `rag_readonly` role, session READ ONLY, statement timeout, row cap.
- Ingest-time PII redaction is irreversible (no mapping persisted). Ops DB
  holds raw values; the output guardrail re-masks drafts. If authorized
  replies should show real names, swap to vault-based detokenization.
- Prototype auth is header-based (`X-API-Key` + `X-User-Id` →
  `db/seed/users.json`). Swap `security/authn.py` for OIDC/JWT in prod —
  the ACL filter interface is unchanged.

## Repo layout

See `src/app/` — layered: `core` (config/logging/DI) → `api` → `agent` →
`retrieval` + `tools/ops` → `security`, `observability`, `data`, `db`.
Evals in `eval/`, deploy in `deploy/`, seeds/migrations in `db/`.
