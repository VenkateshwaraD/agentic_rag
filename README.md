# Agentic RAG Client-Service Copilot

A compliance-aware agentic RAG copilot that drafts cited replies to client
inquiries (fees, statements, processing cutoffs, service teams) by searching
client agreements, SLAs, and service-team docs — plus read-only parameterized
tools over trade, account, and ticket systems of record.

## Architecture

```
Streamlit UI ──► FastAPI /answer ──► authN/Z (user → entitlements)
                                     │ audit row per event ──► Postgres audit_events
                                     ▼
                       LangGraph agent (Langfuse trace per request)
                        guardrail_in → route →
                         ├─ doc tools  : pgvector dense (Titan/Bedrock) + OpenSearch BM25
                         │               → RRF → BGE cross-encoder rerank → top-5
                         └─ ops tools  : get_account / search_trades / search_tickets
                                         (rag_readonly role, fixed parameterized queries)
                        grade (rewrite once) → synthesize cited reply → guardrail_out
```

ACL is enforced as a **pre-filter** in pgvector (`WHERE`), OpenSearch
(`filter` clause), and ops SQL params — unauthorized docs never leave the
store. PII is redacted at ingest (Presidio, irreversible placeholders) and
re-scanned on the output draft.

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
