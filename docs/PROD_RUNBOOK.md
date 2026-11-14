# Production Runbook

Operate the agentic RAG copilot in production: architecture, prerequisites,
configuration, secrets, deploy/rollback, data operations, verification, and
incident response.

Author: venkateshwara doijode

Local/dev operations: `docs/RUNBOOK.md`. Design internals: `docs/WORKFLOW.md`.
Manifests: `deploy/k8s/`; pipeline: `.github/workflows/ci.yml`.

---

## 1. Architecture and environments

### Components

| Component | Runs where | Notes |
|---|---|---|
| **Backend API** (FastAPI, `app.main:app`) | **in-cluster** — `deployment/backend`, ns `agentic-rag`, :8000, HPA 2–8 + PDB | the LangGraph agent; serves `/api/v1/*` |
| **UI** (Streamlit, `src/ui/app.py`) | **in-cluster** — `deployment/streamlit`, :8501, behind ingress | prototype console + eval dashboard |
| **Postgres + pgvector** | **managed** — Amazon RDS (pgvector extension) | dense index + idempotency, audit_events, ingestion jobs, review_queue, doc registry |
| **OpenSearch** | **managed** — Amazon OpenSearch Service | BM25 index; endpoint injected via prod ConfigMap patch |
| **Ops DB** | **managed** — same RDS family, separate DB | read-only ops stores (accounts/trades/tickets) via `rag_readonly` role |
| **Bedrock** (Titan embeddings) | **AWS managed service** | `bedrock:InvokeModel` on `amazon.titan-embed-text-v2:0`, via IRSA |
| **Anthropic API** | **external SaaS** | Claude drafts/routing/judge; `ANTHROPIC_API_KEY` from AWS SM |
| **Langfuse** | **in-cluster or managed** | optional tracing; `LANGFUSE_HOST` points at `langfuse-web:3000` by default — disable by leaving keys empty |

Only the two app images run in EKS. Everything stateful or billable is a
managed service or SaaS — there are no in-cluster StatefulSets.

### Environments

| Env | Manifests | Trigger | Secrets source |
|---|---|---|---|
| staging | `deploy/k8s/overlays/staging` | every push to `main` | `base/secret.yaml` placeholders (dev-only) |
| prod | `deploy/k8s/overlays/prod` | `workflow_dispatch` only | AWS Secrets Manager via `ExternalSecret` |

Both deploy the exact `${{ github.sha }}` image built by the `build` job —
never a floating tag.

## 2. Prerequisites

| Requirement | Detail |
|---|---|
| AWS account + region | `us-east-1` (base ConfigMap) |
| EKS cluster | name in repo secret `EKS_CLUSTER`; namespace `agentic-rag` |
| ECR registry | outputs of `build` job (`needs.build.outputs.registry`) |
| IRSA role | `eks.amazonaws.com/role-arn` on `serviceaccount/copilot-backend` → role granting `bedrock:InvokeModel` for the Titan embed model |
| External Secrets Operator | installed in-cluster + a `ClusterSecretStore` named `aws-secrets-manager` bound to IRSA — required by `overlays/prod/externalsecret.yaml` |
| GitHub environments | `staging` and `production` — configure required reviewers on `production` (that's the prod approval gate) |
| Repo secrets | `AWS_ROLE_ARN`, `EKS_CLUSTER`; optional `SMOKE_API_KEY`/`SMOKE_USER_ID` for the answer probe |

**Approvals needed before first prod deploy**

- **Security review** — auth model (fixture API keys → OIDC/JWT is the
  documented prod target), ACL pre-filters, PII redaction coverage,
  `ENVIRONMENT=production` guard (§3 auth notes).
- **Model-risk review** — LLM usage policy for Anthropic, offline-mode
  behavior, refusal/abstain semantics, human-review queue workflow.
- **Data governance** — corpus provenance, entitlement source-of-truth
  for `users.json`/ops CSVs, PII handling sign-off.

## 3. Configuration reference

Settings load from env/`.env` (`src/app/core/config.py`). Prod values come
from the `copilot-config` ConfigMap + `copilot-secrets` Secret. "Secret?"
marks values that must live in AWS SM, never in ConfigMaps/manifests.

### LLM

| name | purpose | default | prod value | secret? |
|---|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude key — drafts, routing, Ragas judge | `""` (offline draft) | SM `agentic-rag/prod/anthropic-api-key` | **yes** |
| `ANTHROPIC_MODEL` | model id | `claude-sonnet-4-20250514` | ConfigMap, same | no |
| `LLM_TIMEOUT_S` | per-call timeout | `30` | keep | no |

### Embeddings

| name | purpose | default | prod value | secret? |
|---|---|---|---|---|
| `EMBEDDING_PROVIDER` | `bedrock` \| `local` | `bedrock` | `bedrock` | no |
| `BEDROCK_EMBED_MODEL` | Titan model id | `amazon.titan-embed-text-v2:0` | same | no |
| `AWS_REGION` | Bedrock region | `us-east-1` | same | no |
| `AWS_PROFILE` | local-only credential profile | `—` | unset (IRSA instead) | no |
| `EMBEDDING_DIM` | vector dim | `1024` | `1024` for Titan v2 | no |
| `LOCAL_EMBED_MODEL` | offline embedder | `all-MiniLM-L6-v2` | unused in prod | no |

### Datastores

| name | purpose | default | prod value | secret? |
|---|---|---|---|---|
| `DATABASE_URL` | pgvector store + jobs/audit/idempotency | dev DSN | SM `.../database-url` (RDS endpoint) | **yes** |
| `OPS_DATABASE_URL` | read-only ops DB (`rag_readonly`) | dev DSN | SM `.../ops-database-url` | **yes** |
| `OPS_ADMIN_DATABASE_URL` | ops schema init/seeding | dev DSN | SM `.../ops-admin-database-url` | **yes** |
| `OPENSEARCH_URL` | BM25 store | `http://localhost:9200` | ConfigMap patch → Amazon OpenSearch Service endpoint | no |

### Retrieval

| name | purpose | default | prod value |
|---|---|---|---|
| `DENSE_TOP_K` / `LEXICAL_TOP_K` | per-leg candidates | `20`/`20` | keep |
| `RRF_K` / `RRF_*_WEIGHT` | fusion constant/weights | `60`/`1.0`/`1.0` | keep |
| `RERANK_CANDIDATES` | reranker input size | `20` | keep |
| `FINAL_TOP_K` | evidence size into synthesis | `5` | keep |
| `RERANKER_MODEL` | cross-encoder | `BAAI/bge-reranker-base` | keep |
| `RETRIEVAL_STRATEGY` | `dense\|bm25\|hybrid\|hybrid_rerank\|auto` | `hybrid_rerank` | keep |
| `DOC_EXPIRY_MODE` | `exclude` \| `down_rank` expired docs | `exclude` | keep |
| `PARENT_CHILD_ENABLED` | parent-section expansion | `false` | keep |

### Agent budget (hard caps per request)

| name | purpose | default | prod value |
|---|---|---|---|
| `AGENT_MAX_ITERATIONS` | graph loops | `6` | keep |
| `AGENT_MAX_TOOL_CALLS` | ops invocations | `10` | keep |
| `AGENT_MAX_TOKENS` | soft token budget | `20000` | keep |
| `AGENT_MAX_ELAPSED_MS` | wall-clock | `60000` | keep |
| `AGENT_MAX_COST_USD` | soft spend cap | `0.25` | keep |
| `AGENT_MAX_RETRIES` | rewrite bound | `2` | keep |

### Ops tools

| name | purpose | default |
|---|---|---|
| `OPS_ROW_CAP` | rows per tool call | `100` |
| `OPS_STATEMENT_TIMEOUT_MS` | per-statement | `10000` |
| `OPS_POOL_SIZE` / `OPS_POOL_MAX_OVERFLOW` / `OPS_POOL_TIMEOUT_S` | pool | `5`/`5`/`10` |

### Auth / review

| name | purpose | default | prod value | secret? |
|---|---|---|---|---|
| `API_KEYS` | valid caller keys (prototype) | dev fixture keys | **not yet externalized** — prod target is OIDC/JWT | yes (when set) |
| `USERS_FILE` | user/entitlement fixture | `db/seed/users.json` | same gap as above | no |
| `ADMIN_TOKEN` | `/admin/*` bearer | `admin-dev-token` | SM `.../admin-token` | **yes** |
| `HUMAN_REVIEW_ENABLED` | review-queue writes | `true` | `true` | no |
| `REVIEW_CITATION_THRESHOLD` | review trigger | `0.5` | keep | no |
| `ENVIRONMENT` | `development`\|`production` guard | `development` | set `production` **only after** API_KEYS/USERS_FILE are externalized — the guard refuses dev fixtures | no |

### Tracing / API / ingestion

| name | purpose | default | prod value | secret? |
|---|---|---|---|---|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | tracing auth | `""` (off) | SM `.../langfuse-*` | **yes** |
| `LANGFUSE_HOST` | collector URL | `http://localhost:3000` | ConfigMap `langfuse-web:3000` | no |
| `API_HOST` / `API_PORT` | bind | `0.0.0.0`/`8000` | keep | no |
| `REQUEST_TIMEOUT_S` | request ceiling | `60` | keep | no |
| `IDEMPOTENCY_TTL_S` | replay window | `86400` | keep | no |
| `STREAM_ENABLED` | SSE on/off | `true` | keep | no |
| `INDEX_MODE` | `incremental` \| `full` | `incremental` | keep | no |
| `INGEST_MAX_ATTEMPTS` | per-doc retries before DLQ | `3` | keep | no |
| `DOC_VERSION_POLICY` | `latest` \| `keep_all` | `latest` | keep | no |
| `API_URL` | UI → backend base URL | `http://localhost:8000` | `http://backend:8000` | no |

## 4. Secrets

All prod secrets live in AWS Secrets Manager as individual keys consumed
by `overlays/prod/externalsecret.yaml` (refreshInterval `1h`):

| AWS SM key | Injected as | Owner |
|---|---|---|
| `agentic-rag/prod/anthropic-api-key` | `ANTHROPIC_API_KEY` | AI platform team |
| `agentic-rag/prod/database-url` | `DATABASE_URL` | DBA (RDS creds) |
| `agentic-rag/prod/ops-database-url` | `OPS_DATABASE_URL` | DBA (`rag_readonly`) |
| `agentic-rag/prod/ops-admin-database-url` | `OPS_ADMIN_DATABASE_URL` | DBA |
| `agentic-rag/prod/langfuse-public-key` | `LANGFUSE_PUBLIC_KEY` | Observability |
| `agentic-rag/prod/langfuse-secret-key` | `LANGFUSE_SECRET_KEY` | Observability |
| `agentic-rag/prod/admin-token` | `ADMIN_TOKEN` | Security |

**Rotation:** update the SM value → External Secrets syncs within the
hour → force immediate pickup with
`kubectl rollout restart deployment/backend -n agentic-rag`.
Revoke the old value only *after* pods confirm healthy (`rollout status`).

## 5. Deploy and rollback

### Pipeline (`ci.yml`)

```
lint/typecheck/unit ─┐
integration tests   ─┼→ eval gate (PR suite, --fail-on-leak
pip-audit + Trivy   ─┘   --fail-on-thresholds)
        │  └→ build + push backend/frontend images (SHA-tagged, Trivy scan)
        ├→ deploy-staging  (env: staging, every push to main)
        │     └→ rollout status + smoke.sh staging
        └→ deploy-prod     (env: production, workflow_dispatch ONLY)
              └→ rollout status + smoke.sh prod
```

- **Prod approval** = the GitHub `production` environment's required
  reviewers — configure at least one. `deploy-prod` pins `${{ github.sha }}`,
  is `cancel-in-progress: false`, and runs `deploy/scripts/smoke.sh prod`.
- No `|| true` anywhere — a failed apply, rollout, or smoke check fails the job.

### Rollback

```bash
kubectl rollout history deployment/backend -n agentic-rag
kubectl rollout undo deployment/backend   -n agentic-rag   # or to-revision=N
kubectl rollout undo deployment/streamlit -n agentic-rag
```

For DB-migrating releases, roll forward instead (see §6) — `rollout undo`
does not revert schema/data changes.

## 6. Data operations

| Task | How |
|---|---|
| Full re-index | `copilot index --rebuild` — drops and rebuilds both stores |
| Incremental index | `copilot index` (default) — hash-diff vs `document_registry`; changed docs propagate deletes before re-index |
| Drop-in docs | files under `data/real/<collection>/` picked up alongside synthetic (see `data/real/`) |
| Job status / failures | `copilot jobs list`, `copilot jobs dlq [--job <id>]`; retry a dead letter via `POST /admin/ingest/dlq/{path}/retry` (admin token) |
| Schema changes | Alembic under `db/migrations/` (`env.py` reads `DATABASE_URL`); ops schema/seed: `copilot db init` |
| Backups | RDS automated snapshots + OpenSearch Service snapshots — recovery is a managed-service restore + `copilot index --rebuild` |
| Audit retention | `audit_events` in ragdb — append-only, no built-in TTL; set an RDS-side retention/archival policy per compliance |

## 7. Verify

1. **Liveness/readiness** — `GET /api/v1/health` (200) and `GET /api/v1/ready`
   (200 healthy / 503 degraded); `GET /api/v1/deps` for per-store latency.
2. **Smoke script** — `deploy/scripts/smoke.sh <env>` runs in-cluster probes:
   health, readiness, and an authenticated `POST /api/v1/answer` when
   `SMOKE_API_KEY`/`SMOKE_USER_ID` are set. Runs automatically post-deploy.
3. **Post-deploy eval** — quality gate before calling it done:

   ```bash
   python3 eval/runner.py --suite pr --fail-on-leak --fail-on-thresholds
   # then the baseline control for the record:
   python3 eval/runner.py --suite pr --system all
   ```

   Zero ACL leaks + all `eval/gates.yaml` floors must hold; compare
   `eval/reports/eval_<ts>.json` `version_snapshot` to the previous run
   (model, prompt, dataset, git SHA).

## 8. Monitor and respond

| Signal | Where | What to watch |
|---|---|---|
| Traces | Langfuse | per-request node spans, LLM usage, refusal/abstain rates |
| Audit | `audit_events` table | per-node decisions, ACL denials, tool calls — replayable trail |
| Logs | structlog JSON stdout → cluster log pipeline | `eval.acl_leakage`, `threshold_breach`, store errors, budget breaches |
| Health | `/ready`, `/deps` | 503s, rising per-store latency |
| Scaling | HPA 2–8 + PDB | sustained max-replica state, eviction events |
| Review queue | `review_queue` table | pending items (partial/conflicting/low-support/high-impact) |

**Recommended alerts:** readiness 503 sustained >5 min; HTTP 5xx rate;
`acl_leaks > 0` in any eval run or audit trail (page — that's the security
contract); hard budget breaches; eval-gate failure in CI.

**Incident steps**

1. **Degraded deps** — `/deps` shows which store is down; the agent
   abstains (`insufficient_evidence`) rather than guessing, so impact is
   bounded. Fix the store; no app change needed.
2. **Bad deploy** — `kubectl rollout undo` (§5); verify with smoke.sh.
3. **Credential compromise** — rotate the SM key (§4), `rollout restart`,
   revoke old; for `API_KEYS` also rotate users-file entries.
4. **Data leak suspicion** — audit trail first (which principal, which
   docs); ACL denials are expected noise, retrievals outside entitlements
   are not — treat as sev-1, revoke the key, re-run `--fail-on-leak` eval.
