# Deployment runbook

## Environments

| Env     | Manifests                          | Trigger                    |
|---------|------------------------------------|----------------------------|
| staging | `deploy/k8s/overlays/staging`      | every push to `main`       |
| prod    | `deploy/k8s/overlays/prod`         | manual `deploy-prod` job   |

Both deploy the exact `$CI_COMMIT_SHA` image built by the `build` job —
never a floating tag.

## Secrets

- **staging**: `base/secret.yaml` placeholder values (dev-only, rotate freely).
- **prod**: `ExternalSecret` (`overlays/prod/externalsecret.yaml`) pulls from
  AWS Secrets Manager via the External Secrets Operator. No plaintext secrets
  are committed. Required AWS SM keys: `agentic-rag/prod/{anthropic-api-key,
  database-url, ops-database-url, ops-admin-database-url, langfuse-public-key,
  langfuse-secret-key, admin-token}`.

## Health verification (automated)

`deploy/scripts/smoke.sh <env>` runs post-deploy in CI:

1. `GET /api/v1/health` — liveness
2. `GET /api/v1/ready` — readiness (503 tolerated = degraded but reachable)
3. `POST /api/v1/answer` — authenticated end-to-end probe, when
   `SMOKE_API_KEY`/`SMOKE_USER_ID` repo secrets are set

`kubectl rollout status` gates every apply — a stuck rollout fails the job.

## Manual deploy

```bash
cd deploy/k8s/overlays/<env>
kustomize edit set image \
  ECR_REGISTRY/agentic-rag-backend=<registry>/agentic-rag-backend:<sha> \
  ECR_REGISTRY/agentic-rag-frontend=<registry>/agentic-rag-frontend:<sha>
kubectl apply -k .
kubectl rollout status deployment/backend -n agentic-rag --timeout=300s
```

## Rollback

Kubernetes keeps rollout history:

```bash
# inspect history
kubectl rollout history deployment/backend -n agentic-rag

# roll back to previous revision
kubectl rollout undo deployment/backend -n agentic-rag
kubectl rollout undo deployment/streamlit -n agentic-rag

# roll back to a specific revision
kubectl rollout undo deployment/backend -n agentic-rag --to-revision=<N>
```

Or redeploy a known-good SHA: play the manual `deploy-prod` job on that
commit's pipeline, or
`kustomize edit set image ... :<known-good-sha>` + apply, as above.

Verify after rollback:

```bash
kubectl rollout status deployment/backend -n agentic-rag
deploy/scripts/smoke.sh prod
```

## CI gates (release-blocking)

- `test` — ruff, format, mypy, unit tests
- `security-scan` — pip-audit + trivy filesystem scan (CRITICAL/HIGH fail)
- `build` — trivy image scan on both images (CRITICAL/HIGH fail)
- `integration-test` — pytest integration + PR eval suite with the
  zero-tolerance ACL-leakage check (`--fail-on-leak`)
- deploy jobs — rollout-status checks + smoke tests

A red gate anywhere stops the pipeline before deploy.
