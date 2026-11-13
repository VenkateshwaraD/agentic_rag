#!/usr/bin/env bash
# Post-deploy smoke tests — liveness, readiness, and one authenticated answer.
#
# Usage: deploy/scripts/smoke.sh <staging|prod>
# Env:   COPILOT_BASE_URL (default: in-cluster service DNS),
#        SMOKE_API_KEY / SMOKE_USER_ID for the answer probe (skipped when unset)
set -euo pipefail

ENV="${1:?usage: smoke.sh <staging|prod>}"
NAMESPACE="agentic-rag"
BASE="${COPILOT_BASE_URL:-http://backend.${NAMESPACE}.svc.cluster.local:8000}"

echo "[smoke] env=$ENV base=$BASE"

# in-cluster probe via a throwaway pod so we don't depend on ingress DNS
probe() {
    local path="$1" expected="${2:-200}"
    kubectl run "smoke-$RANDOM" --rm -i --restart=Never -n "$NAMESPACE" \
        --image=curlimages/curl:8.10.1 --quiet -- \
        curl -fsS -o /dev/null -w "%{http_code}" --max-time 10 "$BASE$path" \
        | grep -q "$expected"
}

echo "[smoke] liveness /health"
probe "/api/v1/health" 200 || probe "/health" 200

echo "[smoke] readiness /ready"
probe "/api/v1/ready" "200\|503" || { echo "[smoke] FAIL: readiness probe unreachable"; exit 1; }

if [ -n "${SMOKE_API_KEY:-}" ] && [ -n "${SMOKE_USER_ID:-}" ]; then
    echo "[smoke] authenticated answer probe"
    kubectl run "smoke-answer-$RANDOM" --rm -i --restart=Never -n "$NAMESPACE" \
        --image=curlimages/curl:8.10.1 --quiet -- \
        curl -fsS --max-time 60 -X POST "$BASE/api/v1/answer" \
            -H "X-API-Key: $SMOKE_API_KEY" -H "X-User-Id: $SMOKE_USER_ID" \
            -H 'Content-Type: application/json' \
            -d '{"question": "health check — reply briefly"}' \
        | grep -q '"reply"' || { echo "[smoke] FAIL: answer probe"; exit 1; }
else
    echo "[smoke] SMOKE_API_KEY/SMOKE_USER_ID unset — skipping answer probe"
fi

echo "[smoke] all checks passed for $ENV"
