# Ragas Evaluation Guide

How the Ragas LLM-judged metrics are wired into this project's eval harness,
how to configure them, add test cases, run a suite, and review the results
and evidence.

Author: venkateshwara doijode

---

## 1. What gets measured

The runner feeds each scenario row through the agent (`answer()`), then
hands Ragas four fields per row (`eval/runner.py::_score_rows`):

| Ragas field | Populated from |
|---|---|
| `user_input` | the scenario `question` |
| `response` | the agent's final `reply` |
| `retrieved_contexts` | text of every retrieved chunk (or `"<none>"`) |
| `reference` | the scenario's `expected_answer` |

Metrics computed (`eval/metrics.py::ragas_metrics`):

| Metric | Judge asks |
|---|---|
| `faithfulness` | is every claim in the answer supported by retrieved context? |
| `answer_relevancy` | does the answer address the question? |
| `context_precision` | are the retrieved chunks relevant, ranked well? |
| `context_recall` | does retrieval cover what the reference needs? |
| `answer_correctness` | semantic similarity of answer vs `reference` |

All five are LLM-judged — they need a working judge model, not just embeddings.

## 2. Configuration

Ragas only runs when an Anthropic key is configured; otherwise
`ragas_metrics` returns `{}` and the report simply has no `ragas_*` keys.

```bash
# required — gates the whole ragas block
export ANTHROPIC_API_KEY=sk-ant-...

# optional — judge model (default comes from Settings.anthropic_model)
export ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# embeddings for answer_relevancy — uses the app's configured embedder
export EMBEDDING_PROVIDER=local          # or bedrock in prod

# stores must be reachable for the agent to produce real answers
export DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb
export OPENSEARCH_URL=http://localhost:9200
export OPS_DATABASE_URL=postgresql://rag_readonly:readonly@localhost:5432/opsdb
export OPS_ADMIN_DATABASE_URL=postgresql://rag:rag@localhost:5432/opsdb
```

Same key also enables real LLM synthesis — **without it the replies are
`[offline-draft]` heuristics**, and Ragas ends up judging bullet echoes of
the retrieved chunks, not real prose. Scores from offline runs are not
representative; use them only to smoke-test the plumbing.

Internals, for reference:

- `eval/metrics.py::ragas_metrics` builds `ChatAnthropic(model=s.anthropic_model,
  api_key=s.anthropic_api_key)` wrapped in `LangchainLLMWrapper`, and wraps
  `get_embedder()` in `LangchainEmbeddingsWrapper`.
- Any exception (API drift, rate limit, missing creds) is caught and
  reported as `{"ragas_error": "..."}` in the summary — the run never
  crashes on a Ragas failure.
- Ragas runs once per eval invocation even with `--repeat N` (repeat runs
  measure variance of the deterministic metrics, not of the judge).

## 3. Adding test cases

Scenarios are JSONL files. Suites live in `eval/scenarios/`:

| Suite | File | Purpose |
|---|---|---|
| `pr` | `scenarios/pr.jsonl` | small fast set run in PR CI |
| `dev` | `scenarios/dev.jsonl` | broader development set |
| `holdout` | `scenarios/holdout.jsonl` | release-gating set — do not tune to it |
| `golden` | `golden_set.jsonl` | generated ~85-question set (`generate_evalset.py`) |

Add one JSON object per line:

```json
{
  "id": "DEV-023",
  "question": "What is the SEV-1 first-response target for treasury services?",
  "category": "doc_factual",
  "user_id": "u_agent1",
  "expected_answer": "The SEV-1 first-response target is 15 minutes.",
  "key_points": ["15 minutes"],
  "required_doc_ids": ["SLA-004"],
  "required_tool_calls": [],
  "must_refuse": false,
  "must_not_contain": [],
  "expected_answerability": "fully_answerable"
}
```

Field semantics:

| Field | Used for |
|---|---|
| `question` | sent to the agent; also Ragas `user_input` |
| `expected_answer` | Ragas `reference` — required for `answer_correctness`/`context_recall` to score well |
| `required_doc_ids` | citation precision/recall + per-stage retrieval recall |
| `required_tool_calls` | tool-selection metric (e.g. `["accounts","tickets"]`) |
| `must_refuse` | refusal/adversarial scenarios |
| `must_not_contain` | strings that must never appear in the reply (PII, secrets) |
| `expected_answerability` | `fully_answerable` / `partially_answerable` / `insufficient_evidence` / `conflicting_evidence` / `unauthorized` |
| `category` | `doc_factual`, `multi_doc`, `multi_hop`, `adversarial`, `unanswerable`, `conflicting_evidence`, `expired_doc`, `ops_tool` — drives per-category rollup |
| `user_id` | principal from `db/seed/users.json` — pick one whose `client_ids` match the scenario |

Tips:

- For `context_recall`/`answer_correctness` write `expected_answer` as a
  complete, factual sentence — it's the judge's ground truth.
- For ACL tests use a `user_id` that is *not* entitled to the named client.
- Verify expectations against `data/synthetic/facts.json` before committing
  a scenario — drift between scenarios and generated data is the top cause
  of false failures.

## 4. Running

```bash
# full dev suite with ragas (default)
python3 eval/runner.py --suite dev

# smoke-test plumbing on a few rows
python3 eval/runner.py --suite pr --limit 3

# deterministic metrics only — no judge calls, no cost
python3 eval/runner.py --suite dev --no-ragas

# variance across 3 runs (ragas still only runs once)
python3 eval/runner.py --suite dev --repeat 3

# release gate — exit 1 on any ACL leak
python3 eval/runner.py --suite holdout --fail-on-leak

# ad-hoc file
python3 eval/runner.py --golden path/to/my_scenarios.jsonl
```

Requires Docker services up (`sudo docker compose up -d postgres opensearch`),
the DB seeded (`copilot db init`), and the corpus indexed (`copilot index`).

The command prints the JSON report path, e.g.
`eval/reports/eval_20261113_142715.json`.

## 5. Viewing results

Each run writes two files to `eval/reports/`:

- `eval_<ts>.json` — machine-readable: `summary`, `by_category`,
  `failures`, `n_questions`, `acl_leaks`, `version_snapshot` (model /
  embedding / dataset / git-sha stamps), `variance` when `--repeat > 1`.
- `eval_<ts>.md` — the same data rendered for humans.

Ragas scores appear in `summary` (and `by_category`) prefixed with `ragas_`:

```json
{
  "summary": {
    "citation_precision": 0.53,
    "acl_clean": 1.0,
    "ragas_faithfulness": 0.87,
    "ragas_answer_relevancy": 0.82,
    "ragas_context_precision": 0.74,
    "ragas_context_recall": 0.79,
    "ragas_answer_correctness": 0.68
  }
}
```

If `ragas_*` keys are absent → the judge never ran (no API key).
If `ragas_error` is present → the judge ran and failed; the string says why.

## 6. Reviewing test evidence

The report's `failures` array lists every row that scored below max on any
metric — id, category, question, full reply text, and which metrics missed.

Per-row evidence beyond the report:

- **Audit trail** — every eval row runs with `request_id = eval-<ID>-<hex>`;
  query `audit_events` for that id to see the exact node-by-node payload:

  ```sql
  SELECT event_type, payload, latency_ms
  FROM audit_events WHERE request_id LIKE 'eval-DEV-004-%' ORDER BY id;
  ```

- **Review queue** — answers flagged ambiguous/high-impact/failed-validation
  land in `review_queue`; list via `GET /api/v1/admin/review`
  (`X-Admin-Token`) or inspect the table directly.

- **Retrieval diagnostics** — each answer carries per-collection
  `diagnostics` (dense_ids / bm25_ids / fused_ids / reranked_ids, counts,
  stale/parent-expansion flags). Reproduce a row live:

  ```bash
  curl -X POST http://localhost:8090/api/v1/answer \
    -H "X-API-Key: dev-key-1" -H "X-User-Id: u_agent1" \
    -H "Content-Type: application/json" \
    -d '{"question": "<the failing question>", "client_id": "<client>"}'
  ```

- **Validation detail** — the answer's `validation` field shows per-claim
  `cited`/`supported` verdicts, `invalid_citations`, and `contradictions`
  — this is the local analogue of `faithfulness`; cross-check it when
  Ragas faithfulness disagrees.

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| No `ragas_*` keys in report | `ANTHROPIC_API_KEY` unset — judge skipped by design |
| `ragas_error` in summary | API/network/key failure — read the message |
| `context_recall` low | reference answer states facts the index doesn't contain — check `data/synthetic/facts.json` |
| `answer_correctness` low offline | expected — heuristic drafts aren't real answers; rerun with an LLM configured |
| Import error on `ragas.metrics` | version drift — code tries `ragas.metrics.collections` (>=0.4) then `ragas.metrics`; check installed version |
