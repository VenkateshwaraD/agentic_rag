# Baseline Comparison Guide

How to run the plain-RAG baseline against the agentic pipeline, what to
configure, and how to read the comparison report.

Author: venkateshwara doijode

For the *why* — what the baseline is designed to prove and where the delta
should (and shouldn't) appear — see `docs/RAGAS.md` §8. This document is
the operational reference.

---

## 1. Systems under test

`src/app/baseline.py` implements a deliberately naive RAG pipeline:
embed question → search every collection like one merged index → draft
once → reply. **No routing, no tools, no grading, no rewriting, no
validation, no guardrails, no answerability classification** — it always
reports `fully_answerable` and never refuses or abstains.

The control matrix (2 retrieval strategies × ACL on/off) isolates what
the agent loop adds from what retrieval or enforcement adds:

| `--system` value | retrieval | ACL pre-filter | what it isolates |
|---|---|---|---|
| `agentic` | hybrid + rerank | yes | the full system (reference) |
| `plain` | dense top-5 | **none** | realistic naive RAG — leaks like a real one |
| `plain-acl` | dense top-5 | yes | naive RAG + entitlement enforcement |
| `plain-hybrid` | hybrid + rerank | none | same retrieval as agentic, no agent loop |
| `plain-hybrid-acl` | hybrid + rerank | yes | strictest control — agentic minus the loop |
| `all` | — | — | runs all five + writes the comparison report |

Everything else is shared: same scenario rows, same users, same corpus,
same stores, same embedding model, same offline/LLM draft machinery,
same metric code. Differences in the report are attributable to the
missing machinery, not to a different setup.

## 2. Prerequisites

Same as a normal eval run — stores up and indexed:

```bash
make dev-up                                    # Postgres + OpenSearch
copilot db init                                # ops schema + seed CSVs
copilot index                                  # ingest corpus into both stores
export EMBEDDING_PROVIDER=local                # offline embeddings (no AWS/LLM keys needed)
```

## 3. Running it

```bash
# all five systems on one suite + comparison report (~15-20 min on PR)
python3 eval/runner.py --suite pr --system all

# a single system (same report shape as an agentic run)
python3 eval/runner.py --suite dev --system plain-hybrid-acl

# through the copilot CLI (same flags)
copilot eval run --suite pr --system all --no-ragas

# merge two existing per-system reports by hand
python3 eval/compare.py eval/reports/eval_<ts>.json eval/reports/eval_plain_<ts>.json
```

### Flag reference

| flag | meaning |
|---|---|
| `--system` | which pipeline answers each row (`agentic`, `plain*`, or `all`) |
| `--suite` | `pr` \| `dev` \| `holdout` \| `golden`; `--golden` picks a file |
| `--limit N` | run only the first N rows (smoke checks) |
| `--repeat N` | rerun each row N times; report adds ±95% CI |
| `--no-ragas` | skip the Ragas judge (no `ANTHROPIC_API_KEY` needed) |
| `--fail-on-leak` | exit non-zero if any retrieved doc falls outside the user's entitlements |
| `--fail-on-thresholds` | exit non-zero on `eval/gates.yaml` / Ragas floor breaches |

### What `--system all` does differently

- Runs `agentic` first, then each `plain*` variant on the **identical**
  scenario list, and writes one tagged report per system:
  `eval_<ts>.{json,md}` for agentic, `eval_<system>_<ts>.{json,md}` for
  baselines.
- Then calls `eval/compare.py` to write `compare_<ts>.{json,md}`.
- **Leak/threshold gates are not applied inside the loop** — the no-ACL
  baselines leak *by design*, and `--fail-on-leak` on `all` would abort
  before the ACL variants run. Gates still work on single-system runs
  (e.g. `--system agentic --fail-on-leak`, which is what CI does).

## 4. Configuration that affects the baseline

The baseline reads the same settings as everything else
(`src/app/core/config.py`, all via env):

| setting | default | effect on baseline |
|---|---|---|
| `EMBEDDING_PROVIDER` | `bedrock` | `local` for offline runs; shared with agentic |
| `RETRIEVAL_STRATEGY` | `hybrid_rerank` | only used by `agentic` — `plain*` variants pick their own (see table) |
| `DOC_EXPIRY_MODE` | `exclude` | freshness filter applies to baselines too |
| `API_KEYS`, `USERS_FILE` | dev fixtures | same users/entitlements for both systems |
| `ANTHROPIC_API_KEY` | — | set → both sides draft with the LLM; unset → deterministic offline draft |
| `OPENSEARCH_URL`, `DATABASE_URL` | localhost | same stores/indexes for both |

Baseline answer function signature (`src/app/baseline.py`):

```python
plain_answer(question, principal, request_id,
             strategy="dense", use_acl=True, top_k=5) -> dict
```

`--system` maps onto `strategy`/`use_acl`; the output dict mirrors
`graph.answer()` (`reply`, `citations`, `retrieved`, `tool_calls`,
`refusal`, `abstained`, `answerability`, `validation`, `errors`, …) so
every metric in `eval/metrics.py` applies unchanged.

## 5. What the output looks like

A `compare_<ts>.md` has three parts. Real excerpt from
`eval/reports/compare_20261114_044827.md` (PR suite):

```text
ACL leaks: {'agentic': 0, 'plain': 9, 'plain-acl': 0,
            'plain-hybrid': 9, 'plain-hybrid-acl': 0}
```

**Summary table** — mean metric per system (`n` sub-rows show coverage):

| metric | agentic | plain | plain-acl | plain-hybrid | plain-hybrid-acl |
|---|---|---|---|---|---|
| acl_clean | 1.0 | 0.25 | 1.0 | 0.25 | 1.0 |
| answerability_correct | 0.8333 | 0.25 | 0.25 | 0.25 | 0.25 |
| key_points_hit | 0.8 | 0.7 | 0.7 | 0.5 | 0.5 |
| refusal_correct | 1.0 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| citation_completeness | 0.9417 | n/a | n/a | n/a | n/a |
| unsupported_claim_rate | 0.0312 | n/a | n/a | n/a | n/a |

**Δ vs agentic** — same table as signed deltas, negative = baseline loses.

**Per-category tables** — the "gap" metrics broken out by scenario
category (`doc_factual`, `tool_lookup`, `ambiguous`, `conflicting`,
`unauthorized`, `adversarial`, …). This is the section that answers
*where* the agent earns its keep: expect ties on `doc_factual` and gaps
everywhere else.

## 6. Reading the results

- **`n/a` means "not measured", not zero.** Baselines never run the
  validation node, so `citation_correctness`, `citation_completeness`,
  and `unsupported_claim_rate` are undefined for them — the report shows
  `n/a` and excludes them from deltas rather than awarding a free 1.0.
- **`recall_*` of ~0.67 for baselines is by construction.** Naive RAG
  merges all collections into one global top-5 — the agent keeps
  per-collection evidence and expands parents, so it covers more of the
  `required_doc_ids`. Not a bug in the baseline; a real architectural
  difference.
- **`no_banned_content` may tie at 1.0 offline.** The deterministic
  offline draft only echoes ~300 characters per chunk, and the planted
  injection sits deeper than that — nothing to echo. With
  `ANTHROPIC_API_KEY` set, both systems use the LLM and the guardrail
  delta shows up for real.
- **`tool_selection` < 1.0 for baselines is measurement, not
  capability** — they cannot call tools at all; rows where the question
  is answerable from documents alone still "select" correctly by luck.
- **Watch `plain-hybrid-acl` for the honest verdict.** It has identical
  retrieval and enforcement — any remaining gap is provably the agent
  loop (routing, tools, grading, validation, answerability, guardrails).

## 7. Where things live

| path | role |
|---|---|
| `src/app/baseline.py` | `plain_answer()` + merged-index search |
| `eval/runner.py` | `--system` dispatch, per-system tagged reports |
| `eval/compare.py` | merges tagged reports → `compare_<ts>.{json,md}` |
| `eval/metrics.py` | shared metrics; `n/a` semantics for unmeasured dimensions |
| `eval/scenarios/*.jsonl` | shared question set (hard rows: multi-hop, conflicting, ambiguous, unanswerable, adversarial) |
| `eval/reports/compare_*.md` | committed example output |
| `tests/unit/test_baseline_compare.py` | output-shape, ACL pass-through, merge-order, compare-math tests |
