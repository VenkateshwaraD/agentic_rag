# Ragas Evaluation Guide

**Audience:** everyone — engineers, reviewers, and non-technical stakeholders.
Each section starts with a plain-language explanation; the technical commands
and file paths follow in *"For engineers"* blocks.

Author: venkateshwara doijode

---

## 0. The 60-second version

Our copilot answers client questions by reading documents and database
records. **Ragas is an independent grader** — a second AI that reads each
answer and scores it the way a strict examiner would: *is this true? did it
actually answer the question? did it use the right source material?*

Why we run it: our own tests check mechanics ("did the system reply?"),
but only a grader that understands language can check *quality* ("was the
reply correct and well-supported?"). Ragas gives us that quality check —
automatically, on every release.

The scores are 0 to 1, like exam percentages. **1.0 = perfect. Anything
below ~0.8 deserves a look.**

---

## 1. What Ragas measures — the five report cards

Every question we send through the system gets graded on five dimensions.
Think of them as five different red pens:

| Metric | Plain-English question it answers | Example of a failure |
|---|---|---|
| **faithfulness** | "Did the AI make anything up?" | Answer says the wire fee is $15 when the document says $12.73 |
| **answer_relevancy** | "Did it actually answer what was asked?" | User asks "what's the fee?" and gets a paragraph about office hours |
| **context_precision** | "Did it grab the *right* documents?" | It pulled 5 documents but only 1 was relevant — noise drowns signal |
| **context_recall** | "Did it find *all* the documents needed?" | The answer needed both the SLA *and* the fee schedule; it only found the SLA |
| **answer_correctness** | "Is the answer close to the known right answer?" | Compares to a reference answer we wrote by hand |

Two groups worth noticing:

- **Answer quality** — `faithfulness`, `answer_relevancy`, `answer_correctness`:
  is the *output* good?
- **Search quality** — `context_precision`, `context_recall`: is the
  *retrieval engine* fetching the right material? A bad answer with good
  retrieval points at a writing problem; a bad answer with bad retrieval
  points at a search problem. This split is what makes the scores actionable.

## 2. How the grading pipeline works

For each test question, four things happen:

```
Question ──► our copilot answers ──► answer + the documents it used
                                          │
                                          ▼
                                   Ragas judge (an AI examiner)
                                          │
                                          ▼
                              scores + a pass/fail report
```

The judge receives, per question:

| What the judge sees | In plain terms |
|---|---|
| `user_input` | the question we asked |
| `response` | the copilot's actual answer |
| `retrieved_contexts` | every document excerpt the copilot read |
| `reference` | the correct answer *we* wrote — the answer key |

Three honest caveats:

1. **The judge is an AI too.** It can be wrong — treat scores as a strong
   signal, not a verdict. Section 7 shows how to check the underlying
   evidence when a score looks suspicious.
2. **The judge is deliberately a *different* AI than the answerer.**
   Answers are written by Claude Sonnet; the judge is Claude Opus
   (`judge.model` in `eval/ragas.yaml`). Why it matters: a model grading
   its own output is biased — it over-scores familiar phrasing and shares
   its own blind spots (an LLM can ratify the same hallucination it just
   wrote). If the judge model ever matches the answer model, the run logs
   an `eval.ragas_self_judge` warning — treat those scores as advisory.
   This is also why the deterministic checks below are the hard release
   gate: they can't be charmed by prose style.
3. **No API key, no scores.** The judge needs an Anthropic API key to run.
   Without one the whole Ragas block is skipped and the report shows only
   our mechanical checks (citations, access-control, refusals, key-point
   matching) — which still run for free.

## 3. Setup — what needs to be configured

### 3a. The API key (required to get scores)

The judge calls Anthropic's Claude model, which needs a key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The same key also switches the copilot from canned draft answers to real
AI-written answers — **so without a key, the judge ends up grading drafts,
not real prose.** Scores from a keyless run smoke-test the plumbing only;
don't treat them as quality numbers.

### 3b. `eval/ragas.yaml` — the control panel (new)

Everything else is in one small file, `eval/ragas.yaml`. No code changes
needed to retune the judge:

```yaml
enabled: true                    # master switch — same as --no-ragas

judge:
  model: ""                      # blank = use the app's configured model
  api_key_env: ANTHROPIC_API_KEY # which env var holds the key (not the key itself)

metrics:                         # comment a line out to skip that report card
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall
  - answer_correctness

run:                             # judge-call limits
  timeout_s: 180                 # give up on a single grading call after 3 min
  max_retries: 3                 # retry flaky calls
  max_workers: 8                 # grade up to 8 questions in parallel
  seed: 42                       # reproducibility

thresholds: {}                   # quality gate — see below
```

When might you edit it?

- **Skip a metric** — e.g. remove `context_recall` when your test set has
  no hand-written reference answers to compare against (it can't score
  "did we find everything" without an answer key).
- **Tighten the loop** — raise `max_retries`/`timeout_s` if judge calls
  time out on big suites; lower `max_workers` if you hit rate limits.
- **Enforce a quality floor** — add thresholds, then run with
  `--fail-on-thresholds` to turn it into a release gate:

  ```yaml
  thresholds:
    faithfulness: 0.80        # "never ship a run that makes things up >20% of the time"
    context_precision: 0.70
  ```

  Now `eval/runner.py --suite holdout --fail-on-thresholds` exits non-zero
  when any score dips below its floor — the same way `--fail-on-leak`
  already gates on data-access violations.

### 3c. Supporting services (for engineers)

```bash
export EMBEDDING_PROVIDER=local              # embedder for relevancy scoring
export DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb
export OPENSEARCH_URL=http://localhost:9200
export OPS_DATABASE_URL=postgresql://rag_readonly:readonly@localhost:5432/opsdb
export OPS_ADMIN_DATABASE_URL=postgresql://rag:rag@localhost:5432/opsdb
```

## 4. Adding a test case — the "exam question" format

Tests are plain-text files, one JSON object per line — think of each line
as an exam card with a question on the front and the grading rubric on
the back. Suites live in `eval/scenarios/`:

| File | Purpose | When it runs |
|---|---|---|
| `pr.jsonl` | tiny fast smoke set | every pull request |
| `dev.jsonl` | the working set | development |
| `holdout.jsonl` | the grading set | release gates — **never tuned to** |
| `golden_set.jsonl` | large generated set | broad coverage runs |

One card, annotated:

```json
{
  "id": "DEV-023",                                     // unique name
  "question": "What is the SEV-1 first-response target?",  // what's asked
  "category": "doc_factual",                           // groups results in reports
  "user_id": "u_agent1",                               // which test user asks it
  "expected_answer": "The target is 15 minutes.",      // the answer key
  "key_points": ["15"],                                // facts that MUST appear (scored!)
  "required_doc_ids": ["SLA-004"],                     // which docs SHOULD be found
  "required_tool_calls": [],                           // which DB tools SHOULD be used
  "must_refuse": false,                                // true = correct answer is "can't say"
  "must_not_contain": [],                              // strings that must NEVER appear
  "expected_answerability": "fully_answerable"         // expected verdict type
}
```

Field guidance:

- **`expected_answer` matters most for Ragas** — it's the answer key the
  judge compares against. Write it as a complete factual sentence.
- **`key_points` is the cheap correctness check** — substrings that must
  appear in the reply (fees, statuses, dates, names). Scored as
  `key_points_hit`: fraction present, `n/a` when the row has none or the
  answer refused. This is deterministic — it catches wrong answers even
  when the prose looks fluent.
- **`must_not_contain`** also covers *poisoned evidence*: the corpus
  contains one deliberately injected document (POL-001 §3) and injected
  ticket subjects (e.g. TKT-00106) — rows like `must_not_contain:
  ["PWNED"]` prove retrieved content can't hijack the answer.
- **`must_refuse: true`** creates a security test — e.g. a user asking
  about a client they aren't entitled to. The correct answer is a refusal.
- **`user_id` controls access** — pick a user from `db/seed/users.json`
  whose client list matches your scenario. To test a *denial*, pick one
  who is not entitled.
- **Verify against `data/synthetic/facts.json`** before committing — the
  #1 cause of false test failures is a test card that disagrees with the
  generated data.

## 5. Running the evaluation

```bash
# the standard run — full dev suite, Ragas on
python3 eval/runner.py --suite dev

# quick smoke test — 3 questions only
python3 eval/runner.py --suite pr --limit 3

# mechanical checks only — no judge calls, zero cost
python3 eval/runner.py --suite dev --no-ragas

# release gate — fail if any access-control leak
python3 eval/runner.py --suite holdout --fail-on-leak

# release gate — fail on ANY missed floor: Ragas thresholds (ragas.yaml)
# AND deterministic quality gates (eval/gates.yaml)
python3 eval/runner.py --suite holdout --fail-on-thresholds

# use different config files
python3 eval/runner.py --suite dev --ragas-config eval/ragas-experimental.yaml
python3 eval/runner.py --suite dev --gates eval/gates-strict.yaml

# baseline comparison — run the suite through every answer system and
# write a side-by-side delta report (eval/reports/compare_*.md)
python3 eval/runner.py --suite dev --system all --no-ragas

# or one system at a time, then merge any reports by hand
python3 eval/runner.py --suite dev --system plain --no-ragas
python3 eval/compare.py eval/reports/eval_*.json eval/reports/eval_plain_*.json
```

What you'll see: progress logs per question, then a path to the report,
e.g. `eval/reports/eval_20261113_155313.json`.

Prerequisites for a live run: Docker services up
(`sudo docker compose up -d postgres opensearch`), seed data loaded
(`copilot db init`), and the document index built (`copilot index`).

## 6. Reading the results

Each run writes two files to `eval/reports/`:

- `eval_<timestamp>.md` — human-readable, open this one
- `eval_<timestamp>.json` — the same data, machine-readable

A real report (22-question dev suite) looks like this — full image in
[appendix A.4](#appendix-a-what-the-running-system-looks-like):

![eval report](screenshots/eval_report.png)

A summary table looks like:

```json
{
  "summary": {
    "acl_clean": 1.0,
    "citation_precision": 0.53,
    "ragas_faithfulness": 0.87,
    "ragas_answer_relevancy": 0.82,
    "ragas_context_precision": 0.74
  },
  "threshold_checks": {
    "faithfulness": {"value": 0.87, "minimum": 0.80, "passed": true}
  }
}
```

How to read it:

- **`ragas_` prefix = judge-scored.** Absent entirely → the judge never
  ran (no API key). `ragas_error` present → the judge ran and failed;
  the message says why.
- **`threshold_checks` = Ragas gates** — only present when `thresholds:`
  is configured; `passed` is `true`, `false`, or `null` (score missing).
- **`gate_checks` = deterministic gates** — floors from `eval/gates.yaml`
  on the mechanical metrics (refusals, key points, tool selection, banned
  content). `--fail-on-thresholds` enforces both gate sets; `passed:
  null` (no data) never fails.
- **`n/a` means "not applicable", not zero.** A metric is `n/a` when it's
  undefined for a row — e.g. citation precision on a refusal (nothing was
  cited; undefined ≠ wrong), or `key_points_hit` on a row with no key
  points. `n/a` rows are excluded from averages — `*_n` shows how many
  rows each mean covers.
- **`key_points_hit` is the workhorse correctness check** — fraction of
  required facts (from the row's `key_points`) that appear in the reply.
- **Non-`ragas_` metrics are our own mechanical checks** — free, fast,
  deterministic. `acl_clean: 1.0` means no access-control violations
  (the zero-tolerance one).
- **By-category tables** break the same scores down by question type —
  e.g. "unauthorized" should always score perfectly; "conflicting_evidence"
  is where honest weaknesses show.

## 7. Reviewing the evidence — tracing a bad score

A low score is a question, not an answer. The report's `failures` array
lists every row that missed on any metric — question, reply, and which
scores were low. To go deeper:

- **The audit trail** — every test runs with a request id like
  `eval-DEV-004-a1b2c3`; query `audit_events` for it to replay the exact
  step-by-step decisions (which documents were fetched, what the tools
  returned, why the answer was shaped that way).
- **The review queue** — answers the system itself flagged as risky or
  low-confidence land in `review_queue` for human sign-off.
- **The `validation` field** — each answer carries a per-claim record of
  which statements were cited and supported. This is our *internal*
  faithfulness check; when it and Ragas disagree, look at the raw passages.
- **Re-run one question live** to watch the diagnostics:

  ```bash
  curl -X POST http://localhost:8090/api/v1/answer \
    -H "X-API-Key: dev-key-1" -H "X-User-Id: u_agent1" \
    -H "Content-Type: application/json" \
    -d '{"question": "<the failing question>"}'
  ```

## 8. Baseline comparison — is the agentic layer worth it?

Every claim in this guide invites a fair question: *wouldn't plain RAG —
retrieve some chunks, ask the model — do just as well?* The project ships
a baseline to answer that with evidence instead of opinion.

`src/app/baseline.py` is deliberately naive: **no routing, no tools, no
grading, no validation, no guardrails, no answerability** — it searches
every collection like one merged index, drafts once, and always claims
`fully_answerable`. Same corpus, same stores, same embedding model, same
draft machinery as the agentic pipeline — so any score difference is
attributable to the agent loop, not to a better retriever or model.
(Operational how-to — commands, flags, config, report anatomy:
`docs/BASELINE.md`.)

To keep the comparison honest it comes in two variants on each axis:

| variant | retrieval | ACL pre-filter | what it measures |
|---|---|---|---|
| `plain` | dense top-5 | **none** (leaks — like real naive RAG) | the realistic naive system |
| `plain-acl` | dense top-5 | yes | naive RAG + security |
| `plain-hybrid` | hybrid + rerank | none | same retrieval, no agent loop |
| `plain-hybrid-acl` | hybrid + rerank | yes | everything *but* the agent loop |

`plain-hybrid-acl` is the strictest control: identical retrieval and
enforcement — whatever `agentic` still beats it on is provably the agent
machinery (answerability verdicts, abstention, validation, conflict
surfacing, guardrails).

```bash
# run the suite through all 5 systems + write the comparison report
python3 eval/runner.py --suite dev --system all --no-ragas

# → eval/reports/compare_<ts>.md — side-by-side metric table,
#   deltas vs the reference (first) system, and per-category
#   breakdowns for the "gap" metrics (refusals, answerability,
#   key points, ACL, banned content)
```

**Where the delta shows — and where it shouldn't.** On easy factual
questions the baseline ties or nearly ties (retrieval recall is already
~1.0) — that's expected and honest; the agentic layer isn't magic on
easy asks. The gap lives in the categories the scenario suites now
deliberately weight: *conflicting* (baseline picks one doc and asserts
it confidently — the agentic system flags the disagreement with values),
*ambiguous*, *unanswerable* (baseline hallucinates from weak matches —
including leaking *other clients'* documents when ACL is off), and
*adversarial* (baseline echoes the planted `PWNED` instruction — no
output guardrail). Expired-agreement questions (everfield, acme,
ironvale — all `expiry_date: 2024-06-01`) are the cleanest demo: the
agentic system abstains; the baseline answers confidently from someone
else's contract.

## 9. FAQ (the questions managers actually ask)

**How much does it cost?** Judge calls are LLM calls — roughly one per
metric per question. A 22-question suite × 5 metrics ≈ 110 small calls.
`--no-ragas` runs are free.

**Can the judge be wrong?** Yes — it's an AI. Treat scores as a smoke
alarm, not a court verdict. Always spot-check a low score against the
`failures` evidence before acting on it.

**Why did scores drop after a release?** Check `version_snapshot` in the
report — it records the model, embedder, prompt hashes, dataset hash, and
git SHA. Comparing snapshots between two runs shows exactly what changed.

**What's a "good" score?** There's no universal bar — it depends on the
domain. Our convention: access-control and refusal metrics must be 1.0
(non-negotiable); Ragas scores ≥0.8 are healthy, 0.6–0.8 is a
conversation, <0.6 is a problem.

**Why is answerability/refusal accuracy not 1.0?** Those rows cover
deliberately hard cases — ambiguous questions, conflicting documents,
denied access. Some misses are expected while the system matures; the
trend matters more than the point value.

## 10. Glossary

| Term | Meaning |
|---|---|
| **Judge** | the AI model that grades answers (Claude, via Anthropic API) |
| **Context / passage** | a document excerpt the copilot retrieved |
| **Reference / ground truth** | the correct answer, written by a human |
| **Scenario / suite** | a test question / a collection of them |
| **Hallucination** | an answer that states something the sources don't support — what `faithfulness` catches |
| **ACL** | access-control list — who may see which client's data |
| **PII** | personally identifiable information — names, emails, phones |
| **Release gate** | an automatic pass/fail check that can block a release |

## 11. Troubleshooting

| Symptom | Likely cause |
|---|---|
| No `ragas_*` keys in report | `ANTHROPIC_API_KEY` unset — judge skipped by design |
| `ragas_error` in summary | judge call failed (key/network/API drift) — read the message |
| `ragas.yaml` rejected at startup | unknown metric name — the error lists valid ones |
| `context_recall` always low | test rows lack good `expected_answer` references — or drop the metric in `ragas.yaml` |
| All scores low on a no-key run | expected — without a key the copilot emits draft answers; set the key and rerun |
| Threshold gate fails on missing scores | the judge didn't run — check the API key before blaming quality |

---

## Appendix A — what the running system looks like

Real screenshots from a live run (all committed under `docs/screenshots/`).

**A.1 — Chat UI** (`streamlit run src/ui/app.py`, port 8501). Pick a test
user, type a question; citations and answerability appear under each reply.

![chat ui](screenshots/copilot_ui.png)

**A.2 — API surface** (`uvicorn app.main:app`, `/docs`). The versioned
`/api/v1` routes — answer, SSE stream, health, and the admin endpoints
(jobs, DLQ, documents, consistency).

![api docs](screenshots/api_docs.png)

**A.3 — One live answer.** The question, the reply, its versioned
citations, and the claim-level validation the *internal* checker produced
— this is the per-row evidence Ragas scores sit on top of. The raw JSON
behind this screenshot is committed at
[`samples/answer_example.json`](samples/answer_example.json) — reply,
citations, retrieved chunks, claim validation, budget, and diagnostics.

![live answer](screenshots/answer_example.png)

**A.4 — An eval report** (22-row dev suite). Summary table first, then
per-category breakdown, then the `failures` list — every row that missed
any metric with its reply text and which scores fell short.

![eval report](screenshots/eval_report.png)

**A.5 — The Streamlit eval dashboard.** The same report viewed in the UI
(`Page → Eval dashboard`, or deep-link `http://127.0.0.1:8501/?page=eval`;
a specific file via `?page=eval&report=<name>`). The picker defaults to the
newest `eval_*.json`; summary, per-category table, and failures table
render the raw report JSON from §6.

![eval dashboard](screenshots/eval_dashboard.png)
