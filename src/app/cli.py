"""copilot CLI — corpus / db / index / ask / evalset / eval subcommands.

Author: venkateshwara doijode
"""

import csv
import uuid
from pathlib import Path

import psycopg
import typer
from sqlalchemy import create_engine, text

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging import configure_logging, get_logger

app = typer.Typer(help="Agentic RAG client-service copilot CLI")
corpus_app = typer.Typer()
db_app = typer.Typer()
eval_app = typer.Typer()
evalset_app = typer.Typer()
docs_app = typer.Typer()
jobs_app = typer.Typer()
app.add_typer(corpus_app, name="corpus")
app.add_typer(db_app, name="db")
app.add_typer(eval_app, name="eval")
app.add_typer(evalset_app, name="evalset")
app.add_typer(docs_app, name="docs")
app.add_typer(jobs_app, name="jobs")

log = get_logger(__name__)


@corpus_app.command("generate")
def corpus_generate():
    """Generate the deterministic synthetic corpus + ops seed + facts.

    Writes agreement/SLA/team markdown, ops CSVs, facts.json, and users.json,
    then echoes a summary count of what was produced.
    """
    configure_logging()
    from app.data.corpus.generate import generate

    facts = generate()
    typer.echo(
        f"corpus written: {len(facts['agreements'])} agreements, "
        f"{len(facts['slas'])} SLAs, {len(facts['tickets'])} tickets"
    )


@db_app.command("init")
def db_init():
    """Create ops tables + readonly role, seed ops CSVs, create audit table.

    Applies the seed SQL scripts against the ops admin connection, loads the
    generated CSVs into the ops tables (fails if the corpus wasn't generated
    first), and ensures the audit table exists in the rag database.
    """
    configure_logging()
    s = get_settings()
    seed = PROJECT_ROOT / "db" / "seed"

    with psycopg.connect(s.ops_admin_database_url, autocommit=True) as conn:
        conn.execute((seed / "init_ops.sql").read_text())
        conn.execute((seed / "readonly_role.sql").read_text())
    log.info("db.ops_schema_ready")

    engine = create_engine(s.ops_admin_database_url)
    for table in ("accounts", "trades", "tickets"):
        csv_path = s.synthetic_dir / "ops" / f"{table}.csv"
        if not csv_path.exists():
            typer.echo(f"missing {csv_path} — run `copilot corpus generate` first")
            raise typer.Exit(1)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        stmt = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})")
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {table} CASCADE"))
            conn.execute(stmt, rows)
        log.info("db.seeded", table=table, rows=len(rows))

    from app.core.dependencies import get_audit

    get_audit().ensure_table()
    typer.echo("db init complete")


@app.command("index")
def index(rebuild: bool = typer.Option(False, "--rebuild")):
    """Chunk → redact → embed → write both stores.

    Pass ``--rebuild`` to wipe pgvector and the OpenSearch indexes first.
    """
    configure_logging()
    from app.retrieval.indexing import run_index

    counts = run_index(rebuild=rebuild)
    typer.echo(f"indexed: {counts}")


@app.command("ask")
def ask(question: str, user: str = typer.Option("u_agent1", "--user")):
    """Ask the copilot a question as a given user (entitlement demo).

    Looks up the fixture user's API key, authenticates, runs the agent graph,
    and prints the reply, citations, tool calls, refusal flag, and trace id.
    """
    configure_logging()
    import json as _json

    from app.agent.graph import answer
    from app.security.authn import authenticate

    users = _json.loads(Path(get_settings().users_file).read_text())
    api_key = next((u["api_key"] for u in users if u["user_id"] == user), None)
    principal = authenticate(api_key, user)
    out = answer(question, principal, request_id=uuid.uuid4().hex[:16])
    typer.echo("\n=== REPLY ===")
    typer.echo(out["reply"])
    typer.echo(f"\ncitations: {out['citations']}")
    typer.echo(f"tool_calls: {out['tool_calls']}")
    typer.echo(f"refusal: {out['refusal']}  trace: {out['trace_id']}")


@evalset_app.command("generate")
def evalset_generate(output: Path | None = None):
    """Bootstrap golden-set questions from facts.json (review before golden).

    Writes a JSONL eval set to ``--output`` or ``eval/golden_set.jsonl``.
    """
    configure_logging()
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from generate_evalset import generate as gen  # noqa: E402

    path = output or (get_settings().eval_dir / "golden_set.jsonl")
    n = gen(path)
    typer.echo(f"wrote {n} questions → {path}")


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option("golden", "--suite"),
    limit: int = typer.Option(0, "--limit"),
    repeat: int = typer.Option(1, "--repeat"),
    golden: Path | None = None,
    fail_on_leak: bool = typer.Option(False, "--fail-on-leak"),
):
    """Run an eval suite and write a report.

    ``--suite`` picks pr|dev|holdout|golden; ``--golden`` selects an
    explicit JSONL file; ``--repeat`` reruns for variance; ``--fail-on-leak``
    exits non-zero on any ACL leakage (release gate).
    """
    configure_logging()
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from runner import run_eval  # noqa: E402

    report = run_eval(suite, golden, limit=limit or None, repeat=repeat, fail_on_leak=fail_on_leak)
    typer.echo(f"report → {report}")


# --- document lifecycle ----------------------------------------------------


@docs_app.command("list")
def docs_list(collection: str = typer.Option("agreements", "--collection")):
    """List indexed doc_ids + chunk counts for a collection (pgvector side)."""
    configure_logging()
    from app.core.dependencies import get_vectorstore

    docs = get_vectorstore().list_documents(collection)
    for d in docs:
        typer.echo(f"{d['doc_id']}  chunks={d['chunks']}  version={d.get('doc_version', '')}")
    typer.echo(f"{len(docs)} documents")


@docs_app.command("delete")
def docs_delete(doc_id: str):
    """Delete a document's chunks from pgvector, OpenSearch, and the registry."""
    configure_logging()
    from app.retrieval.indexing import delete_document

    out = delete_document(doc_id)
    typer.echo(f"deleted {doc_id}: pg={out['pg_deleted']} os={out['os_deleted']}")


@docs_app.command("verify")
def docs_verify():
    """Report pgvector↔OpenSearch↔registry chunk parity per collection."""
    configure_logging()
    import json as _json

    from app.retrieval.indexing import verify_consistency

    typer.echo(_json.dumps(verify_consistency(), indent=2))


@docs_app.command("index-info")
def docs_index_info():
    """Show the embedding-model registry (name/version/dim) per index."""
    configure_logging()
    import json as _json

    from app.retrieval.indexing import index_info

    typer.echo(_json.dumps(index_info(), indent=2, default=str))


# --- ingestion jobs --------------------------------------------------------


@jobs_app.command("list")
def jobs_list(limit: int = typer.Option(20, "--limit")):
    """List recent ingestion jobs with status/progress/metrics."""
    configure_logging()
    import json as _json

    from app.data.ingest import jobs

    typer.echo(_json.dumps(jobs.list_jobs(limit), indent=2, default=str))


@jobs_app.command("dlq")
def jobs_dlq(job_id: str | None = typer.Option(None, "--job"), limit: int = 100):
    """List dead-lettered documents (optionally scoped to one job)."""
    configure_logging()
    import json as _json

    from app.data.ingest import jobs

    typer.echo(_json.dumps(jobs.dead_letters(job_id=job_id, limit=limit), indent=2, default=str))
