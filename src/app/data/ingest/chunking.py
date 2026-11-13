"""Chunk documents with the canonical metadata schema.

Metadata per chunk: {doc_id, chunk_id, source, doc_type, date,
effective_date, expiry_date, owner, entitlements, doc_version, section,
page_number, content_hash, parent_id}
entitlements = {"clients": [client_id, ...] | "*", "roles": [role, ...]}

Chunking is metadata-aware: the ``doc_type`` field (agreement | sla | faq |
table | ops) selects the splitter — agreements/SLAs split on sections with
tables kept atomic, FAQs pair Q/A blocks, CSV/table docs group rows, and
ops docs use the section default. Long sections emit parent+child chunks
when ``parent_child_enabled`` so retrieval can expand to full context.

Author: venkateshwara doijode
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.constants import ROLE_ADMIN, ROLE_AGENT, ROLE_SENIOR_AGENT

ALL_ROLES = [ROLE_AGENT, ROLE_SENIOR_AGENT, ROLE_ADMIN]

_META_RE = re.compile(r"<!--\s*doc_id:\s*([A-Z0-9-]+)\s*(.*?)\s*-->", re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
_SECTION_RE = re.compile(r"^(#{1,3})\s+(§?\S*.*)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_FAQ_Q_RE = re.compile(r"^(?:Q\d*[:.]|\*\*Q[:.]?)", re.MULTILINE)
_PARENT_MIN_CHARS = 1200  # sections longer than this split into child chunks


@dataclass
class Chunk:
    """One indexed unit of a document: a section of text plus its metadata.

    Carries provenance (doc_id, chunk_id, source, doc_type, dates, owner,
    version, section, page, client), the entitlement record used for ACL
    filtering at query time, dedup/versioning keys (content_hash,
    parent_id), and the embedding-model metadata recorded at index time.
    """

    doc_id: str
    source: str
    date: str
    owner: str
    entitlements: dict = field(default_factory=dict)
    doc_version: str = "1.0"
    section: str = ""
    client_id: str = ""
    text: str = ""
    chunk_id: str = ""
    doc_type: str = "generic"
    page_number: int | None = None
    effective_date: str = ""
    expiry_date: str = ""
    content_hash: str = ""
    parent_id: str = ""
    embed_model: str = ""
    embed_version: str = ""
    embed_dim: int = 0


def parse_metadata_comment(text: str) -> dict[str, str]:
    """Extract the ``<!-- doc_id: X | key: value ... -->`` header into a dict.

    Returns an empty dict when the document has no metadata comment.
    """
    m = _META_RE.search(text)
    if not m:
        return {}
    inner = m.group(0).removeprefix("<!--").removesuffix("-->")
    meta: dict[str, str] = {}
    for part in inner.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def content_hash(text: str) -> str:
    """sha256 over whitespace-normalized text — dedup/idempotency key."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def _slug(section: str) -> str:
    """Slugify a section title for use inside chunk ids."""
    return re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")[:48] or "body"


def _split_sections(text: str) -> list[tuple[str, str, int | None]]:
    """Split markdown on headers; returns [(section_title, section_text, page)].

    HTML comments (metadata blocks) are stripped first; ``<!-- page: N -->``
    markers emitted by the PDF loader map each section to its last-seen
    page number.
    """
    pages: dict[int, int] = {}
    for m in _PAGE_RE.finditer(text):
        pages[m.start()] = int(m.group(1))
    text = _COMMENT_RE.sub("", text)
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("body", text, _page_at(0, pages))]
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            (m.group(2).strip(), text[m.end() : end].strip(), _page_at(m.start(), pages))
        )
    return sections


def _page_at(offset: int, pages: dict[int, int]) -> int | None:
    """Return the last page marker seen before ``offset`` (PDF provenance)."""
    candidates = [p for pos, p in pages.items() if pos <= offset]
    return candidates[-1] if candidates else None


def _split_table_rows(body: str) -> list[str]:
    """Group markdown/CSV-flattened table rows into row-group chunks.

    Header lines stay with the first group; each subsequent group carries
    up to ~15 rows so wide tables don't produce giant single chunks.
    """
    rows = [ln for ln in body.split("\n") if ln.strip()]
    groups, current = [], []
    for ln in rows:
        current.append(ln)
        if len(current) >= 15:
            groups.append("\n".join(current))
            current = []
    if current:
        groups.append("\n".join(current))
    return groups


def _split_paragraphs(body: str) -> list[str]:
    """Split a long section body into paragraph children (parent-child mode)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return paras if len(paras) > 1 else [body]


def _build_entitlements(client_id: str, source: str, roles: str = "") -> dict:
    """Client-scoped docs (agreements) restrict to the owning client;
    bank-wide docs (SLAs, service teams) are visible to all entitled clients.

    ``roles`` is an optional comma-separated allowlist from document
    metadata (``roles: senior_agent,admin`` in the header comment) — valid
    role names only; empty means the default ALL_ROLES visibility.
    """
    clients = [client_id] if client_id else "*"
    parsed = [r.strip() for r in roles.split(",") if r.strip() in ALL_ROLES]
    return {"clients": clients, "roles": parsed or ALL_ROLES}


def _infer_doc_type(meta: dict, path: Path, source: str) -> str:
    """Determine the chunk doc_type: explicit metadata first, else inference."""
    if meta.get("doc_type"):
        return meta["doc_type"].strip().lower()
    if path.suffix.lower() == ".csv":
        return "table"
    stem = path.stem.lower()
    if "faq" in stem:
        return "faq"
    return {"agreements": "agreement", "slas": "sla", "service_teams": "ops"}.get(source, "generic")


def chunk_document(text: str, path: Path, source: str, parent_child: bool = False) -> list[Chunk]:
    """Split one document's text into ``Chunk`` records with full metadata.

    Metadata comes from the HTML comment header (falling back to the file
    stem for doc_id); each section becomes a chunk with entitlements from
    the doc's client scope. ``doc_type`` selects the splitter; long
    agreement/SLA sections emit parent + paragraph children when
    ``parent_child`` is set.
    """
    meta = parse_metadata_comment(text)
    doc_id = meta.get("doc_id", path.stem)
    client_id = meta.get("client_id", "")
    date = meta.get("date", "") or meta.get("effective_date", "")
    owner = meta.get("owner", client_id or "bank")
    version = meta.get("version", "1.0")
    doc_type = _infer_doc_type(meta, path, source)
    effective = meta.get("effective_date", date)
    expiry = meta.get("expiry_date", "")
    ents = _build_entitlements(client_id, source, meta.get("roles", ""))

    chunks: list[Chunk] = []
    for i, (section, body, page) in enumerate(_split_sections(text)):
        if not body.strip():
            continue

        # tables: keep markdown tables atomic by splitting on row groups
        bodies = _split_table_rows(body) if doc_type == "table" else [body]

        for j, part in enumerate(bodies):
            base: dict[str, Any] = dict(
                doc_id=doc_id,
                source=source,
                date=date,
                owner=owner,
                entitlements=ents,
                doc_version=version,
                section=section if len(bodies) == 1 else f"{section} [rows {j + 1}]",
                client_id=client_id,
                doc_type=doc_type,
                page_number=page,
                effective_date=effective,
                expiry_date=expiry,
            )
            if parent_child and doc_type in ("agreement", "sla") and len(part) > _PARENT_MIN_CHARS:
                # chunk ids embed doc_version so keep_all versioning never
                # collides and 'latest' deletes cleanly by doc_id
                parent_id = f"{doc_id}:v{version}:{_slug(section)}:p{i}"
                chunks.append(
                    Chunk(
                        **base,
                        chunk_id=parent_id,
                        text=part,
                        content_hash=content_hash(part),
                    )
                )
                for k, child in enumerate(_split_paragraphs(part)):
                    if child == part:
                        continue  # single-paragraph section — parent suffices
                    chunks.append(
                        Chunk(
                            **base,
                            chunk_id=f"{parent_id}:c{k}",
                            parent_id=parent_id,
                            text=child,
                            content_hash=content_hash(child),
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        **base,
                        chunk_id=f"{doc_id}:v{version}:{_slug(base['section'])}:{i}",
                        text=part,
                        content_hash=content_hash(part),
                    )
                )
    return chunks


def chunk_directory(
    root: Path, source: str, loaders_iter, parent_child: bool = False
) -> list[Chunk]:
    """Load and chunk every supported document under ``root`` for a collection."""
    from app.data.ingest.loaders import load_document

    chunks: list[Chunk] = []
    for path in loaders_iter(root):
        chunks.extend(chunk_document(load_document(path), path, source, parent_child))
    return chunks
