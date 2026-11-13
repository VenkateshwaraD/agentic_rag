"""Chunk documents with the canonical metadata schema.

Metadata per chunk: {doc_id, source, date, owner, entitlements, doc_version, section}
entitlements = {"clients": [client_id, ...] | "*", "roles": [role, ...]}
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.constants import ROLE_ADMIN, ROLE_AGENT, ROLE_SENIOR_AGENT

ALL_ROLES = [ROLE_AGENT, ROLE_SENIOR_AGENT, ROLE_ADMIN]

_META_RE = re.compile(r"<!--\s*doc_id:\s*([A-Z0-9-]+)\s*(.*?)\s*-->", re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SECTION_RE = re.compile(r"^(#{1,3})\s+(§?\S*.*)$", re.MULTILINE)


@dataclass
class Chunk:
    doc_id: str
    source: str
    date: str
    owner: str
    entitlements: dict = field(default_factory=dict)
    doc_version: str = "1.0"
    section: str = ""
    client_id: str = ""
    text: str = ""


def parse_metadata_comment(text: str) -> dict[str, str]:
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


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on headers; returns [(section_title, section_text)].
    HTML comments (metadata blocks) are stripped first."""
    text = _COMMENT_RE.sub("", text)
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("body", text)]
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[m.end() : end].strip()))
    return sections


def _build_entitlements(client_id: str, source: str) -> dict:
    """Client-scoped docs (agreements) restrict to the owning client;
    bank-wide docs (SLAs, service teams) are visible to all entitled clients."""
    clients = [client_id] if client_id else "*"
    return {"clients": clients, "roles": ALL_ROLES}


def chunk_document(text: str, path: Path, source: str) -> list[Chunk]:
    meta = parse_metadata_comment(text)
    doc_id = meta.get("doc_id", path.stem)
    client_id = meta.get("client_id", "")
    date = meta.get("effective_date", meta.get("date", ""))
    owner = meta.get("owner", client_id or "bank")
    version = meta.get("version", "1.0")

    chunks = []
    for section, body in _split_sections(text):
        if not body.strip():
            continue
        chunks.append(
            Chunk(
                doc_id=doc_id,
                source=source,
                date=date,
                owner=owner,
                entitlements=_build_entitlements(client_id, source),
                doc_version=version,
                section=section,
                client_id=client_id,
                text=body,
            )
        )
    return chunks


def chunk_directory(root: Path, source: str, loaders_iter) -> list[Chunk]:
    from app.data.ingest.loaders import load_document

    chunks: list[Chunk] = []
    for path in loaders_iter(root):
        chunks.extend(chunk_document(load_document(path), path, source))
    return chunks
