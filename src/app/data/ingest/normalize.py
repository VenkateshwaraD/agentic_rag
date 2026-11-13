"""Text normalization stage — deterministic cleanup before chunking.

Strips BOM, normalizes line endings, and trims trailing whitespace so
content hashes and chunk boundaries are stable across platforms and
loader quirks. Idempotent: ``normalize(normalize(t)) == normalize(t)``.

Author: venkateshwara doijode
"""


def normalize_text(text: str) -> str:
    """Return ``text`` with BOM removed, ``\\r\\n``→``\\n``, rstrip'd lines."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(ln.rstrip() for ln in text.split("\n"))
