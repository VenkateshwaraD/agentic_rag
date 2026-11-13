"""Strict Pydantic schemas for ops-tool inputs and outputs.

Every tool takes a typed query model (validated limits, enums, pagination)
and returns ``OpsResult[T]`` — typed rows plus freshness timestamps,
pagination state, latency metrics, and a structured error slot instead of
raw dicts or raised exceptions.

Author: venkateshwara doijode
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

OPS_DOMAINS = ("accounts", "trades", "tickets")

TicketStatus = Literal["open", "in_progress", "resolved", "closed"]
TicketCategory = Literal[
    "fee_dispute", "missing_statement", "failed_wire", "cutoff_inquiry", "access_request"
]
TradeStatus = Literal["settled", "pending", "failed", "returned"]
TradeType = Literal["wire", "ach", "check", "book_transfer"]


class _Page(BaseModel):
    """Shared pagination params — limit is hard-capped at the row cap."""

    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class AccountsQuery(_Page):
    """Input for the accounts store: optional point lookup by client_id."""

    client_id: str | None = None


class TradesQuery(_Page):
    """Input for the trades store: point lookup or filtered scan."""

    client_id: str | None = None
    trade_id: str | None = None
    status: TradeStatus | None = None
    trade_type: TradeType | None = None
    start_date: str | None = None
    end_date: str | None = None


class TicketsQuery(_Page):
    """Input for the tickets store: point lookup or filtered scan."""

    client_id: str | None = None
    ticket_id: str | None = None
    status: TicketStatus | None = None
    category: TicketCategory | None = None
    keyword: str | None = Field(default=None, max_length=100)


class AccountRow(BaseModel):
    """One row of the accounts table."""

    client_id: str
    name: str
    region: str
    tier: str
    owner_team: str
    agreement_doc: str


class TradeRow(BaseModel):
    """One row of the trades table."""

    trade_id: str
    client_id: str
    type: str
    status: str
    amount: float
    currency: str
    trade_date: str


class TicketRow(BaseModel):
    """One row of the tickets table."""

    ticket_id: str
    client_id: str
    category: str
    status: str
    subject: str
    resolution: str
    created_at: str


OpsErrorCode = Literal[
    "store_unavailable",
    "no_results",
    "partial_results",
    "unauthorized",
    "invalid_input",
]


class OpsError(BaseModel):
    """Structured error carried in ``OpsResult.error`` (never raised)."""

    code: OpsErrorCode
    message: str
    detail: str = ""


class StoreHealth(BaseModel):
    """Result of a store ``healthcheck()`` — surfaced in /ready and ops-health."""

    store: str
    ok: bool
    latency_ms: float = 0.0
    error: str = ""


class OpsResult[T](BaseModel):
    """Typed result envelope for every ops-tool query."""

    rows: list[T] = Field(default_factory=list)
    has_more: bool = False
    fresh_at: datetime | None = None  # when this query executed
    source_updated_at: str | None = None  # max(source timestamp) in rows
    store: str = ""
    snapshot_ts: datetime | None = None  # shared snapshot clock, multi-store reads
    latency_ms: float = 0.0
    error: OpsError | None = None
