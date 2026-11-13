"""Store protocols — one injectable interface per operational system.

Each interface is implemented once by its Postgres-backed class (accounts.py,
trades.py, tickets.py); ``core.dependencies`` lazily returns the real store
or a test fake. Keeping the boundary typed lets tests substitute fakes
without a database.

Author: venkateshwara doijode
"""

from typing import Protocol

from app.security.authn import Principal
from app.tools.ops.schemas import (
    AccountRow,
    AccountsQuery,
    OpsResult,
    StoreHealth,
    TicketRow,
    TicketsQuery,
    TradeRow,
    TradesQuery,
)


class AccountsStore(Protocol):
    """Read interface for the accounts system."""

    def get(
        self, query: AccountsQuery, principal: Principal, conn=None
    ) -> OpsResult[AccountRow]: ...

    def healthcheck(self) -> StoreHealth: ...


class TradesStore(Protocol):
    """Read interface for the trades system."""

    def get(self, query: TradesQuery, principal: Principal, conn=None) -> OpsResult[TradeRow]: ...

    def healthcheck(self) -> StoreHealth: ...


class TicketsStore(Protocol):
    """Read interface for the tickets system."""

    def get(self, query: TicketsQuery, principal: Principal, conn=None) -> OpsResult[TicketRow]: ...

    def healthcheck(self) -> StoreHealth: ...
