-- Ops schema: synthetic trade / account / ticket systems of record.
CREATE TABLE IF NOT EXISTS accounts (
    client_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    region          TEXT NOT NULL,
    tier            TEXT NOT NULL,
    owner_team      TEXT NOT NULL,
    agreement_doc   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id    TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL REFERENCES accounts(client_id),
    type        TEXT NOT NULL,
    status      TEXT NOT NULL,
    amount      NUMERIC(14,2) NOT NULL,
    currency    TEXT NOT NULL DEFAULT 'USD',
    trade_date  DATE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_client ON trades(client_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL REFERENCES accounts(client_id),
    category    TEXT NOT NULL,
    status      TEXT NOT NULL,
    subject     TEXT NOT NULL,
    resolution  TEXT,
    created_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_client ON tickets(client_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
