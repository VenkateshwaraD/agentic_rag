"""SQLAlchemy models for the rag database (ragdb)."""

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(32), nullable=False, index=True)
    source = Column(String(32), nullable=False, index=True)
    client_id = Column(String(64), index=True, default="")
    date = Column(String(16), default="")
    owner = Column(String(128), default="")
    doc_version = Column(String(16), default="1.0")
    section = Column(String(256), default="")
    entitlements = Column(JSON, nullable=False, default=dict)
    text = Column(Text, nullable=False)
    # embedding column added via migration / index DDL (pgvector)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    trace_id = Column(String(64), default="")
    user_id = Column(String(64), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    event_type = Column(String(32), nullable=False, index=True)
    payload = Column(JSON, nullable=False, default=dict)
    latency_ms = Column(Float, default=0.0)


class UserEntitlement(Base):
    __tablename__ = "entitlements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, unique=True, index=True)
    role = Column(String(32), nullable=False)
    client_ids = Column(JSON, nullable=False, default=list)
