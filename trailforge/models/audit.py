from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from trailforge.database.base import Base, UTCDateTime, utc_now
from trailforge.domain.enums import AuditAction
from trailforge.models.mixins import IntegerPrimaryKeyMixin


class AuditLog(IntegerPrimaryKeyMixin, Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_entity", "entity_type", "entity_id"),
        Index("ix_audit_actor_time", "actor_id", "occurred_at"),
        Index("uq_audit_chain_position", "chain_key", "chain_seq", unique=True),
        Index(
            "ix_audit_unsealed",
            "id",
            sqlite_where=text("chain_key IS NULL"),
        ),
    )

    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[AuditAction] = mapped_column(String(40), nullable=False)
    before_state: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    after_state: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    context: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(120), index=True)
    # Tamper-evident chain seal. Nullable so legacy rows can exist until the
    # backfill migration seals them; every new write is sealed atomically.
    chain_key: Mapped[str | None] = mapped_column(String(200))
    chain_seq: Mapped[int | None] = mapped_column(Integer)
    chain_version: Mapped[str | None] = mapped_column(String(20))
    prev_digest: Mapped[str | None] = mapped_column(String(64))
    record_digest: Mapped[str | None] = mapped_column(String(64), index=True)


class AuditChainHead(Base):
    """Per-object chain tip. Acts as the serialization point for concurrent
    writers: appending to a chain requires updating its head row, and SQLite
    serializes writers on that row's write lock, so each chain grows as one
    unique, gapless sequence. Updated in the same transaction as the audit
    log row, so a rolled-back business transaction leaves no orphan node."""

    __tablename__ = "audit_chain_heads"

    chain_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    last_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    last_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, onupdate=utc_now, nullable=False
    )


class IdempotencyRecord(IntegerPrimaryKeyMixin, Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("scope", "idempotency_key", name="uq_idempotency_scope_key"),
    )

    scope: Mapped[str] = mapped_column(String(100), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(160), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[int] = mapped_column(Integer, nullable=False)
    response_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class SchemaMigration(IntegerPrimaryKeyMixin, Base):
    __tablename__ = "schema_migrations"

    version: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
