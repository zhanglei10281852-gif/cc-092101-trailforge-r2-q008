"""Sealing for per-object tamper-evident audit chains.

Sealing happens in the same database transaction as the business write, so a
rolled-back transaction leaves no orphan chain node. Concurrency on one chain
is serialized through the chain head row: appending requires updating that
row, and SQLite's single-writer lock makes concurrent appenders proceed one
at a time, producing one unique, gapless sequence per object.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from trailforge.audit_chain.digest import (
    GENESIS_PREV_DIGEST,
    V1,
    chain_key,
    compute_digest,
)
from trailforge.models.audit import AuditChainHead, AuditLog


def record_fields(log: AuditLog) -> dict[str, object]:
    """Stable fields sealed into the digest, taken from the stored row.

    The state payloads are already sanitized at the masking boundary before
    they reach the row, so the digest never sees unmasked content.
    """
    return {
        "actor_id": log.actor_id,
        "occurred_at": log.occurred_at,
        "entity_type": log.entity_type,
        "entity_id": log.entity_id,
        "action": log.action,
        "before_state": log.before_state,
        "after_state": log.after_state,
        "context": log.context,
        "correlation_id": log.correlation_id,
    }


class AuditChainService:
    """Appends sealed audit records inside the caller's transaction."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def seal_new(self, log: AuditLog) -> AuditLog:
        """Seal a freshly inserted audit log row as the next chain node."""
        return self._seal(log)

    def seal_existing(self, log: AuditLog) -> AuditLog:
        """Seal a legacy row during backfill. Idempotent per row."""
        if log.record_digest is not None:
            return log
        return self._seal(log)

    def _seal(self, log: AuditLog) -> AuditLog:
        key = chain_key(log.entity_type, log.entity_id)
        head = self.lock_head(key)
        log.chain_key = key
        log.chain_seq = head.last_seq + 1
        log.chain_version = V1
        log.prev_digest = head.last_digest
        log.record_digest = compute_digest(
            version=V1,
            chain_key=key,
            chain_seq=log.chain_seq,
            prev_digest=log.prev_digest,
            fields=record_fields(log),
        )
        head.last_seq = log.chain_seq
        head.last_digest = log.record_digest
        self.session.flush()
        return log

    def lock_head(self, key: str) -> AuditChainHead:
        """Insert the head row if absent, then return it.

        The insert-or-ignore plus the later head update make the head row the
        chain's serialization point: concurrent writers on the same chain
        queue on SQLite's writer lock and then read the committed tip, so
        sequence numbers stay unique and continuous.
        """
        self.session.execute(
            sqlite_insert(AuditChainHead)
            .values(chain_key=key, last_seq=0, last_digest=GENESIS_PREV_DIGEST)
            .on_conflict_do_nothing(index_elements=["chain_key"])
        )
        head = self.session.scalar(
            select(AuditChainHead).where(AuditChainHead.chain_key == key)
        )
        if head is None:  # pragma: no cover - defensive, insert above guarantees the row
            raise RuntimeError(f"audit chain head {key} could not be initialized")
        return head
