"""Tamper-evident sealing of audit logs into per-object hash chains.

Every audit log row is sealed by exactly one :class:`AuditChainLink` in the
same database transaction as the business write, so a rolled-back
transaction never leaves an orphan link. Links of one business object
(entity_type, entity_id) form a single continuous chain: ``sequence`` grows
by one and each link stores the predecessor's ``entry_hash`` in
``previous_hash``. Digests are computed over a stable canonical
serialization of the *stored* audit row, whose sensitive fields were
already redacted by ``ServiceBase._sanitize`` — sealing and verification
therefore never touch values behind the redaction boundary.

The algorithm carries an explicit version identifier (``algorithm`` column)
so a future upgrade can introduce ``sha256-chain-v2`` while still
recognizing v1 links. Everything runs locally against SQLite; no external
service is involved in writing, migrating, or verifying the chain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trailforge.models.audit import AuditChainLink, AuditLog

if TYPE_CHECKING:
    from trailforge.database.session import Database

CHAIN_ALGORITHM_V1 = "sha256-chain-v1"
SUPPORTED_ALGORITHMS = frozenset({CHAIN_ALGORITHM_V1})
GENESIS_PREVIOUS_HASH = "0" * 64

BREAK_UNSEALED_LOG = "unsealed_log"
BREAK_ORPHAN_LINK = "orphan_link"
BREAK_UNKNOWN_ALGORITHM = "unknown_algorithm"
BREAK_PAYLOAD_MISMATCH = "payload_mismatch"
BREAK_ENTRY_MISMATCH = "entry_mismatch"
BREAK_PREVIOUS_HASH_MISMATCH = "previous_hash_mismatch"
BREAK_SEQUENCE_GAP = "sequence_gap"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_EMPTY = "empty"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def payload_document(log: AuditLog) -> dict[str, Any]:
    """Stable field serialization of one audit record (algorithm v1).

    ``before_state``/``after_state``/``context`` are the already-sanitized
    stored JSON values, so redacted secrets stay behind the boundary.
    """
    action = log.action.value if isinstance(log.action, Enum) else str(log.action)
    return {
        "algorithm": CHAIN_ALGORITHM_V1,
        "audit_log_id": log.id,
        "entity_type": log.entity_type,
        "entity_id": log.entity_id,
        "action": action,
        "actor_id": log.actor_id,
        "occurred_at": _iso_z(log.occurred_at),
        "before_state": log.before_state,
        "after_state": log.after_state,
        "context": log.context,
        "correlation_id": log.correlation_id,
    }


def compute_payload_hash(log: AuditLog) -> str:
    return _sha256(payload_document(log))


def compute_entry_hash(
    *,
    algorithm: str,
    entity_type: str,
    entity_id: int,
    sequence: int,
    occurred_at: datetime,
    payload_hash: str,
    previous_hash: str,
) -> str:
    return _sha256(
        {
            "algorithm": algorithm,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "sequence": sequence,
            "occurred_at": _iso_z(occurred_at),
            "payload_hash": payload_hash,
            "previous_hash": previous_hash,
        }
    )


class AuditChainSealer:
    """Appends chain links for audit logs within the caller's transaction."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._heads: dict[tuple[str, int], tuple[int, str]] = {}

    def seal(self, log: AuditLog) -> AuditChainLink:
        key = (log.entity_type, log.entity_id)
        if key not in self._heads:
            self._heads[key] = self._load_head(*key)
        head_sequence, head_hash = self._heads[key]
        sequence = head_sequence + 1
        payload_hash = compute_payload_hash(log)
        entry_hash = compute_entry_hash(
            algorithm=CHAIN_ALGORITHM_V1,
            entity_type=log.entity_type,
            entity_id=log.entity_id,
            sequence=sequence,
            occurred_at=log.occurred_at,
            payload_hash=payload_hash,
            previous_hash=head_hash,
        )
        link = AuditChainLink(
            audit_log_id=log.id,
            entity_type=log.entity_type,
            entity_id=log.entity_id,
            sequence=sequence,
            algorithm=CHAIN_ALGORITHM_V1,
            occurred_at=log.occurred_at,
            payload_hash=payload_hash,
            previous_hash=head_hash,
            entry_hash=entry_hash,
        )
        self.session.add(link)
        self.session.flush()
        self._heads[key] = (sequence, entry_hash)
        return link

    def _load_head(self, entity_type: str, entity_id: int) -> tuple[int, str]:
        head = self.session.scalar(
            select(AuditChainLink)
            .where(
                AuditChainLink.entity_type == entity_type,
                AuditChainLink.entity_id == entity_id,
            )
            .order_by(AuditChainLink.sequence.desc())
            .limit(1)
        )
        if head is None:
            return (0, GENESIS_PREVIOUS_HASH)
        return (head.sequence, head.entry_hash)


def unsealed_logs_statement(entity_type: str | None = None, entity_id: int | None = None):
    statement = (
        select(AuditLog)
        .outerjoin(AuditChainLink, AuditChainLink.audit_log_id == AuditLog.id)
        .where(AuditChainLink.id.is_(None))
    )
    if entity_type is not None:
        statement = statement.where(AuditLog.entity_type == entity_type)
    if entity_id is not None:
        statement = statement.where(AuditLog.entity_id == entity_id)
    return statement


def seal_next_batch(session: Session, *, batch_size: int) -> int:
    """Seals up to ``batch_size`` unsealed logs in deterministic order.

    Returns the number of logs sealed. The caller owns the transaction, so a
    failure rolls the whole batch back and a later run can safely resume.
    """
    logs = session.scalars(
        unsealed_logs_statement()
        .order_by(AuditLog.occurred_at.asc(), AuditLog.id.asc())
        .limit(batch_size)
    ).all()
    sealer = AuditChainSealer(session)
    for log in logs:
        sealer.seal(log)
    return len(logs)


def backfill_audit_chain(database: Database, *, batch_size: int = 500) -> dict[str, int]:
    """One-shot backfill sealing all existing audit logs, resumable after
    interruption: each batch commits separately and already-sealed logs are
    skipped on the next run."""
    sealed = 0
    batches = 0
    while True:
        with database.session() as session:
            count = seal_next_batch(session, batch_size=batch_size)
        if count == 0:
            break
        sealed += count
        batches += 1
    return {"sealed": sealed, "batches": batches}


@dataclass(frozen=True)
class ChainBreak:
    """First detected break. Carries digests only — never record content."""

    reason: str
    link_id: int | None = None
    audit_log_id: int | None = None
    sequence: int | None = None
    expected_hash: str | None = None
    actual_hash: str | None = None


@dataclass(frozen=True)
class ChainVerification:
    entity_type: str
    entity_id: int
    status: str
    checked_links: int
    total_links: int
    unsealed_logs: int
    algorithms: tuple[str, ...] = ()
    page: int = 1
    page_size: int = 1
    pages: int = 0
    first_break: ChainBreak | None = None


class AuditChainVerifier:
    """Recomputes and compares chain links for one business object."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def verify(
        self,
        *,
        entity_type: str,
        entity_id: int,
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ChainVerification:
        filters = [
            AuditChainLink.entity_type == entity_type,
            AuditChainLink.entity_id == entity_id,
        ]
        if occurred_after is not None:
            filters.append(AuditChainLink.occurred_at >= occurred_after)
        if occurred_before is not None:
            filters.append(AuditChainLink.occurred_at <= occurred_before)
        scoped = select(AuditChainLink).where(*filters)
        total = int(
            self.session.scalar(
                select(func.count()).select_from(scoped.subquery())
            )
            or 0
        )
        pages = (total + page_size - 1) // page_size if total else 0
        links = list(
            self.session.scalars(
                scoped.order_by(AuditChainLink.sequence.asc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        first_break = self._first_break(entity_type, entity_id, links)
        unsealed = self._unsealed(entity_type, entity_id, occurred_after, occurred_before)
        if first_break is None and unsealed:
            earliest = unsealed[0]
            first_break = ChainBreak(reason=BREAK_UNSEALED_LOG, audit_log_id=earliest.id)
        if first_break is not None:
            status = STATUS_FAILED
        elif total == 0 and not unsealed:
            status = STATUS_EMPTY
        else:
            status = STATUS_OK
        return ChainVerification(
            entity_type=entity_type,
            entity_id=entity_id,
            status=status,
            checked_links=len(links),
            total_links=total,
            unsealed_logs=len(unsealed),
            algorithms=tuple(sorted({link.algorithm for link in links})),
            page=page,
            page_size=page_size,
            pages=pages,
            first_break=first_break,
        )

    def _first_break(
        self, entity_type: str, entity_id: int, links: list[AuditChainLink]
    ) -> ChainBreak | None:
        if not links:
            return None
        logs = {
            log.id: log
            for log in self.session.scalars(
                select(AuditLog).where(AuditLog.id.in_([link.audit_log_id for link in links]))
            )
        }
        predecessors = self._predecessors(entity_type, entity_id, links)
        for link in links:
            log = logs.get(link.audit_log_id)
            if log is None:
                return ChainBreak(
                    reason=BREAK_ORPHAN_LINK,
                    link_id=link.id,
                    audit_log_id=link.audit_log_id,
                    sequence=link.sequence,
                )
            if link.algorithm not in SUPPORTED_ALGORITHMS:
                return ChainBreak(
                    reason=BREAK_UNKNOWN_ALGORITHM,
                    link_id=link.id,
                    audit_log_id=link.audit_log_id,
                    sequence=link.sequence,
                )
            payload_hash = compute_payload_hash(log)
            if payload_hash != link.payload_hash:
                return ChainBreak(
                    reason=BREAK_PAYLOAD_MISMATCH,
                    link_id=link.id,
                    audit_log_id=link.audit_log_id,
                    sequence=link.sequence,
                    expected_hash=link.payload_hash,
                    actual_hash=payload_hash,
                )
            entry_hash = compute_entry_hash(
                algorithm=link.algorithm,
                entity_type=link.entity_type,
                entity_id=link.entity_id,
                sequence=link.sequence,
                occurred_at=link.occurred_at,
                payload_hash=link.payload_hash,
                previous_hash=link.previous_hash,
            )
            if entry_hash != link.entry_hash:
                return ChainBreak(
                    reason=BREAK_ENTRY_MISMATCH,
                    link_id=link.id,
                    audit_log_id=link.audit_log_id,
                    sequence=link.sequence,
                    expected_hash=link.entry_hash,
                    actual_hash=entry_hash,
                )
            if link.sequence == 1:
                if link.previous_hash != GENESIS_PREVIOUS_HASH:
                    return ChainBreak(
                        reason=BREAK_PREVIOUS_HASH_MISMATCH,
                        link_id=link.id,
                        audit_log_id=link.audit_log_id,
                        sequence=link.sequence,
                        expected_hash=GENESIS_PREVIOUS_HASH,
                        actual_hash=link.previous_hash,
                    )
                continue
            predecessor = predecessors.get(link.sequence - 1)
            if predecessor is None:
                return ChainBreak(
                    reason=BREAK_SEQUENCE_GAP,
                    link_id=link.id,
                    audit_log_id=link.audit_log_id,
                    sequence=link.sequence,
                )
            if link.previous_hash != predecessor.entry_hash:
                return ChainBreak(
                    reason=BREAK_PREVIOUS_HASH_MISMATCH,
                    link_id=link.id,
                    audit_log_id=link.audit_log_id,
                    sequence=link.sequence,
                    expected_hash=predecessor.entry_hash,
                    actual_hash=link.previous_hash,
                )
        return None

    def _predecessors(
        self, entity_type: str, entity_id: int, links: list[AuditChainLink]
    ) -> dict[int, AuditChainLink]:
        wanted = {link.sequence - 1 for link in links if link.sequence > 1}
        if not wanted:
            return {}
        rows = self.session.scalars(
            select(AuditChainLink).where(
                AuditChainLink.entity_type == entity_type,
                AuditChainLink.entity_id == entity_id,
                AuditChainLink.sequence.in_(wanted),
            )
        )
        return {row.sequence: row for row in rows}

    def _unsealed(
        self,
        entity_type: str,
        entity_id: int,
        occurred_after: datetime | None,
        occurred_before: datetime | None,
    ) -> list[AuditLog]:
        statement = unsealed_logs_statement(entity_type, entity_id)
        if occurred_after is not None:
            statement = statement.where(AuditLog.occurred_at >= occurred_after)
        if occurred_before is not None:
            statement = statement.where(AuditLog.occurred_at <= occurred_before)
        statement = statement.order_by(AuditLog.occurred_at.asc(), AuditLog.id.asc())
        return list(self.session.scalars(statement))
