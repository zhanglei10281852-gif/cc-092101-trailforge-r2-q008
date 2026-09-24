"""Read-only verification of per-object audit chains.

Verification never returns record content, digests of stored rows, or masked
fields: results only carry structural coordinates (chain, sequence, audit id,
time range) and the *kind* of the first break detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from trailforge.audit_chain.digest import (
    GENESIS_PREV_DIGEST,
    SUPPORTED_CHAIN_VERSIONS,
    compute_digest,
)
from trailforge.audit_chain.service import record_fields
from trailforge.models.audit import AuditChainHead, AuditLog

BreakKind = Literal[
    "unsealed_record",
    "gap",
    "prev_digest_mismatch",
    "record_digest_mismatch",
    "unsupported_version",
    "head_mismatch",
    "head_orphan",
]

# Stable ordering when several breaks share a chain: structural checks first.
_KIND_ORDER: dict[str, int] = {
    "unsealed_record": 0,
    "gap": 1,
    "unsupported_version": 2,
    "prev_digest_mismatch": 3,
    "record_digest_mismatch": 4,
    "head_mismatch": 5,
    "head_orphan": 6,
}


@dataclass(frozen=True)
class ChainBreak:
    kind: BreakKind
    chain_key: str
    entity_type: str
    entity_id: int
    expected_seq: int | None = None
    actual_seq: int | None = None
    audit_id: int | None = None
    occurred_at: datetime | None = None
    head_seq: int | None = None


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    chains_checked: int
    total_chains: int
    records_checked: int
    checked_from: datetime | None
    checked_to: datetime | None
    first_break: ChainBreak | None = None
    breaks: list[ChainBreak] = field(default_factory=list)


def _chain_scope(
    session: Session,
    *,
    entity_type: str | None,
    entity_id: int | None,
    occurred_from: datetime | None,
    occurred_to: datetime | None,
) -> list[tuple[str, str, int]]:
    """Return (chain_key, entity_type, entity_id) triples to verify.

    Scoping is decided by which chains have *any* record inside the time
    window; each selected chain is then walked from seq 1 so that a node
    before the window that was deleted is still caught.
    """
    statement = (
        select(AuditLog.chain_key, AuditLog.entity_type, AuditLog.entity_id)
        .where(AuditLog.chain_key.is_not(None))
        .distinct()
    )
    if entity_type is not None:
        statement = statement.where(AuditLog.entity_type == entity_type)
    if entity_id is not None:
        statement = statement.where(AuditLog.entity_id == entity_id)
    if occurred_from is not None:
        statement = statement.where(AuditLog.occurred_at >= occurred_from)
    if occurred_to is not None:
        statement = statement.where(AuditLog.occurred_at <= occurred_to)
    statement = statement.order_by(AuditLog.chain_key)
    return [(key, etype, eid) for key, etype, eid in session.execute(statement).all()]


def verify_chains(
    session: Session,
    *,
    entity_type: str | None = None,
    entity_id: int | None = None,
    occurred_from: datetime | None = None,
    occurred_to: datetime | None = None,
    limit_chains: int | None = None,
    offset_chains: int = 0,
) -> ChainVerification:
    """Verify audit chains, optionally scoped to an object and time range.

    Chains are iterated in deterministic key order and can be paged with
    ``offset_chains`` / ``limit_chains``. A "chain unit" is either a chain
    that has sealed records inside the scope or a chain head whose records
    were all deleted, so paging never repeats or hides breaks. Every break
    found in the page is returned, ordered so the first break leads.
    """
    if (entity_id is not None) and entity_type is None:
        raise ValueError("entity_id requires entity_type")

    scoped = _chain_scope(
        session,
        entity_type=entity_type,
        entity_id=entity_id,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
    )
    scoped_keys = {key for key, _, _ in scoped}

    head_rows: dict[str, AuditChainHead] = {
        row.chain_key: row for row in session.scalars(select(AuditChainHead)).all()
    }

    # A head is orphaned only when the chain has no records at all, regardless
    # of the time window: a chain whose records merely fall outside the window
    # is out of scope, not broken.
    all_chain_keys = {
        key
        for (key,) in session.execute(
            select(AuditLog.chain_key).where(AuditLog.chain_key.is_not(None)).distinct()
        ).all()
    }
    orphan_units: list[tuple[str, str, int]] = []
    for key in sorted(head_rows):
        if key in all_chain_keys:
            continue
        etype, _, raw = key.rpartition(":")
        eid = int(raw)
        if entity_type is not None and etype != entity_type:
            continue
        if entity_id is not None and eid != entity_id:
            continue
        orphan_units.append((key, etype, eid))

    units = sorted(scoped + orphan_units, key=lambda item: item[0])
    total_chains = len(units)
    page = units[offset_chains : offset_chains + limit_chains] if limit_chains else units

    breaks: list[ChainBreak] = []
    records_checked = 0

    # Unsealed rows are a global (cross-chain) break; surface them only on the
    # first page so paging does not repeat the same finding.
    if offset_chains == 0:
        unsealed_filters = [AuditLog.chain_key.is_(None)]
        if entity_type is not None:
            unsealed_filters.append(AuditLog.entity_type == entity_type)
        if entity_id is not None:
            unsealed_filters.append(AuditLog.entity_id == entity_id)
        if occurred_from is not None:
            unsealed_filters.append(AuditLog.occurred_at >= occurred_from)
        if occurred_to is not None:
            unsealed_filters.append(AuditLog.occurred_at <= occurred_to)
        first_unsealed = session.scalar(
            select(AuditLog)
            .where(*unsealed_filters)
            .order_by(AuditLog.entity_type, AuditLog.entity_id, AuditLog.occurred_at, AuditLog.id)
            .limit(1)
        )
        if first_unsealed is not None:
            breaks.append(
                ChainBreak(
                    kind="unsealed_record",
                    chain_key="",
                    entity_type=first_unsealed.entity_type,
                    entity_id=first_unsealed.entity_id,
                    audit_id=first_unsealed.id,
                    occurred_at=first_unsealed.occurred_at,
                )
            )

    for key, etype, eid in page:
        if key not in scoped_keys:
            # Orphan head: every record of the chain is missing.
            breaks.append(
                ChainBreak(
                    kind="head_orphan",
                    chain_key=key,
                    entity_type=etype,
                    entity_id=eid,
                    head_seq=head_rows[key].last_seq,
                )
            )
            continue

        rows = list(
            session.scalars(
                select(AuditLog)
                .where(AuditLog.chain_key == key)
                .order_by(AuditLog.chain_seq, AuditLog.id)
            )
        )
        records_checked += len(rows)
        head = head_rows.get(key)
        expected_digest = GENESIS_PREV_DIGEST
        expected_seq = 1
        for row in rows:
            if row.chain_seq != expected_seq:
                breaks.append(
                    ChainBreak(
                        kind="gap",
                        chain_key=key,
                        entity_type=etype,
                        entity_id=eid,
                        expected_seq=expected_seq,
                        actual_seq=row.chain_seq,
                        audit_id=row.id,
                        occurred_at=row.occurred_at,
                    )
                )
                # Continue from the actual position so later breaks surface too.
                expected_seq = row.chain_seq if row.chain_seq is not None else expected_seq
            if row.chain_version not in SUPPORTED_CHAIN_VERSIONS:
                breaks.append(
                    ChainBreak(
                        kind="unsupported_version",
                        chain_key=key,
                        entity_type=etype,
                        entity_id=eid,
                        actual_seq=row.chain_seq,
                        audit_id=row.id,
                        occurred_at=row.occurred_at,
                    )
                )
            else:
                if row.prev_digest != expected_digest:
                    breaks.append(
                        ChainBreak(
                            kind="prev_digest_mismatch",
                            chain_key=key,
                            entity_type=etype,
                            entity_id=eid,
                            expected_seq=row.chain_seq,
                            audit_id=row.id,
                            occurred_at=row.occurred_at,
                        )
                    )
                if row.record_digest is not None:
                    recomputed = compute_digest(
                        version=row.chain_version,
                        chain_key=key,
                        chain_seq=row.chain_seq,
                        prev_digest=row.prev_digest
                        if row.prev_digest is not None
                        else "",
                        fields=record_fields(row),
                    )
                    if recomputed != row.record_digest:
                        breaks.append(
                            ChainBreak(
                                kind="record_digest_mismatch",
                                chain_key=key,
                                entity_type=etype,
                                entity_id=eid,
                                actual_seq=row.chain_seq,
                                audit_id=row.id,
                                occurred_at=row.occurred_at,
                            )
                        )
            expected_digest = row.record_digest or ""
            expected_seq = (row.chain_seq or expected_seq - 1) + 1
        if head is None:
            breaks.append(
                ChainBreak(
                    kind="head_mismatch",
                    chain_key=key,
                    entity_type=etype,
                    entity_id=eid,
                    expected_seq=expected_seq - 1,
                    head_seq=0,
                )
            )
        else:
            last_seq = rows[-1].chain_seq if rows else 0
            last_digest = rows[-1].record_digest if rows else GENESIS_PREV_DIGEST
            if head.last_seq != last_seq or head.last_digest != last_digest:
                breaks.append(
                    ChainBreak(
                        kind="head_mismatch",
                        chain_key=key,
                        entity_type=etype,
                        entity_id=eid,
                        expected_seq=last_seq,
                        head_seq=head.last_seq,
                        occurred_at=rows[-1].occurred_at if rows else None,
                    )
                )

    breaks.sort(key=lambda item: (item.chain_key, item.expected_seq or 0, _KIND_ORDER[item.kind]))
    first_break = breaks[0] if breaks else None
    return ChainVerification(
        ok=not breaks,
        chains_checked=len(page),
        total_chains=total_chains,
        records_checked=records_checked,
        checked_from=occurred_from,
        checked_to=occurred_to,
        first_break=first_break,
        breaks=breaks,
    )
