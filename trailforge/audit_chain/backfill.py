"""One-shot, resumable backfill of chain seals for legacy audit rows."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trailforge.audit_chain.service import AuditChainService
from trailforge.database.session import Database
from trailforge.models.audit import AuditLog

DEFAULT_CHAINS_PER_BATCH = 100


def count_unsealed(session: Session) -> int:
    return int(
        session.scalar(select(func.count()).select_from(AuditLog).where(AuditLog.chain_key.is_(None)))
        or 0
    )


def seal_batch(session: Session, *, max_chains: int) -> int:
    """Seal every unsealed record of up to ``max_chains`` chains.

    Chains are processed in ascending (entity_type, entity_id) order and each
    chain's records in (occurred_at, id) order, so the backfill order is
    deterministic. The whole batch is one transaction: an interruption rolls
    the unfinished batch back and a later run rediscovers the chains as
    unsealed and resumes safely.
    """
    entity_rows = session.execute(
        select(AuditLog.entity_type, AuditLog.entity_id)
        .where(AuditLog.chain_key.is_(None))
        .distinct()
        .order_by(AuditLog.entity_type, AuditLog.entity_id)
        .limit(max_chains)
    ).all()
    sealed = 0
    service = AuditChainService(session)
    for entity_type, entity_id in entity_rows:
        rows = list(
            session.scalars(
                select(AuditLog)
                .where(
                    AuditLog.chain_key.is_(None),
                    AuditLog.entity_type == entity_type,
                    AuditLog.entity_id == entity_id,
                )
                .order_by(AuditLog.occurred_at, AuditLog.id)
            )
        )
        for row in rows:
            service.seal_existing(row)
            sealed += 1
    return sealed


def backfill_audit_chains(
    database: Database,
    *,
    chains_per_batch: int = DEFAULT_CHAINS_PER_BATCH,
    max_batches: int | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Seal all legacy audit rows, one deterministic batch per transaction.

    Safe to interrupt at any point: committed batches stay sealed and the
    next run resumes from the remaining unsealed rows. Runs entirely against
    the local database; no external service is involved.
    """
    batches = 0
    sealed_total = 0
    while True:
        if max_batches is not None and batches >= max_batches:
            break
        sealed = database.run_write(
            lambda session: seal_batch(session, max_chains=chains_per_batch)
        )
        batches += 1
        sealed_total += sealed
        if on_batch is not None:
            on_batch(batches, sealed)
        if sealed == 0:
            break
    with database.session() as session:
        remaining = count_unsealed(session)
    return {"batches": batches, "sealed": sealed_total, "remaining": remaining}
