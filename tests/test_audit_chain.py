from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from trailforge.audit_chain.backfill import backfill_audit_chains, count_unsealed
from trailforge.audit_chain.digest import GENESIS_PREV_DIGEST, V1, chain_key
from trailforge.audit_chain.verify import verify_chains
from trailforge.config import Settings
from trailforge.database.migrations import initialize_database
from trailforge.database.session import Database
from trailforge.domain.enums import AuditAction
from trailforge.models.audit import AuditChainHead, AuditLog
from trailforge.services.base import ServiceBase

BASE_TIME = datetime(2026, 1, 1, 8, 0, 0, tzinfo=UTC)


def write_audit(
    session: Session,
    *,
    entity_type: str = "expedition",
    entity_id: int = 1,
    action: AuditAction = AuditAction.STATUS_CHANGED,
    after: dict | None = None,
    context: dict | None = None,
) -> AuditLog:
    return ServiceBase(session).audit(
        actor_id=None,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        after=after if after is not None else {"status": "cancelled"},
        context=context,
    )


def insert_legacy_row(
    session: Session,
    *,
    entity_type: str,
    entity_id: int,
    occurred_at: datetime,
    note: str = "legacy",
) -> None:
    """Insert a pre-chain audit row directly, bypassing the sealing service."""
    session.execute(
        text(
            "INSERT INTO audit_logs "
            "(actor_id, occurred_at, entity_type, entity_id, action, "
            " before_state, after_state, context, correlation_id) "
            "VALUES (:actor, :occurred, :etype, :eid, :action, "
            " :before, :after, :context, :correlation)"
        ),
        {
            "actor": None,
            "occurred": occurred_at.isoformat().replace("+00:00", "Z"),
            "etype": entity_type,
            "eid": entity_id,
            "action": "status_changed",
            "before": json.dumps({}),
            "after": json.dumps({"note": note}),
            "context": json.dumps({}),
            "correlation": None,
        },
    )


def chain_rows(session: Session, entity_type: str, entity_id: int) -> list[AuditLog]:
    return list(
        session.scalars(
            select(AuditLog)
            .where(
                AuditLog.entity_type == entity_type,
                AuditLog.entity_id == entity_id,
            )
            .order_by(AuditLog.chain_seq)
        )
    )


# ---------------------------------------------------------------------------
# Sealing on the normal write path
# ---------------------------------------------------------------------------


def test_new_audit_records_are_sealed_with_version(database: Database) -> None:
    with database.session() as session:
        write_audit(session, entity_id=10)
        write_audit(session, entity_id=10)
        write_audit(session, entity_id=20)

    with database.session() as session:
        rows = chain_rows(session, "expedition", 10)
        assert [row.chain_seq for row in rows] == [1, 2]
        assert all(row.chain_version == V1 for row in rows)
        assert rows[0].prev_digest == GENESIS_PREV_DIGEST
        assert rows[1].prev_digest == rows[0].record_digest
        assert rows[0].chain_key == chain_key("expedition", 10)
        head = session.get(AuditChainHead, chain_key("expedition", 10))
        assert head is not None
        assert head.last_seq == 2
        assert head.last_digest == rows[1].record_digest
        other = session.get(AuditChainHead, chain_key("expedition", 20))
        assert other is not None and other.last_seq == 1


def test_chain_is_per_business_object(database: Database) -> None:
    with database.session() as session:
        for _ in range(3):
            write_audit(session, entity_type="expedition", entity_id=1)
        for _ in range(2):
            write_audit(session, entity_type="gear_item", entity_id=1)
        write_audit(session, entity_type="expedition", entity_id=2)

    with database.session() as session:
        assert [r.chain_seq for r in chain_rows(session, "expedition", 1)] == [1, 2, 3]
        assert [r.chain_seq for r in chain_rows(session, "gear_item", 1)] == [1, 2]
        assert [r.chain_seq for r in chain_rows(session, "expedition", 2)] == [1]
        result = verify_chains(session)
        assert result.ok, result.first_break
        assert result.total_chains == 3


def test_sensitive_fields_stay_masked_and_still_verify(database: Database) -> None:
    with database.session() as session:
        write_audit(
            session,
            entity_id=5,
            after={"status": "cancelled", "password": "s3cret-value", "token": "abc"},
        )
    with database.session() as session:
        row = chain_rows(session, "expedition", 5)[0]
        assert row.after_state["password"] == "[REDACTED]"
        assert row.after_state["token"] == "[REDACTED]"
        assert verify_chains(session).ok


# ---------------------------------------------------------------------------
# Rollback: no orphan chain nodes
# ---------------------------------------------------------------------------


def test_rolled_back_transaction_leaves_no_chain_node(database: Database) -> None:
    with pytest.raises(RuntimeError), database.session() as session:
        write_audit(session, entity_id=30)
        write_audit(session, entity_id=30)
        raise RuntimeError("business transaction failed")

    with database.session() as session:
        assert chain_rows(session, "expedition", 30) == []
        assert session.get(AuditChainHead, chain_key("expedition", 30)) is None
        assert verify_chains(session).ok

    # The next write after the rollback starts the chain cleanly at seq 1.
    with database.session() as session:
        write_audit(session, entity_id=30)
    with database.session() as session:
        rows = chain_rows(session, "expedition", 30)
        assert [row.chain_seq for row in rows] == [1]
        assert verify_chains(session).ok


def test_partial_batch_rollback_keeps_chain_continuous(database: Database) -> None:
    with database.session() as session:
        write_audit(session, entity_id=31)
    with pytest.raises(RuntimeError), database.session() as session:
        write_audit(session, entity_id=31)
        write_audit(session, entity_id=31)
        raise RuntimeError("second leg of the batch failed")
    with database.session() as session:
        write_audit(session, entity_id=31)

    with database.session() as session:
        rows = chain_rows(session, "expedition", 31)
        assert [row.chain_seq for row in rows] == [1, 2]
        assert verify_chains(session).ok


# ---------------------------------------------------------------------------
# Concurrency: one unique, continuous chain per object
# ---------------------------------------------------------------------------


def test_concurrent_writes_form_single_continuous_chain(database: Database) -> None:
    def write(index: int) -> int:
        def operation(session: Session) -> int:
            log = write_audit(session, entity_id=40, after={"seq_marker": index})
            return log.id

        return database.run_write(operation)

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(write, range(24)))

    assert len(set(ids)) == 24
    with database.session() as session:
        rows = chain_rows(session, "expedition", 40)
        assert len(rows) == 24
        assert sorted(row.chain_seq for row in rows) == list(range(1, 25))
        digests = {row.record_digest for row in rows}
        assert len(digests) == 24
        result = verify_chains(session)
        assert result.ok, result.first_break


def test_concurrent_writes_across_objects(database: Database) -> None:
    def write(index: int) -> None:
        entity_id = (index % 4) + 50
        database.run_write(lambda session: write_audit(session, entity_id=entity_id).id)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(write, range(40)))

    with database.session() as session:
        for entity_id in range(50, 54):
            seqs = sorted(row.chain_seq for row in chain_rows(session, "expedition", entity_id))
            assert seqs == list(range(1, 11))
        assert verify_chains(session).ok


# ---------------------------------------------------------------------------
# Tamper detection against the real database file
# ---------------------------------------------------------------------------


def sealed_chain(database: Database, entity_id: int = 60, length: int = 5) -> None:
    with database.session() as session:
        for index in range(length):
            write_audit(session, entity_id=entity_id, after={"step": index})


def test_verify_passes_on_untouched_chain(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        result = verify_chains(session)
        assert result.ok
        assert result.records_checked == 5
        assert result.first_break is None


def test_tampered_record_content_is_detected(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        session.execute(
            text("UPDATE audit_logs SET after_state = :forged WHERE chain_seq = 3"),
            {"forged": json.dumps({"step": "forged"})},
        )
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        assert result.first_break is not None
        assert result.first_break.kind == "record_digest_mismatch"
        assert result.first_break.actual_seq == 3


def test_deleted_middle_record_is_detected(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        session.execute(text("DELETE FROM audit_logs WHERE chain_seq = 3"))
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        assert result.first_break is not None
        assert result.first_break.kind == "gap"
        assert result.first_break.expected_seq == 3
        kinds = {item.kind for item in result.breaks}
        assert "prev_digest_mismatch" in kinds


def test_truncated_chain_tip_is_detected(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        session.execute(text("DELETE FROM audit_logs WHERE chain_seq >= 4"))
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        assert result.first_break is not None
        assert result.first_break.kind == "head_mismatch"
        assert result.first_break.head_seq == 5


def test_rewritten_prev_digest_is_detected(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        session.execute(
            text("UPDATE audit_logs SET prev_digest = 'x' || substr(prev_digest, 2) "
                 "WHERE chain_seq = 2")
        )
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        kinds = {item.kind for item in result.breaks}
        assert "prev_digest_mismatch" in kinds


def test_resequenced_records_are_detected(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        # Swap the stored positions of two nodes without touching digests.
        session.execute(text("UPDATE audit_logs SET chain_seq = 99 WHERE chain_seq = 2"))
        session.execute(text("UPDATE audit_logs SET chain_seq = 2 WHERE chain_seq = 3"))
        session.execute(text("UPDATE audit_logs SET chain_seq = 3 WHERE chain_seq = 99"))
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        kinds = {item.kind for item in result.breaks}
        assert kinds & {"prev_digest_mismatch", "record_digest_mismatch"}


def test_deleted_head_is_detected(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        session.execute(text("DELETE FROM audit_chain_heads"))
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        assert result.first_break is not None
        assert result.first_break.kind == "head_mismatch"


def test_fully_deleted_chain_is_detected(database: Database) -> None:
    sealed_chain(database, entity_id=61)
    sealed_chain(database, entity_id=62)
    with database.session() as session:
        session.execute(text("DELETE FROM audit_logs WHERE entity_id = 61"))
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        orphan = [b for b in result.breaks if b.kind == "head_orphan"]
        assert orphan and orphan[0].entity_id == 61
        # The surviving chain still verifies on its own.
        scoped = verify_chains(session, entity_type="expedition", entity_id=62)
        assert scoped.ok


def test_unknown_chain_version_is_reported(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        session.execute(
            text("UPDATE audit_logs SET chain_version = 'v99' WHERE chain_seq = 1")
        )
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        kinds = {item.kind for item in result.breaks}
        assert "unsupported_version" in kinds


def test_unsealed_row_is_reported(database: Database) -> None:
    sealed_chain(database)
    with database.session() as session:
        insert_legacy_row(
            session,
            entity_type="expedition",
            entity_id=60,
            occurred_at=BASE_TIME,
        )
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        assert result.first_break is not None
        assert result.first_break.kind == "unsealed_record"


def test_verification_response_never_contains_masked_content(database: Database) -> None:
    with database.session() as session:
        write_audit(
            session,
            entity_id=70,
            after={"status": "cancelled", "password": "top-secret-value"},
            context={"authorization": "Bearer raw-token"},
        )
    with database.session() as session:
        session.execute(
            text("UPDATE audit_logs SET after_state = :forged"),
            {"forged": json.dumps({"status": "completed"})},
        )
    with database.session() as session:
        result = verify_chains(session)
        assert not result.ok
        for item in result.breaks:
            payload = json.dumps(item.__dict__, default=str)
            assert "top-secret-value" not in payload
            assert "raw-token" not in payload
            assert "[REDACTED]" not in payload
            assert "password" not in payload
            assert "authorization" not in payload


# ---------------------------------------------------------------------------
# Object and time-range scoping
# ---------------------------------------------------------------------------


def test_verify_scoped_by_object(database: Database) -> None:
    sealed_chain(database, entity_id=80, length=2)
    sealed_chain(database, entity_id=81, length=2)
    with database.session() as session:
        session.execute(
            text("UPDATE audit_logs SET after_state = '{}' WHERE entity_id = 81 AND chain_seq = 1")
        )
    with database.session() as session:
        clean = verify_chains(session, entity_type="expedition", entity_id=80)
        assert clean.ok
        broken = verify_chains(session, entity_type="expedition", entity_id=81)
        assert not broken.ok
        assert broken.first_break is not None
        assert broken.first_break.entity_id == 81


def test_verify_scoped_by_time_range(database: Database) -> None:
    with database.session() as session:
        insert_legacy_row(
            session, entity_type="expedition", entity_id=90, occurred_at=BASE_TIME
        )
        insert_legacy_row(
            session,
            entity_type="expedition",
            entity_id=91,
            occurred_at=BASE_TIME + timedelta(days=10),
        )
    backfill_audit_chains(database)
    with database.session() as session:
        window = verify_chains(
            session,
            occurred_from=BASE_TIME + timedelta(days=5),
            occurred_to=BASE_TIME + timedelta(days=15),
        )
        assert window.ok
        assert window.total_chains == 1
        everything = verify_chains(session)
        assert everything.total_chains == 2


def test_verify_requires_entity_type_with_entity_id(database: Database) -> None:
    with database.session() as session, pytest.raises(ValueError, match="entity_type"):
        verify_chains(session, entity_id=1)


# ---------------------------------------------------------------------------
# Paged verification
# ---------------------------------------------------------------------------


def test_paged_verification_covers_every_chain_once(database: Database) -> None:
    for entity_id in range(100, 107):
        sealed_chain(database, entity_id=entity_id, length=2)
    with database.session() as session:
        session.execute(
            text("UPDATE audit_logs SET after_state = '{}' WHERE entity_id = 105 AND chain_seq = 1")
        )

    seen: set[str] = set()
    breaks = []
    offset = 0
    with database.session() as session:
        while True:
            page = verify_chains(session, limit_chains=3, offset_chains=offset)
            if page.chains_checked == 0:
                break
            page_keys = {item.chain_key for item in page.breaks}
            assert seen.isdisjoint(page_keys)
            seen.update(page_keys)
            offset += page.chains_checked
            breaks.extend(page.breaks)
            if offset >= page.total_chains:
                break
        assert offset == 7
        assert len(breaks) >= 1
        assert any(item.entity_id == 105 for item in breaks)

    # Paging a clean database yields no breaks and full coverage.
    with database.session() as session:
        session.execute(text("DELETE FROM audit_logs WHERE entity_id = 105"))
        session.execute(
            text("DELETE FROM audit_chain_heads WHERE chain_key = 'expedition:105'")
        )
    offset = 0
    checked = 0
    with database.session() as session:
        while True:
            page = verify_chains(session, limit_chains=2, offset_chains=offset)
            if page.chains_checked == 0:
                break
            assert page.ok, page.first_break
            checked += page.chains_checked
            offset += page.chains_checked
            if offset >= page.total_chains:
                break
        assert checked == 6


# ---------------------------------------------------------------------------
# Backfill of legacy rows
# ---------------------------------------------------------------------------


def test_backfill_seals_legacy_rows_in_deterministic_order(database: Database) -> None:
    with database.session() as session:
        # Insert out of chronological order on purpose; seq must follow
        # (occurred_at, id), not insertion order.
        insert_legacy_row(
            session,
            entity_type="expedition",
            entity_id=200,
            occurred_at=BASE_TIME + timedelta(hours=2),
            note="second",
        )
        insert_legacy_row(
            session, entity_type="expedition", entity_id=200, occurred_at=BASE_TIME, note="first"
        )
        insert_legacy_row(
            session, entity_type="gear_item", entity_id=7, occurred_at=BASE_TIME, note="gear"
        )
        assert count_unsealed(session) == 3

    result = backfill_audit_chains(database)
    assert result == {"batches": 2, "sealed": 3, "remaining": 0}

    with database.session() as session:
        rows = chain_rows(session, "expedition", 200)
        assert [row.chain_seq for row in rows] == [1, 2]
        assert rows[0].after_state == {"note": "first"}
        assert rows[1].after_state == {"note": "second"}
        assert rows[0].prev_digest == GENESIS_PREV_DIGEST
        assert rows[1].prev_digest == rows[0].record_digest
        assert verify_chains(session).ok


def test_backfill_resumes_after_interruption(database: Database) -> None:
    with database.session() as session:
        for entity_id in range(300, 305):
            insert_legacy_row(
                session,
                entity_type="expedition",
                entity_id=entity_id,
                occurred_at=BASE_TIME,
            )

    first = backfill_audit_chains(database, chains_per_batch=2, max_batches=1)
    assert first["sealed"] == 2
    assert first["remaining"] == 3

    # Simulate a restart: dispose the engine, then resume on a fresh one.
    database.engine.dispose()
    resumed = backfill_audit_chains(database, chains_per_batch=2)
    assert resumed["remaining"] == 0
    assert resumed["sealed"] == 3

    with database.session() as session:
        assert verify_chains(session).ok
        for entity_id in range(300, 305):
            rows = chain_rows(session, "expedition", entity_id)
            assert [row.chain_seq for row in rows] == [1]


def test_backfill_is_idempotent_when_nothing_pending(database: Database) -> None:
    sealed_chain(database, entity_id=310, length=2)
    result = backfill_audit_chains(database)
    assert result["sealed"] == 0
    assert result["remaining"] == 0
    with database.session() as session:
        assert verify_chains(session).ok


def test_backfill_extends_chain_that_already_has_sealed_tip(database: Database) -> None:
    # A chain that received new sealed writes while legacy rows were still
    # pending: backfill must not fork or renumber the existing chain.
    with database.session() as session:
        insert_legacy_row(
            session, entity_type="expedition", entity_id=320, occurred_at=BASE_TIME
        )
    with database.session() as session:
        write_audit(session, entity_id=320)
    result = backfill_audit_chains(database)
    assert result["sealed"] == 1
    with database.session() as session:
        rows = chain_rows(session, "expedition", 320)
        assert [row.chain_seq for row in rows] == [1, 2]
        assert verify_chains(session).ok


# ---------------------------------------------------------------------------
# Restart durability
# ---------------------------------------------------------------------------


def test_sealed_chain_survives_engine_restart(settings: Settings) -> None:
    first = Database(settings)
    initialize_database(first)
    with first.session() as session:
        write_audit(session, entity_id=400)
        write_audit(session, entity_id=400)
    first.engine.dispose()

    second = Database(settings)
    initialize_database(second)
    with second.session() as session:
        result = verify_chains(session)
        assert result.ok
        assert result.records_checked == 2
        # Writes continue the chain after restart.
        write_audit(session, entity_id=400)
    with second.session() as session:
        rows = chain_rows(session, "expedition", 400)
        assert [row.chain_seq for row in rows] == [1, 2, 3]
        assert verify_chains(session).ok
    second.engine.dispose()


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_verify_endpoint_reports_tamper_without_content(client) -> None:
    database = client.app.state.database
    sealed_chain(database, entity_id=500, length=3)
    with database.session() as session:
        session.execute(text("DELETE FROM audit_logs WHERE chain_seq = 2"))

    response = client.get("/api/v1/audit/chains/verify")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["first_break"]["kind"] == "gap"
    assert payload["first_break"]["expected_seq"] == 2
    body = response.text
    assert "after_state" not in body
    assert "record_digest" not in body

    scoped = client.get(
        "/api/v1/audit/chains/verify",
        params={"entity_type": "expedition", "entity_id": 500},
    )
    assert scoped.status_code == 200
    assert scoped.json()["ok"] is False

    missing = client.get(
        "/api/v1/audit/chains/verify",
        params={"entity_type": "expedition", "entity_id": 999},
    )
    assert missing.json()["ok"] is True


def test_verify_endpoint_rejects_naive_datetime(client) -> None:
    response = client.get(
        "/api/v1/audit/chains/verify",
        params={"occurred_from": "2026-01-01T00:00:00"},
    )
    assert response.status_code in {400, 422}


def test_backfill_endpoint_seals_legacy_rows(client) -> None:
    database = client.app.state.database
    with database.session() as session:
        insert_legacy_row(
            session, entity_type="expedition", entity_id=510, occurred_at=BASE_TIME
        )
    response = client.post("/api/v1/audit/chains/backfill")
    assert response.status_code == 200
    assert response.json() == {"batches": 2, "sealed": 1, "remaining": 0}
    verify = client.get("/api/v1/audit/chains/verify")
    assert verify.json()["ok"] is True
