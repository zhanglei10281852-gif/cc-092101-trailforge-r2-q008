"""Tests for per-object tamper-evident audit hash chains."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tests.conftest import create_expedition, create_route, create_user
from trailforge.config import Settings
from trailforge.database.base import utc_now
from trailforge.database.migrations import initialize_database
from trailforge.database.session import Database
from trailforge.domain.enums import AuditAction, EmergencyStatus, EmergencyType, RiskLevel
from trailforge.errors import InventoryError
from trailforge.models.audit import AuditChainLink, AuditLog
from trailforge.schemas.activities import ActivityStateChange
from trailforge.schemas.gear import (
    GearCatalogCreate,
    GearInventoryCreate,
    InventoryAdjustment,
)
from trailforge.schemas.safety import EmergencyIncidentCreate, EmergencyIncidentUpdate
from trailforge.services.activities import ExpeditionService
from trailforge.services.gear import GearService
from trailforge.services.safety import SafetyService
from trailforge.services.sealing import (
    CHAIN_ALGORITHM_V1,
    GENESIS_PREVIOUS_HASH,
    AuditChainSealer,
    AuditChainVerifier,
    backfill_audit_chain,
    compute_entry_hash,
    compute_payload_hash,
    seal_next_batch,
)


def _write_audit(
    session: Session,
    *,
    entity_type: str = "expedition",
    entity_id: int = 1,
    action: AuditAction = AuditAction.UPDATED,
    after: dict | None = None,
) -> AuditLog:
    """Insert a raw audit log without sealing (for backfill tests)."""
    log = AuditLog(
        actor_id=None,
        occurred_at=utc_now(),
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        before_state={},
        after_state=after or {"status": "open"},
        context={},
        correlation_id=None,
    )
    session.add(log)
    session.flush()
    return log


def _links(session: Session, entity_type: str = "expedition", entity_id: int = 1) -> list:
    return list(
        session.scalars(
            select(AuditChainLink)
            .where(
                AuditChainLink.entity_type == entity_type,
                AuditChainLink.entity_id == entity_id,
            )
            .order_by(AuditChainLink.sequence)
        )
    )


def _verify(session: Session, **kwargs):
    defaults = {"entity_type": "expedition", "entity_id": 1}
    defaults.update(kwargs)
    return AuditChainVerifier(session).verify(**defaults)


# ---------------------------------------------------------------------------
# Basic sealing invariants
# ---------------------------------------------------------------------------


def test_business_audit_write_creates_sealed_chain_link(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)

    links = _links(session, "expedition", expedition_id)
    assert len(links) == 1
    link = links[0]
    assert link.sequence == 1
    assert link.algorithm == CHAIN_ALGORITHM_V1
    assert link.previous_hash == GENESIS_PREVIOUS_HASH

    log = session.get(AuditLog, link.audit_log_id)
    assert log is not None
    assert link.payload_hash == compute_payload_hash(log)
    assert link.entry_hash == compute_entry_hash(
        algorithm=CHAIN_ALGORITHM_V1,
        entity_type="expedition",
        entity_id=expedition_id,
        sequence=1,
        occurred_at=log.occurred_at,
        payload_hash=link.payload_hash,
        previous_hash=GENESIS_PREVIOUS_HASH,
    )


def test_chain_verifies_ok_after_multiple_writes(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)
    ExpeditionService(session).change_status(
        expedition_id,
        ActivityStateChange(
            target_status="cancelled",
            actor_id=user_id,
            reason="weather window closed",
        ),
    )

    links = _links(session, "expedition", expedition_id)
    assert [link.sequence for link in links] == [1, 2]
    assert links[1].previous_hash == links[0].entry_hash
    report = _verify(session, entity_id=expedition_id)
    assert report.status == "ok"
    assert report.first_break is None
    assert report.total_links == 2
    assert report.algorithms == (CHAIN_ALGORITHM_V1,)


def test_sealing_covers_cancellation_inventory_and_incident_closures(
    session: Session,
) -> None:
    """The three scenarios from the audit brief each land on their own chain."""
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)

    safety = SafetyService(session)
    incident = safety.record_incident(
        EmergencyIncidentCreate(
            expedition_id=expedition_id,
            reported_by=user_id,
            incident_type=EmergencyType.WEATHER,
            risk_level=RiskLevel.HIGH,
            occurred_at=utc_now(),
            description="sudden thunderstorm near ridge",
            idempotency_key="incident-create-0001",
        )
    )
    safety.update_incident(
        incident.id,
        EmergencyIncidentUpdate(
            status=EmergencyStatus.RESOLVED,
            resolution="sheltered until storm passed",
            resolved_at=utc_now(),
            actor_id=user_id,
        ),
    )

    gear = GearService(session)
    catalog = gear.create_catalog(
        GearCatalogCreate(sku="ROPE-01", name="Climbing Rope", category="rope"),
        actor_id=user_id,
    )
    inventory = gear.create_inventory(
        GearInventoryCreate(
            catalog_id=catalog.id,
            ownership="club",
            quantity_total=5,
            actor_id=user_id,
            idempotency_key="inventory-create-0001",
        )
    )
    gear.adjust_inventory(
        inventory.id,
        InventoryAdjustment(
            quantity_delta=-2,
            reason="damaged strands removed",
            actor_id=user_id,
            idempotency_key="inventory-adjust-0001",
        ),
    )

    ExpeditionService(session).change_status(
        expedition_id,
        ActivityStateChange(
            target_status="cancelled",
            actor_id=user_id,
            reason="participants withdrew",
        ),
    )

    assert _verify(session, entity_type="expedition", entity_id=expedition_id).status == "ok"
    assert _verify(session, entity_type="gear_inventory", entity_id=inventory.id).status == "ok"
    assert _verify(session, entity_type="emergency_incident", entity_id=incident.id).status == "ok"
    assert len(_links(session, "gear_inventory", inventory.id)) == 2
    assert len(_links(session, "emergency_incident", incident.id)) == 2


# ---------------------------------------------------------------------------
# Tamper detection: mutate a copy of the database file, expect detection
# ---------------------------------------------------------------------------


def test_modifying_record_content_is_detected(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)
    ExpeditionService(session).change_status(
        expedition_id,
        ActivityStateChange(
            target_status="cancelled",
            actor_id=user_id,
            reason="weather window closed",
        ),
    )
    session.commit()

    log = session.scalar(
        select(AuditLog)
        .where(AuditLog.entity_type == "expedition", AuditLog.entity_id == expedition_id)
        .order_by(AuditLog.id)
    )
    # Attacker rewrites the reason for the cancellation afterwards.
    log.context = {"reason": "trail closed by rangers"}
    session.commit()

    report = _verify(session, entity_id=expedition_id)
    assert report.status == "failed"
    assert report.first_break is not None
    assert report.first_break.reason == "payload_mismatch"
    # The response must not expose the rewritten/redacted content.
    assert "rangers" not in str(report.first_break)
    assert report.first_break.expected_hash != report.first_break.actual_hash


def test_deleting_a_row_in_copied_file_leaves_detectable_orphan(
    database: Database,
) -> None:
    with database.session() as session:
        user_id = create_user(session)
        route_id = create_route(session, actor_id=user_id)
        expedition_id = create_expedition(
            session, organizer_id=user_id, route_id=route_id
        )
        ExpeditionService(session).change_status(
            expedition_id,
            ActivityStateChange(
                target_status="cancelled",
                actor_id=user_id,
                reason="weather window closed",
            ),
        )

    # Threat model: the database file was copied and a row removed with a
    # plain SQLite client (which does not enforce foreign keys), leaving the
    # chain link behind.
    import sqlite3

    conn = sqlite3.connect(database.path)
    conn.execute(
        "DELETE FROM audit_logs WHERE entity_type = ? AND entity_id = ? AND action = ?",
        ("expedition", expedition_id, AuditAction.STATUS_CHANGED.value),
    )
    conn.commit()
    conn.close()

    with database.session() as session:
        report = _verify(session, entity_id=expedition_id)
        assert report.status == "failed"
        assert report.first_break.reason == "orphan_link"
        assert report.first_break.audit_log_id is not None
        assert report.first_break.sequence == 2


def test_deleting_a_link_is_detected_as_gap(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)
    service = ExpeditionService(session)
    service.change_status(
        expedition_id,
        ActivityStateChange(target_status="open", actor_id=user_id),
    )
    service.change_status(
        expedition_id,
        ActivityStateChange(
            target_status="cancelled",
            actor_id=user_id,
            reason="weather window closed",
        ),
    )
    assert [link.sequence for link in _links(session, "expedition", expedition_id)] == [
        1,
        2,
        3,
    ]
    session.commit()

    middle = session.scalar(
        select(AuditChainLink)
        .where(
            AuditChainLink.entity_type == "expedition",
            AuditChainLink.entity_id == expedition_id,
            AuditChainLink.sequence == 2,
        )
    )
    session.delete(middle)
    session.commit()

    report = _verify(session, entity_id=expedition_id)
    assert report.status == "failed"
    # Link 3 references a predecessor (sequence 2) that no longer exists.
    assert report.first_break.reason == "sequence_gap"
    assert report.first_break.sequence == 3


def test_rewriting_entry_hash_is_detected(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)
    session.commit()

    link = _links(session, "expedition", expedition_id)[0]
    link.entry_hash = "f" * 64
    session.commit()

    report = _verify(session, entity_id=expedition_id)
    assert report.first_break.reason == "entry_mismatch"


def test_unknown_algorithm_version_is_reported(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)
    session.commit()

    link = _links(session, "expedition", expedition_id)[0]
    link.algorithm = "sha512-chain-v9"
    session.commit()

    report = _verify(session, entity_id=expedition_id)
    assert report.first_break.reason == "unknown_algorithm"


def test_tamper_detection_works_after_database_file_restart(settings: Settings) -> None:
    database = Database(settings)
    initialize_database(database)
    with database.session() as session:
        user_id = create_user(session)
        route_id = create_route(session, actor_id=user_id)
        expedition_id = create_expedition(
            session, organizer_id=user_id, route_id=route_id
        )
    database.engine.dispose()

    # Simulate an offline attacker editing the copied database file directly.
    import sqlite3

    path = settings.database_path
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE audit_logs SET after_state = ? WHERE entity_type = ? AND entity_id = ?",
        ('{"status": "completed"}', "expedition", expedition_id),
    )
    conn.commit()
    conn.close()

    reopened = Database(settings)
    with reopened.session() as session:
        report = _verify(session, entity_id=expedition_id)
        assert report.status == "failed"
        assert report.first_break.reason == "payload_mismatch"
        assert "completed" not in str(report.first_break)
    reopened.engine.dispose()


def test_sealing_respects_redaction_boundary(session: Session) -> None:
    """Sensitive fields stay redacted; the chain seals only sanitized data."""
    from trailforge.services.base import ServiceBase

    log = ServiceBase(session).audit(
        actor_id=None,
        entity_type="expedition",
        entity_id=1,
        action=AuditAction.UPDATED,
        after={"status": "open", "token": "super-secret-value"},
        context={"password": "another-secret"},
    )
    session.commit()

    # Stored record keeps the existing redaction boundary.
    assert log.after_state["token"] == "[REDACTED]"
    assert log.context["password"] == "[REDACTED]"

    report = _verify(session)
    assert report.status == "ok"
    # Verification output (digests and structural fields) never contains the
    # secret values, even after tampering forces a mismatch report.
    link = _links(session)[0]
    link.entry_hash = "0" * 64
    session.commit()
    broken = _verify(session)
    assert broken.status == "failed"
    rendered = str(broken)
    assert "super-secret-value" not in rendered
    assert "another-secret" not in rendered


# ---------------------------------------------------------------------------
# Concurrency: same object -> one unique continuous chain
# ---------------------------------------------------------------------------


def test_concurrent_writes_to_same_object_form_single_chain(database: Database) -> None:
    def append_log(index: int) -> None:
        def operation(session: Session) -> None:
            log = _write_audit(
                session, entity_type="widget", entity_id=77, after={"n": index}
            )
            AuditChainSealer(session).seal(log)

        database.run_write(operation)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append_log, range(16)))

    with database.session() as session:
        links = _links(session, "widget", 77)
        assert len(links) == 16
        assert sorted(link.sequence for link in links) == list(range(1, 17))
        assert {link.audit_log_id for link in links} and len({
            link.audit_log_id for link in links
        }) == len(links)
        # Recursively walk previous_hash links: each must point to the
        # predecessor's stored entry_hash.
        by_sequence = {link.sequence: link for link in links}
        assert by_sequence[1].previous_hash == GENESIS_PREVIOUS_HASH
        for sequence in range(2, 17):
            assert (
                by_sequence[sequence].previous_hash
                == by_sequence[sequence - 1].entry_hash
            )
        report = AuditChainVerifier(session).verify(entity_type="widget", entity_id=77)
        assert report.status == "ok"


def test_duplicate_sequence_insert_is_rejected(session: Session) -> None:
    first_log = _write_audit(session, after={"status": "open"})
    AuditChainSealer(session).seal(first_log)
    second_log = _write_audit(session, after={"status": "draft"})
    # A second link claiming sequence 1 for the same object must be rejected
    # rather than forking the chain.
    rogue = AuditChainLink(
        audit_log_id=second_log.id,
        entity_type="expedition",
        entity_id=1,
        sequence=1,
        algorithm=CHAIN_ALGORITHM_V1,
        occurred_at=second_log.occurred_at,
        payload_hash="a" * 64,
        previous_hash=GENESIS_PREVIOUS_HASH,
        entry_hash="b" * 64,
    )
    session.add(rogue)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


# ---------------------------------------------------------------------------
# Rollback: failed business transactions leave no orphan chain nodes
# ---------------------------------------------------------------------------


def test_business_transaction_rollback_leaves_no_orphan_link(database: Database) -> None:
    with database.session() as session:
        user_id = create_user(session)

    # Invalid adjustment (would make inventory negative): the audit and its
    # link must disappear with the transaction.
    with database.session() as session:
        gear = GearService(session)
        catalog = gear.create_catalog(
            GearCatalogCreate(sku="TENT-01", name="Tent", category="shelter"),
            actor_id=user_id,
        )
        inventory = gear.create_inventory(
            GearInventoryCreate(
                catalog_id=catalog.id,
                ownership="club",
                quantity_total=2,
                actor_id=user_id,
                idempotency_key="inv-create-orphan-01",
            )
        )
        inventory_id = inventory.id

    with pytest.raises(InventoryError), database.session() as session:
        GearService(session).adjust_inventory(
            inventory_id,
            InventoryAdjustment(
                quantity_delta=-99,
                reason="impossible write",
                actor_id=user_id,
                idempotency_key="inv-adjust-orphan-01",
            ),
        )

    with database.session() as session:
        assert session.query(AuditLog).filter(
            AuditLog.entity_type == "gear_inventory",
            AuditLog.entity_id == inventory_id,
            AuditLog.action == AuditAction.INVENTORY_CHANGED,
        ).count() == 0
        assert session.query(AuditChainLink).filter(
            AuditChainLink.entity_type == "gear_inventory",
            AuditChainLink.entity_id == inventory_id,
        ).count() == 1  # only the create link remains
        report = _verify(session, entity_type="gear_inventory", entity_id=inventory_id)
        assert report.status == "ok"


# ---------------------------------------------------------------------------
# Paged verification
# ---------------------------------------------------------------------------


def test_verify_paginates_links_without_losing_context(session: Session) -> None:
    # 5 raw logs, chain them, then verify one page at a time.
    logs = [_write_audit(session, after={"n": n}) for n in range(5)]
    sealer = AuditChainSealer(session)
    for log in logs:
        sealer.seal(log)
    session.commit()

    verifier = AuditChainVerifier(session)
    pages = [
        verifier.verify(entity_type="expedition", entity_id=1, page=1, page_size=2),
        verifier.verify(entity_type="expedition", entity_id=1, page=2, page_size=2),
        verifier.verify(entity_type="expedition", entity_id=1, page=3, page_size=2),
    ]
    assert [page.checked_links for page in pages] == [2, 2, 1]
    assert pages[0].total_links == 5
    assert pages[0].pages == 3
    assert all(page.status == "ok" for page in pages)

    # Tamper with the middle link; the page covering it reports the first
    # break while earlier pages stay clean.
    target = session.scalar(
        select(AuditChainLink)
        .where(
            AuditChainLink.entity_type == "expedition",
            AuditChainLink.entity_id == 1,
            AuditChainLink.sequence == 3,
        )
    )
    target.payload_hash = "9" * 64
    session.commit()

    assert (
        verifier.verify(
            entity_type="expedition", entity_id=1, page=1, page_size=2
        ).status
        == "ok"
    )
    broken = verifier.verify(entity_type="expedition", entity_id=1, page=2, page_size=2)
    assert broken.status == "failed"
    assert broken.first_break.reason == "payload_mismatch"
    assert broken.first_break.sequence == 3


def test_verify_supports_time_range_filter(session: Session) -> None:
    user_id = create_user(session)
    route_id = create_route(session, actor_id=user_id)
    expedition_id = create_expedition(session, organizer_id=user_id, route_id=route_id)
    ExpeditionService(session).change_status(
        expedition_id,
        ActivityStateChange(
            target_status="cancelled",
            actor_id=user_id,
            reason="weather window closed",
        ),
    )
    session.commit()

    links = _links(session, "expedition", expedition_id)
    cutoff = links[1].occurred_at
    report = _verify(
        session,
        entity_id=expedition_id,
        occurred_after=cutoff,
    )
    assert report.total_links == 1
    assert report.status == "ok"


def test_verify_empty_object_returns_empty_status(session: Session) -> None:
    report = _verify(session, entity_type="expedition", entity_id=9999)
    assert report.status == "empty"
    assert report.total_links == 0


# ---------------------------------------------------------------------------
# Backfill: deterministic order, one-shot, resumable after interruption
# ---------------------------------------------------------------------------


def test_backfill_seals_existing_rows_in_deterministic_order(
    database: Database,
) -> None:
    with database.session() as session:
        _write_audit(session, entity_type="thing", entity_id=3, after={"n": 0})
        _write_audit(session, entity_type="thing", entity_id=3, after={"n": 1})
        _write_audit(session, entity_type="thing", entity_id=9, after={"n": 0})

    result = backfill_audit_chain(database, batch_size=10)
    assert result["sealed"] == 3

    with database.session() as session:
        for object_id in (3, 9):
            links = _links(session, "thing", object_id)
            expected = 2 if object_id == 3 else 1
            assert len(links) == expected
            assert [link.sequence for link in links] == list(range(1, expected + 1))
            assert AuditChainVerifier(session).verify(
                entity_type="thing", entity_id=object_id
            ).status == "ok"

    # Re-running is a no-op: nothing is sealed twice.
    assert backfill_audit_chain(database, batch_size=10) == {"sealed": 0, "batches": 0}


def test_backfill_resumes_safely_after_interruption(database: Database) -> None:
    with database.session() as session:
        for index in range(7):
            _write_audit(session, entity_type="bulk", entity_id=5, after={"n": index})

    # First run seals one committed batch of 3, then "crashes".
    with database.session() as session:
        sealed = seal_next_batch(session, batch_size=3)
        assert sealed == 3
    # Simulate interruption mid-second batch: rollback leaves no partial nodes.
    with database.session() as session:
        assert seal_next_batch(session, batch_size=3) == 3
        session.rollback()

    with database.session() as session:
        assert session.query(AuditChainLink).filter(
            AuditChainLink.entity_type == "bulk"
        ).count() == 3

    # Resume: remaining four rows are sealed and the whole chain verifies.
    result = backfill_audit_chain(database, batch_size=3)
    assert result["sealed"] == 4

    with database.session() as session:
        links = _links(session, "bulk", 5)
        assert [link.sequence for link in links] == [1, 2, 3, 4, 5, 6, 7]
        assert AuditChainVerifier(session).verify(
            entity_type="bulk", entity_id=5
        ).status == "ok"


# ---------------------------------------------------------------------------
# API surface: verify endpoint exposes no record content
# ---------------------------------------------------------------------------


def test_verify_endpoint_reports_empty_for_fresh_object(client) -> None:
    response = client.get(
        "/api/v1/audit-chain/verify",
        params={"entity_type": "expedition", "entity_id": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "empty"
    assert body["algorithms"] == []
    assert body["first_break"] is None


def test_verify_endpoint_requires_object_parameters(client) -> None:
    response = client.get("/api/v1/audit-chain/verify", params={"entity_type": "expedition"})
    assert response.status_code == 422


def test_verify_endpoint_detects_tampering(client) -> None:
    database = client.app.state.database
    with database.session() as session:
        user_id = create_user(session)
        route_id = create_route(session, actor_id=user_id)
        expedition_id = create_expedition(
            session, organizer_id=user_id, route_id=route_id
        )
    with database.session() as session:
        log = session.scalar(
            select(AuditLog)
            .where(
                AuditLog.entity_type == "expedition",
                AuditLog.entity_id == expedition_id,
            )
        )
        log.context = {"note": "altered after sealing"}

    response = client.get(
        "/api/v1/audit-chain/verify",
        params={"entity_type": "expedition", "entity_id": expedition_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["first_break"]["reason"] == "payload_mismatch"
    # Only structural/digest fields, never audit content or redacted values.
    assert set(body["first_break"]) <= {
        "reason",
        "link_id",
        "audit_log_id",
        "sequence",
        "expected_hash",
        "actual_hash",
    }
    assert "altered" not in response.text
