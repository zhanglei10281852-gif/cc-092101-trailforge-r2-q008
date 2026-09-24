from __future__ import annotations

import json

from sqlalchemy import text

from trailforge.audit_chain.backfill import backfill_audit_chains
from trailforge.audit_chain.verify import verify_chains
from trailforge.config import Settings
from trailforge.database.migrations import initialize_database, migration_status
from trailforge.database.session import Database

OLD_AUDIT_SCHEMA = """
CREATE TABLE audit_logs (
    id INTEGER NOT NULL PRIMARY KEY,
    actor_id INTEGER,
    occurred_at VARCHAR(32) NOT NULL,
    entity_type VARCHAR(80) NOT NULL,
    entity_id INTEGER NOT NULL,
    action VARCHAR(40) NOT NULL,
    before_state JSON NOT NULL,
    after_state JSON NOT NULL,
    context JSON NOT NULL,
    correlation_id VARCHAR(120),
    FOREIGN KEY(actor_id) REFERENCES users (id) ON DELETE SET NULL
)
"""


def _build_legacy_database(settings: Settings) -> Database:
    """Create a database exactly as it looked after migration 0001."""
    import sqlite3

    assert settings.database_path is not None
    connection = sqlite3.connect(settings.database_path)
    try:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                id INTEGER NOT NULL PRIMARY KEY,
                version VARCHAR(60) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                applied_at VARCHAR(32) NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) "
            "VALUES ('0001', 'Initial TrailForge schema', '2026-01-01T00:00:00Z')"
        )
        connection.execute(OLD_AUDIT_SCHEMA)
        connection.execute(
            "INSERT INTO audit_logs "
            "(actor_id, occurred_at, entity_type, entity_id, action, "
            " before_state, after_state, context, correlation_id) "
            "VALUES (NULL, :ts, 'emergency_incident', 1, 'status_changed', "
            " '{}', :after, '{}', NULL)",
            {
                "ts": "2026-01-01T00:00:00Z",
                "after": json.dumps({"status": "resolved"}),
            },
        )
        connection.commit()
    finally:
        connection.close()
    return Database(settings)


def test_migration_0002_upgrades_legacy_database(settings: Settings) -> None:
    legacy = _build_legacy_database(settings)
    try:
        applied = initialize_database(legacy)
        assert applied == ["0002"]
        # Idempotent: a second run changes nothing.
        assert initialize_database(legacy) == []
        status = migration_status(legacy)
        assert status["applied"] == ["0001", "0002"]
        assert status["pending"] == []

        # Legacy rows are present but unsealed until the explicit backfill.
        with legacy.session() as session:
            columns = {
                row[1]
                for row in session.execute(text("PRAGMA table_info(audit_logs)")).all()
            }
            assert {
                "chain_key",
                "chain_seq",
                "chain_version",
                "prev_digest",
                "record_digest",
            } <= columns

        result = backfill_audit_chains(legacy)
        assert result["remaining"] == 0
        with legacy.session() as session:
            verification = verify_chains(session)
            assert verification.ok, verification.first_break
            assert verification.records_checked == 1

        # Normal writes after the upgrade continue sealing.
        from trailforge.domain.enums import AuditAction
        from trailforge.services.base import ServiceBase

        with legacy.session() as session:
            ServiceBase(session).audit(
                actor_id=None,
                entity_type="emergency_incident",
                entity_id=1,
                action=AuditAction.STATUS_CHANGED,
                after={"status": "closed"},
            )
        with legacy.session() as session:
            assert verify_chains(session).ok
    finally:
        legacy.engine.dispose()


def test_migration_0002_is_noop_on_fresh_database(database: Database) -> None:
    status = migration_status(database)
    assert "0002" in status["applied"]
    # Running the upgrade statements again must not error (indexes exist).
    assert initialize_database(database) == []
