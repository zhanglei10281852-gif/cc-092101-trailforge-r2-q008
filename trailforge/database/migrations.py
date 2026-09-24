from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import inspect, text

from trailforge.database.session import Database
from trailforge.models.audit import SchemaMigration


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    upgrade: Callable[[Database], None] | None = None


def _existing_columns(database: Database, table: str) -> set[str]:
    inspector = inspect(database.engine)
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def _existing_indexes(database: Database, table: str) -> set[str]:
    inspector = inspect(database.engine)
    if table not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table)}


def _upgrade_0002_audit_chains(database: Database) -> None:
    """Add the per-object audit chain seal to pre-existing databases.

    Every statement is idempotent so it is safe on a fresh schema (where
    create_all already created the objects) and on an interrupted rerun.
    """
    column_types = {
        "chain_key": "VARCHAR(200)",
        "chain_seq": "INTEGER",
        "chain_version": "VARCHAR(20)",
        "prev_digest": "VARCHAR(64)",
        "record_digest": "VARCHAR(64)",
    }
    existing = _existing_columns(database, "audit_logs")
    indexes = _existing_indexes(database, "audit_logs")
    with database.engine.begin() as connection:
        for name, column_type in column_types.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE audit_logs ADD COLUMN {name} {column_type}"))

        if "uq_audit_chain_position" not in indexes:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX uq_audit_chain_position "
                    "ON audit_logs (chain_key, chain_seq)"
                )
            )
        if "ix_audit_logs_record_digest" not in indexes:
            connection.execute(
                text("CREATE INDEX ix_audit_logs_record_digest ON audit_logs (record_digest)")
            )
        if "ix_audit_unsealed" not in indexes:
            connection.execute(
                text("CREATE INDEX ix_audit_unsealed ON audit_logs (id) WHERE chain_key IS NULL")
            )


MIGRATIONS = [
    Migration(version="0001", description="Initial TrailForge schema"),
    Migration(
        version="0002",
        description="Per-object tamper-evident audit chains",
        upgrade=_upgrade_0002_audit_chains,
    ),
]


def initialize_database(database: Database) -> list[str]:
    database.create_schema()
    applied: list[str] = []
    with database.session() as session:
        known = {
            row.version
            for row in session.query(SchemaMigration).order_by(SchemaMigration.version).all()
        }
        pending = [migration for migration in MIGRATIONS if migration.version not in known]
    for migration in pending:
        if migration.upgrade is not None:
            migration.upgrade(database)
        with database.session() as session:
            session.add(
                SchemaMigration(
                    version=migration.version,
                    description=migration.description,
                )
            )
        applied.append(migration.version)
    return applied


def migration_status(database: Database) -> dict[str, object]:
    inspector = inspect(database.engine)
    if "schema_migrations" not in inspector.get_table_names():
        return {
            "initialized": False,
            "applied": [],
            "pending": [item.version for item in MIGRATIONS],
        }
    with database.session() as session:
        applied = [
            row.version
            for row in session.query(SchemaMigration).order_by(SchemaMigration.version).all()
        ]
    pending = [item.version for item in MIGRATIONS if item.version not in set(applied)]
    return {"initialized": True, "applied": applied, "pending": pending}


def assert_database_integrity(database: Database) -> dict[str, object]:
    with database.engine.connect() as connection:
        integrity = connection.exec_driver_sql("PRAGMA integrity_check").scalar_one()
        foreign_key_rows = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
    return {
        "integrity_check": str(integrity),
        "foreign_key_violations": [list(row) for row in foreign_key_rows],
        "healthy": integrity == "ok" and not foreign_key_rows,
    }
