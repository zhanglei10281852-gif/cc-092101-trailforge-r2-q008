from __future__ import annotations

import argparse
import json
from pathlib import Path

from trailforge.audit_chain.backfill import backfill_audit_chains
from trailforge.audit_chain.verify import verify_chains
from trailforge.config import get_settings
from trailforge.database.migrations import (
    assert_database_integrity,
    initialize_database,
    migration_status,
)
from trailforge.database.session import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trailforge", description="TrailForge maintenance CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="create the SQLite schema and apply migrations")
    subparsers.add_parser("migration-status", help="show applied and pending migrations")
    subparsers.add_parser("check-db", help="run SQLite integrity and foreign-key checks")
    backfill = subparsers.add_parser(
        "backfill-audit-chains", help="seal existing audit rows into per-object chains"
    )
    backfill.add_argument("--chains-per-batch", type=int, default=100)
    backfill.add_argument("--max-batches", type=int, default=None)
    verify = subparsers.add_parser("verify-audit-chains", help="verify audit chain seals")
    verify.add_argument("--entity-type", default=None)
    verify.add_argument("--entity-id", type=int, default=None)
    verify.add_argument("--limit-chains", type=int, default=None)
    reset = subparsers.add_parser("reset-db", help="delete and recreate the local SQLite database")
    reset.add_argument("--confirm", action="store_true", help="confirm destructive local reset")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    database = Database(settings)
    if args.command == "init-db":
        applied = initialize_database(database)
        print(json.dumps({"database": str(database.path), "applied": applied}, ensure_ascii=False))
        return 0
    if args.command == "migration-status":
        print(json.dumps(migration_status(database), ensure_ascii=False))
        return 0
    if args.command == "check-db":
        result = assert_database_integrity(database)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["healthy"] else 1
    if args.command == "backfill-audit-chains":
        result = backfill_audit_chains(
            database,
            chains_per_batch=args.chains_per_batch,
            max_batches=args.max_batches,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["remaining"] == 0 else 1
    if args.command == "verify-audit-chains":
        with database.session() as session:
            result = verify_chains(
                session,
                entity_type=args.entity_type,
                entity_id=args.entity_id,
                limit_chains=args.limit_chains,
            )
        payload = {
            "ok": result.ok,
            "chains_checked": result.chains_checked,
            "total_chains": result.total_chains,
            "records_checked": result.records_checked,
            "first_break": (
                None
                if result.first_break is None
                else {
                    "kind": result.first_break.kind,
                    "chain_key": result.first_break.chain_key,
                    "entity_type": result.first_break.entity_type,
                    "entity_id": result.first_break.entity_id,
                    "expected_seq": result.first_break.expected_seq,
                    "actual_seq": result.first_break.actual_seq,
                    "audit_id": result.first_break.audit_id,
                }
            ),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if result.ok else 1
    if args.command == "reset-db":
        if not args.confirm:
            parser = build_parser()
            parser.error("reset-db requires --confirm")
        path = database.path
        if path is None:
            raise SystemExit("reset-db is unavailable for in-memory SQLite")
        database.engine.dispose()
        allowed_suffixes = {".db", ".sqlite", ".sqlite3"}
        if path.suffix.lower() not in allowed_suffixes:
            raise SystemExit("refusing to reset a file without a SQLite extension")
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            if candidate.exists():
                candidate.unlink()
        recreated = Database(settings)
        applied = initialize_database(recreated)
        print(json.dumps({"database": str(path), "applied": applied}, ensure_ascii=False))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
