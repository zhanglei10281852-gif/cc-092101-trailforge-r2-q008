from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from trailforge.api.dependencies import get_session
from trailforge.audit_chain.backfill import backfill_audit_chains
from trailforge.audit_chain.verify import verify_chains
from trailforge.database.session import Database
from trailforge.errors import ValidationError
from trailforge.schemas.audit import (
    AuditChainBackfillResponse,
    AuditChainVerificationResponse,
)
from trailforge.schemas.common import require_aware

router = APIRouter(prefix="/audit", tags=["audit"])
SessionDep = Annotated[Session, Depends(get_session)]


def get_database(request: Request) -> Database:
    return request.app.state.database


DatabaseDep = Annotated[Database, Depends(get_database)]


def _require_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    try:
        return require_aware(value)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


@router.get("/chains/verify", response_model=AuditChainVerificationResponse)
def verify_audit_chains(
    session: SessionDep,
    entity_type: str | None = Query(default=None, max_length=80),
    entity_id: int | None = Query(default=None, gt=0),
    occurred_from: datetime | None = None,
    occurred_to: datetime | None = None,
    limit_chains: int = Query(default=100, ge=1, le=1000),
    offset_chains: int = Query(default=0, ge=0),
) -> AuditChainVerificationResponse:
    if entity_id is not None and entity_type is None:
        raise ValidationError("entity_id requires entity_type")
    result = verify_chains(
        session,
        entity_type=entity_type,
        entity_id=entity_id,
        occurred_from=_require_aware(occurred_from),
        occurred_to=_require_aware(occurred_to),
        limit_chains=limit_chains,
        offset_chains=offset_chains,
    )
    return AuditChainVerificationResponse.model_validate(result, from_attributes=True)


@router.post("/chains/backfill", response_model=AuditChainBackfillResponse)
def backfill_audit_chain_seals(
    database: DatabaseDep,
    chains_per_batch: int = Query(default=100, ge=1, le=5000),
    max_batches: int | None = Query(default=None, ge=1),
) -> AuditChainBackfillResponse:
    result = backfill_audit_chains(
        database,
        chains_per_batch=chains_per_batch,
        max_batches=max_batches,
    )
    return AuditChainBackfillResponse(**result)
