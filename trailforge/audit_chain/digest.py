"""Versioned, deterministic hashing for per-object audit chains.

Every audit log row is sealed as a node in a per-business-object hash chain.
The digest is computed over a canonical JSON serialization of the row's
*stored* fields (which are already sanitized at the masking boundary), so
verification never needs to touch unmasked content and never depends on any
external service.

The algorithm carries an explicit version identifier so future upgrades can
change the serialization or hash function without breaking verification of
older chains.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Version identifier for the chain algorithm. Bump (and register a new
# builder) when the payload layout, serialization, or hash function changes.
V1 = "v1"

SUPPORTED_CHAIN_VERSIONS: frozenset[str] = frozenset({V1})

# Digest stored on the first node of every chain.
GENESIS_PREV_DIGEST = "GENESIS"

# Stable fields sealed into the digest, in declaration order. These are the
# columns written once at insert time and never mutated afterwards.
V1_FIELDS: tuple[str, ...] = (
    "actor_id",
    "occurred_at",
    "entity_type",
    "entity_id",
    "action",
    "before_state",
    "after_state",
    "context",
    "correlation_id",
)


def chain_key(entity_type: str, entity_id: int) -> str:
    """Deterministic identifier of the chain a record belongs to."""
    return f"{entity_type}:{entity_id}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must include timezone information")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a payload so the same logical content always hashes identically."""
    normalized = {key: _json_value(value) for key, value in payload.items()}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_v1(
    *,
    chain_key: str,
    chain_seq: int,
    prev_digest: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": V1,
        "chain_key": chain_key,
        "chain_seq": chain_seq,
        "prev_digest": prev_digest,
    }
    for name in V1_FIELDS:
        payload[name] = fields.get(name)
    return payload


def compute_payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compute_digest(
    *,
    version: str,
    chain_key: str,
    chain_seq: int,
    prev_digest: str,
    fields: dict[str, Any],
) -> str:
    """Compute the record digest for a supported chain algorithm version."""
    if version == V1:
        return compute_payload_digest(
            _payload_v1(
                chain_key=chain_key,
                chain_seq=chain_seq,
                prev_digest=prev_digest,
                fields=fields,
            )
        )
    raise ValueError(f"unsupported audit chain version: {version}")
