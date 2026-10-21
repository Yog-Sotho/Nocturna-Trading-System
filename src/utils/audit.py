"""
NOCTURNA Trading System - Audit trail.

Security- and trading-relevant events are written to the ``audit_logs`` table
and mirrored to the audit log file. Database writes use their own session, so
an event is kept even if the caller's transaction later rolls back, and a
failure to record an event never breaks the request.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import g, has_request_context, request
from sqlalchemy.orm import Session

from src.models.user import AuditLog, db
from src.utils.logger import get_audit_logger, sanitize_details

logger = logging.getLogger(__name__)


def _actor_id(details: dict[str, Any]) -> int | None:
    """The authenticated caller, else the user an event names (e.g. at login)."""
    caller = str(getattr(g, "user_id", "") or "") if has_request_context() else ""
    if caller.isdigit():
        return int(caller)
    named = details.get("user_id")
    return named if isinstance(named, int) and not isinstance(named, bool) else None


def record_audit_event(
    event_type: str,
    success: bool,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """
    Record an audit event.

    Args:
        event_type: Upper-case event name, e.g. LOGIN_FAILED or EMERGENCY_STOP.
        success: Whether the audited action succeeded.
        details: Event data; sensitive keys are redacted before storage.
        error: Optional error description.
    """
    clean = json.loads(json.dumps(sanitize_details(details or {}), default=str))
    user_id = _actor_id(clean)
    request_id = g.get("request_id") if has_request_context() else None

    file_logger = get_audit_logger()
    if file_logger and file_logger.audit_logger:
        file_logger.log_event(
            event_type=event_type,
            user_id=str(user_id) if user_id else "anonymous",
            details=clean,
            success=success,
            request_id=request_id,
        )

    row: dict[str, Any] = {
        "event_type": event_type[:50],
        "success": success,
        "event_data": clean,
        "user_id": user_id,
        "error_message": error,
        "request_id": request_id,
    }
    if has_request_context():
        row.update(
            username=(getattr(g, "username", None) or None),
            ip_address=(request.remote_addr or "")[:45] or None,
            user_agent=request.user_agent.string[:500] or None,
            endpoint=(request.endpoint or request.path)[:200],
            method=request.method,
        )
    try:
        with Session(db.engine) as session:
            session.add(AuditLog(**row))
            session.commit()
    except Exception:
        logger.exception("Failed to persist audit event %s", event_type)
