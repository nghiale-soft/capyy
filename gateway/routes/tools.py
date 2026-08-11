from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..deps import check_local_auth
from ..services.tool_approval import TOOL_LABELS, TOOL_MODES


router = APIRouter(prefix="/api/tools", tags=["tools"])
logger = logging.getLogger("gateway.routes.tools")


def _tools(request: Request) -> Any:
    return request.app.state.tool_approval


@router.get("/permissions")
async def get_permissions(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Per-tool approval modes (allow / ask / deny) + labels for the UI."""
    service = _tools(request)
    return {
        "permissions": service.permissions(),
        "labels": TOOL_LABELS,
        "modes": list(TOOL_MODES),
        "timeout": service.settings.tool_approval_timeout,
    }


@router.put("/permissions")
async def set_permissions(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Update one or more tool modes. Body: {"permissions": {"bash": "ask"}}."""
    body = await request.json()
    updates = body.get("permissions") or {}
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="permissions must be an object")
    for tool, mode in updates.items():
        if mode not in TOOL_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"invalid mode '{mode}' for tool '{tool}' (allowed: {TOOL_MODES})",
            )
    service = _tools(request)
    for tool, mode in updates.items():
        service.set_permission(str(tool), str(mode))
    logger.info("updated tool permissions: %s", updates)
    return {"ok": True, "permissions": service.permissions()}


@router.get("/pending")
async def list_pending(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Pending tool approvals waiting for the user on the dashboard."""
    service = _tools(request)
    return {"pending": service.list_pending(), "timeout": service.settings.tool_approval_timeout}


@router.post("/pending/{approval_id}/{decision}")
async def decide_pending(
    request: Request,
    approval_id: str,
    decision: str,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Approve or deny a pending tool approval (wakes the waiting agent loop)."""
    if decision not in ("approve", "deny"):
        raise HTTPException(
            status_code=400,
            detail="decision must be 'approve' or 'deny'",
        )
    service = _tools(request)
    resolved = await service.decide(
        approval_id,
        "approved" if decision == "approve" else "denied",
    )
    if not resolved:
        raise HTTPException(status_code=404, detail="approval not found or already resolved")
    return {"ok": True, "pending": service.list_pending()}
