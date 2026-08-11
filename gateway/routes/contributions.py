from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..deps import check_local_auth

router = APIRouter(prefix="/api/contributions", tags=["contributions"])
_ALLOWED_KINDS = {"tool-mapping", "tool-protocol", "provider", "history", "dashboard", "other"}
_MAX_TEXT = 280
_MAX_METADATA = 12


def _service(request: Request) -> Any:
    return request.app.state.contributions


@router.get("")
async def list_contributions(request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    service = _service(request)
    return {"repository": service.repository, "contributions": service.list()}


@router.post("")
async def add_contribution(request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    body = await request.json()
    kind, title, summary, metadata = (body.get("kind"), body.get("title"), body.get("summary"), body.get("metadata"))
    valid_strings = all(isinstance(value, str) and 0 < len(value.strip()) <= _MAX_TEXT for value in (title, summary))
    valid_metadata = (
        isinstance(metadata, dict)
        and len(metadata) <= _MAX_METADATA
        and all(isinstance(key, str) and isinstance(value, str) and len(key) <= 80 and len(value) <= _MAX_TEXT for key, value in metadata.items())
    )
    if kind not in _ALLOWED_KINDS or not valid_strings or not valid_metadata:
        raise HTTPException(400, "Invalid contribution metadata")
    return {"contribution": _service(request).add(kind, title.strip(), summary.strip(), metadata)}


@router.post("/{contribution_id}/approve")
async def approve(contribution_id: str, request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    url = _service(request).issue_url(contribution_id)
    if not url:
        raise HTTPException(404, "Pending contribution or repository not found")
    return {"issue_url": url}


@router.post("/{contribution_id}/deny")
async def deny(contribution_id: str, request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    if not _service(request).deny(contribution_id):
        raise HTTPException(404, "Pending contribution not found")
    return {"ok": True}
