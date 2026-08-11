from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request
from ..deps import check_local_auth

router = APIRouter(prefix="/api/tool-mappings", tags=["tool-mappings"])
def _service(request: Request) -> Any: return request.app.state.tool_mapping_contributions

@router.get("/contributions")
async def contributions(request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    service = _service(request); return {"repository": service.repository, "contributions": service.list()}

@router.post("/contributions")
async def add_contribution(request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    body = await request.json(); mapping = body.get("argument_mapping") or {}
    if not all(isinstance(body.get(key), str) and body[key] for key in ("client", "upstream_tool", "client_tool")) or not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise HTTPException(400, "Only client/tool names and argument-key mapping are accepted")
    return {"contribution": _service(request).add(body["client"], body["upstream_tool"], body["client_tool"], mapping)}

@router.post("/contributions/{contribution_id}/approve")
async def approve(contribution_id: str, request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    url = _service(request).issue_url(contribution_id)
    if not url: raise HTTPException(404, "Pending contribution or repository not found")
    return {"issue_url": url}

@router.post("/contributions/{contribution_id}/deny")
async def deny(contribution_id: str, request: Request, _: None = Depends(check_local_auth)) -> dict[str, Any]:
    if not _service(request).deny(contribution_id): raise HTTPException(404, "Pending contribution not found")
    return {"ok": True}
