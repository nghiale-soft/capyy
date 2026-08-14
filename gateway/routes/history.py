from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ..deps import check_local_auth, record_contribution
from ..services.chat_history import _sanitize_filename
from ..services.history_scan import scan_local_history


router = APIRouter(prefix="/api/history", tags=["history"])
logger = logging.getLogger("gateway.routes.history")


def _history(request: Request) -> Any:
    return request.app.state.chat_history


@router.get("")
async def list_history(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Return lightweight conversation previews; never read chat transcripts."""
    service = _history(request)
    return {
        "conversations": service.conversations(),
        "stats": service.stats(),
    }


@router.post("/scan")
async def scan_external_history(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Scan local chat history from Claude Code / Codex and import automatically."""
    service = _history(request)
    # Parsing local JSONL sessions can take seconds for a long-lived workspace;
    # keep it off the async API loop so gateway requests remain responsive.
    try:
        result = await run_in_threadpool(scan_local_history, service)
        await run_in_threadpool(service.refresh_conversation_index)
    except Exception:
        record_contribution(
            request,
            "history",
            "Local history scan failed",
            "Capyy could not scan a local Claude or Codex history source.",
            {"route": "/api/history/scan"},
        )
        raise
    logger.info(
        "scanned local history records_imported=%s sources=%s",
        result["records_imported"],
        [s["source"] for s in result["sources"]],
    )
    return result


@router.get("/{project_key}")
async def get_history(
    request: Request,
    project_key: str,
    limit: int = 200,
    offset: int = 0,
    session_id: str | None = None,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Read the newest messages of one project (paginated backwards)."""
    service = _history(request)
    key = _sanitize_filename(project_key)
    page_limit = min(max(limit, 1), 1000)
    rows = service.messages(
        key,
        session_id=session_id,
        # Filter a session before paging; otherwise a busy project's newest
        # messages could hide an older but selected conversation.
        limit=1_000_000 if session_id else page_limit,
        offset=0 if session_id else max(offset, 0),
        newest_first_page=not session_id,
    )
    if session_id:
        rows = rows[-page_limit:]
    return {"project": key, "count": len(rows), "records": rows}


@router.get("/{project_key}/sessions")
async def get_history_sessions(
    request: Request, project_key: str, _: None = Depends(check_local_auth)
) -> dict[str, Any]:
    key = _sanitize_filename(project_key)
    return {"project": key, "sessions": _history(request).sessions(key)}


@router.delete("/{project_key}")
async def delete_history(
    request: Request,
    project_key: str,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Delete all history of one project."""
    service = _history(request)
    key = _sanitize_filename(project_key)
    service.delete_project(key)
    logger.info("deleted chat history project=%s", key)
    return {"ok": True, "project": key}
