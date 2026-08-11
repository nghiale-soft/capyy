from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..deps import check_local_auth
from ..services.figma import FigmaTokenStore

router = APIRouter(prefix="/api/figma", tags=["figma"])


def _store(request: Request) -> FigmaTokenStore:
    store = getattr(request.app.state, "figma_tokens", None)
    if store is None:
        store = FigmaTokenStore()
        request.app.state.figma_tokens = store
    return store


@router.get("/tokens")
async def get_tokens(
    request: Request,
    reveal: int = 0,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Return status + masked token pool.

    `?reveal=1` (local dashboard only) also includes the real `value` so the
    UI can show/hide tokens with the eye toggle.
    """
    return _store(request).status(reveal=bool(reveal))


@router.post("/tokens")
async def add_token(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Add one token to the pool."""
    token = str(body.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    try:
        tokens = _store(request).add_token(token)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"ok": True, "status": _store(request).status()}


@router.put("/tokens")
async def replace_tokens(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Replace the whole token pool. Body: {"tokens": ["a", "b"]}."""
    raw = body.get("tokens")
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="tokens must be a list")
    try:
        _store(request).replace_tokens([t for t in raw if isinstance(t, str)])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"ok": True, "status": _store(request).status()}


@router.delete("/tokens/{index}")
async def delete_token(
    request: Request,
    index: int,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Remove one token by index from the pool."""
    try:
        _store(request).remove_token(index)
    except IndexError:
        raise HTTPException(status_code=404, detail="token index out of range")
    return {"ok": True, "status": _store(request).status()}


@router.delete("/tokens")
async def clear_tokens(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Clear the pool (fall back to env / legacy file)."""
    _store(request).clear_all()
    return {"ok": True, "status": _store(request).status()}
