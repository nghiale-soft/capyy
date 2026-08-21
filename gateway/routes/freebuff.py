from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..deps import check_local_auth


router = APIRouter(prefix="/api/freebuff", tags=["freebuff"])


def _mask_token(token: str, keep: int = 4) -> str:
    """Mask a token for the UI — keep only a few leading/trailing chars."""
    if len(token) <= keep:
        return "*" * len(token)
    if len(token) <= keep + 4:
        return "*" * min(len(token) - keep, 4) + token[-keep:]
    return token[:3] + "*" * 8 + token[-keep:]


@router.get("/models")
async def list_freebuff_models(
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """FreeBuff model catalog for the provider form's model picker.

    This is intentionally the small, current FreeBuff picker catalog.  It is
    not ALL_MODELS: that compatibility catalog also contains retired ids and
    internal sub-agent routes that must never be offered as a direct provider
    selection in the dashboard.
    """
    from ..compat.models import FREEBUFF_PICKER_MODELS

    return {"models": [model.id for model in FREEBUFF_PICKER_MODELS]}


@router.get("/tokens")
async def get_tokens(
    request: Request,
    reveal: int = 0,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Return status + masked token list.

    `?reveal=1` (local dashboard only) also includes the real `value` for
    each token so the UI can show/hide it with the eye toggle.
    """
    accounts = request.app.state.accounts
    statuses = {
        item["index"]: item
        for item in accounts.token_statuses()
    }
    tokens = []
    for i, t in enumerate(accounts.tokens):
        entry: dict[str, Any] = {
            "index": i,
            "masked": _mask_token(t),
            **statuses.get(
                i,
                {
                    "status": "available",
                    "is_default": False,
                    "retry_at": None,
                    "last_error_status": None,
                },
            ),
        }
        if reveal:
            entry["value"] = t
        tokens.append(entry)
    return {
        "source": accounts.token_source,
        "account_count": accounts.account_count,
        "configured": accounts.account_count > 0,
        "tokens": tokens,
    }


@router.post("/tokens")
async def add_token(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Add one token to the config file and reload the pool immediately."""
    token = str(body.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    accounts = request.app.state.accounts
    current = list(accounts.tokens)
    if token in current:
        raise HTTPException(status_code=400, detail="token already exists")
    remaining = await accounts.update_tokens([*current, token])
    return {
        "ok": True,
        "account_count": accounts.account_count,
        "source": accounts.token_source,
        "tokens": [
            {"index": i, "masked": _mask_token(t)}
            for i, t in enumerate(remaining)
        ],
    }


@router.put("/tokens")
async def update_tokens(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Write the token list to the config file and reload the pool immediately."""
    raw = body.get("tokens")
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="tokens must be a list")
    tokens = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    if not tokens:
        raise HTTPException(status_code=400, detail="at least one token is required")

    accounts = request.app.state.accounts
    remaining = await accounts.update_tokens(tokens)
    return {
        "ok": True,
        "account_count": accounts.account_count,
        "source": accounts.token_source,
        "tokens": [
            {"index": i, "masked": _mask_token(t)}
            for i, t in enumerate(remaining)
        ],
    }


@router.delete("/tokens/{index}")
async def delete_token(
    request: Request,
    index: int,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Delete one token by index from the config file."""
    accounts = request.app.state.accounts
    try:
        remaining = await accounts.remove_token(index)
    except IndexError:
        raise HTTPException(status_code=404, detail="token index out of range")
    return {
        "ok": True,
        "account_count": accounts.account_count,
        "source": accounts.token_source,
        "tokens": [
            {"index": i, "masked": _mask_token(t)}
            for i, t in enumerate(remaining)
        ],
    }


@router.delete("/tokens")
async def clear_tokens(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Delete the dashboard-managed token config file and clear the pool."""
    accounts = request.app.state.accounts
    await accounts.clear_tokens()
    return {
        "ok": True,
        "account_count": accounts.account_count,
        "source": accounts.token_source,
    }
