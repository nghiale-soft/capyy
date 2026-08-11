from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..deps import check_local_auth


router = APIRouter()


@router.get("/healthz")
async def healthz(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    return {"status": "ok"}
