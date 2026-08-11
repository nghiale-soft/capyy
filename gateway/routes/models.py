from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..deps import check_local_auth, get_settings


router = APIRouter()


@router.get("/v1/models")
async def list_models(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    gateway = request.app.state.gateway
    settings = get_settings(request)
    try:
        return await gateway.list_models()
    except Exception as error:
        if settings.debug:
            request.app.state.logger.exception("model list failed")
        raise error
