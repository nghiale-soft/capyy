from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..deps import check_local_auth, get_settings
from ..services.provider_crud import ProviderConfig, ProviderCrudService
from providers.openai_compatible import OpenAICompatibleProvider


router = APIRouter(prefix="/api/providers", tags=["providers"])
logger = logging.getLogger("gateway.routes.providers")


def _crud(request: Request) -> ProviderCrudService:
    return request.app.state.provider_crud


def _gateway(request: Request) -> Any:
    return request.app.state.gateway


def _reload_registry(request: Request) -> None:
    reload = getattr(request.app.state, "reload_registry", None)
    if reload is not None:
        reload()


@router.get("")
async def list_providers(
    request: Request,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    crud = _crud(request)
    providers = [p.to_public() for p in crud.list()]
    return {"providers": providers}


@router.post("/fetch-models")
async def fetch_models(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Try to list models from an OpenAI-compatible /models endpoint.

    Returns {"models": [...]} on success, or a 400 with a hint when the
    endpoint is unreachable / not OpenAI-compatible (the UI then falls back
    to manual model entry).
    """
    import httpx

    base_url = str(body.get("base_url") or "").strip().rstrip("/")
    api_key = str(body.get("api_key") or "").strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    for suffix in ("/chat/completions", "/models", "/v1"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
    url = f"{base_url}/v1/models"
    headers = {"Accept": "*/*"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=12.0, trust_env=False) as client:
            resp = await client.get(url, headers=headers)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not reach {url}: {error}. Enter models manually.",
        ) from error
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{url} returned HTTP {resp.status_code} — the API may not expose "
                "a model list. Enter models manually."
            ),
        )
    try:
        data = resp.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail="Response was not JSON. Enter models manually.") from error
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Response was not a JSON object. Enter models manually.")
    ids = [
        str(item["id"])
        for item in data.get("data") or []
        if isinstance(item, dict) and item.get("id")
    ]
    if not ids:
        raise HTTPException(status_code=400, detail="No models returned by the API. Enter them manually.")
    return {"models": ids}


@router.post("/test-config")
async def test_provider_config(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Test the provider values currently entered in the form without saving.

    The dashboard test is a non-destructive connectivity check. It verifies
    that the configured endpoint (or FreeBuff service) is reachable without
    starting a model session, consuming quota, or changing an active session.
    Command providers other than FreeBuff do not have an executor in Capyy, so
    pretending to test them would be misleading.
    """
    cfg = _body_to_config(body, str(body.get("id") or "provider-test"))
    if cfg.source == "command":
        if cfg.command != "freebuff":
            return {
                "ok": False,
                "info": f"Local command '{cfg.command}' cannot be tested because Capyy has no command runner.",
            }
        try:
            await request.app.state.accounts.default_client.health()
        except Exception as error:
            logger.info("FreeBuff connection test failed: %s", error)
            return {"ok": False, "info": f"FreeBuff connection test failed: {error}"}
        return {"ok": True, "info": "FreeBuff service is reachable."}

    if not cfg.base_url:
        raise HTTPException(status_code=400, detail="base_url is required")

    provider = OpenAICompatibleProvider(
        "provider-test",
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        models=cfg.models,
    )
    try:
        ok = await provider.health()
    finally:
        await provider.aclose()
    return {
        "ok": ok,
        "info": "Provider connection and credentials verified." if ok else "Provider rejected the connection or credentials.",
    }


@router.put("/order")
async def reorder_providers(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Set failover priority from an ordered list of provider ids."""
    crud = _crud(request)
    order = body.get("order")
    if not isinstance(order, list) or not order:
        raise HTTPException(status_code=400, detail="order must be a non-empty list")
    try:
        crud.reorder([str(item) for item in order])
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    _reload_registry(request)
    return {"ok": True, "providers": [p.to_public() for p in crud.ordered()]}


@router.post("")
async def create_provider(
    request: Request,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> Any:
    crud = _crud(request)
    provider_id = str(body.get("id") or "").strip()
    if not provider_id:
        # The dashboard uses name as the id (id = name, slugified client-side).
        # Fall back to a server-side slug so API clients never hit a 400.
        name = str(body.get("name") or "").strip()
        if name:
            import re as _re
            provider_id = _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider name is required")
    if crud.get(provider_id) is not None:
        raise HTTPException(status_code=409, detail=f"provider '{provider_id}' exists")
    cfg = _body_to_config(body, provider_id)
    # New providers are appended to the END of the failover order unless the
    # client explicitly provides a priority.
    if "priority" not in body:
        existing = crud.list()
        cfg.priority = max((p.priority for p in existing), default=-1) + 1
    try:
        created = crud.create(cfg)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _reload_registry(request)
    return created.to_public()


@router.get("/{provider_id}")
async def get_provider(
    request: Request,
    provider_id: str,
    _: None = Depends(check_local_auth),
) -> Any:
    crud = _crud(request)
    cfg = crud.get(provider_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="provider not found")
    return cfg.to_public()


@router.put("/{provider_id}")
async def update_provider(
    request: Request,
    provider_id: str,
    body: dict[str, Any],
    _: None = Depends(check_local_auth),
) -> Any:
    crud = _crud(request)
    if crud.get(provider_id) is None:
        raise HTTPException(status_code=404, detail="provider not found")
    changes = {k: v for k, v in body.items() if k != "id"}
    # nếu api_key là "***" bỏ qua để giữ nguyên
    if changes.get("api_key") == "***" or changes.get("api_key") == "":
        changes.pop("api_key", None)
    # Command providers never carry an HTTP type; normalize to avoid storing
    # type="" which would break registry.build_from_config on the next reload.
    if changes.get("source") == "command" or (
        changes.get("command") and changes.get("command") != "freebuff"
    ):
        changes.pop("type", None)
        changes.pop("base_url", None)
    try:
        updated = crud.update(provider_id, changes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _reload_registry(request)
    return updated.to_public()


@router.delete("/{provider_id}")
async def delete_provider(
    request: Request,
    provider_id: str,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    crud = _crud(request)
    try:
        crud.delete(provider_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="provider not found") from error
    _reload_registry(request)
    return {"ok": True}


@router.post("/{provider_id}/test")
async def test_provider(
    request: Request,
    provider_id: str,
    _: None = Depends(check_local_auth),
) -> dict[str, Any]:
    gateway = _gateway(request)
    try:
        ok = await gateway.health(provider_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="provider not found")
    except Exception as error:
        return {"ok": False, "info": str(error)}
    return {"ok": ok, "info": "provider reachable" if ok else "provider unreachable"}


def _body_to_config(body: dict[str, Any], provider_id: str) -> ProviderConfig:
    source = str(body.get("source") or "url")
    command = str(body.get("command") or "")
    if source == "command" and not command:
        command = "freebuff"
    return ProviderConfig(
        id=provider_id,
        name=str(body.get("name") or provider_id),
        source=source,
        command=command,
        type=str(body.get("type") or ("freebuff" if command == "freebuff" else "openai-compatible")),
        base_url=str(body.get("base_url") or ""),
        api_key=body.get("api_key"),
        models=[str(m) for m in body.get("models") or []],
        enabled=bool(body.get("enabled", True)),
        default=bool(body.get("default", False)),
        priority=int(body.get("priority") or 0),
        extra=dict(body.get("extra") or {}),
    )
