"""Dashboard web app — runs on its own port (default 2222).

Serves the management UI at ``/`` and proxies every ``/api/*`` to the main
gateway (API port). This lets the dashboard reuse existing APIs without CORS
and without occupying paths on the API port.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .core.config import load_settings

logger = logging.getLogger("gateway.webapp")

_WEB_DIR = Path(__file__).resolve().parent.parent / "tool" / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

# Headers not forwarded when proxying (httpx sets host/connection itself,
# content-length is recomputed from the body).
_PROXY_SKIP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "accept-encoding",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    app.state.gateway_base = f"http://127.0.0.1:{settings.port}"
    logger.info("dashboard web listening; proxying /api/* -> %s", app.state.gateway_base)
    yield


app = FastAPI(title="capyy-dashboard", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard_index() -> HTMLResponse:
    content = (_TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/favicon.ico")
async def favicon() -> Response:
    return FileResponse(_STATIC_DIR / "favicon.png", media_type="image/png")


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_api(path: str, request: Request) -> Response:
    """Forward the request to the main gateway, keeping method/headers/body."""
    gateway_base: str = request.app.state.gateway_base
    target = f"{gateway_base}/api/{path}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PROXY_SKIP_HEADERS
    }
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            upstream = await client.request(
                request.method,
                target,
                headers=headers,
                content=body,
                params=dict(request.query_params),
            )
    except httpx.HTTPError as error:
        logger.warning("proxy %s failed: %s", target, error)
        return JSONResponse(
            status_code=502,
            content={"detail": f"gateway unreachable: {error}"},
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _PROXY_SKIP_HEADERS
        and key.lower() != "content-type"
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/json",
        headers=response_headers,
    )
