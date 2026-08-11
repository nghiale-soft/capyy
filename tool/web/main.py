from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

BASE_FREEBUFF = "https://freebuff.com"
BASE_CODEBUFF = "https://www.codebuff.com"
VERIFY_URL = "https://www.codebuff.com/api/v1/freebuff/session"
SUPPORTED_MODES = {"freebuff", "codebuff"}
NO_STORE_HEADERS = {"Cache-Control": "no-store"}

app = FastAPI(title="Freebuff Token Web")
web_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

_sessions: dict[str, dict[str, Any]] = {}


def _endpoints(mode: str) -> tuple[str, str]:
    base = BASE_CODEBUFF if mode == "codebuff" else BASE_FREEBUFF
    return f"{base}/api/auth/cli/code", f"{base}/api/auth/cli/status"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _proxy_url() -> str | None:
    if not _env_bool("FREEBUFF_PROXY_ENABLED", False):
        return None
    return (os.getenv("FREEBUFF_PROXY_URL") or "").strip() or None


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(proxy=_proxy_url(), trust_env=False)


async def _request_code(fingerprint_id: str, code_url: str) -> dict[str, Any]:
    async with _http_client() as client:
        resp = await client.post(
            code_url,
            json={"fingerprintId": fingerprint_id},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Bun/1.3.11",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


async def _poll_status_sse(
    fingerprint_id: str,
    fingerprint_hash: str,
    expires_at: int,
    status_url: str,
):
    qs = urllib.parse.urlencode(
        {
            "fingerprintId": fingerprint_id,
            "fingerprintHash": fingerprint_hash,
            "expiresAt": str(expires_at),
        }
    )
    url = f"{status_url}?{qs}"
    deadline = time.monotonic() + 5 * 60
    attempt = 0

    async with _http_client() as client:
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = await client.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "Bun/1.3.11",
                    },
                    timeout=15,
                )
                if resp.status_code == 401:
                    yield (
                        f'data: {json.dumps({"event": "pending", "attempt": attempt}, ensure_ascii=False)}\n\n'
                    )
                    await asyncio.sleep(2.0)
                    continue
                if resp.status_code >= 400:
                    yield (
                        f'data: {json.dumps({"event": "error", "message": f"HTTP {resp.status_code}"}, ensure_ascii=False)}\n\n'
                    )
                    return

                try:
                    data = resp.json()
                except ValueError:
                    yield (
                        f'data: {json.dumps({"event": "error", "message": "invalid upstream response"}, ensure_ascii=False)}\n\n'
                    )
                    return

                user = data.get("user")
                if user and user.get("authToken"):
                    yield (
                        f'data: {json.dumps({"event": "success", "user": user}, ensure_ascii=False)}\n\n'
                    )
                    return
                yield (
                    f'data: {json.dumps({"event": "pending", "attempt": attempt}, ensure_ascii=False)}\n\n'
                )
            except Exception as e:
                yield (
                    f'data: {json.dumps({"event": "error", "message": str(e)}, ensure_ascii=False)}\n\n'
                )
                return

            await asyncio.sleep(2.0)

        yield f'data: {json.dumps({"event": "timeout"}, ensure_ascii=False)}\n\n'


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = web_dir / "templates" / "index.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers=NO_STORE_HEADERS,
    )


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204, headers=NO_STORE_HEADERS)


@app.post("/api/start")
async def start_auth(request: Request):
    body = await request.json()
    mode = body.get("mode", "freebuff")
    if mode not in SUPPORTED_MODES:
        raise HTTPException(status_code=400, detail="unsupported mode")

    code_url, status_url = _endpoints(mode)

    fingerprint_id = f"fb-{secrets.token_hex(8)}"
    code = await _request_code(fingerprint_id, code_url)

    _sessions[fingerprint_id] = {
        "mode": mode,
        "fingerprint_id": fingerprint_id,
        "fingerprint_hash": code["fingerprintHash"],
        "expires_at": code["expiresAt"],
        "status_url": status_url,
    }

    return {
        "mode": mode,
        "fingerprint_id": fingerprint_id,
        "login_url": code["loginUrl"],
        "expires_at": code["expiresAt"],
    }


@app.get("/api/poll/{fingerprint_id}")
async def poll_auth(fingerprint_id: str):
    session = _sessions.get(fingerprint_id)
    if not session:

        async def _err():
            yield f'data: {json.dumps({"event": "error", "message": "session not found"})}\n\n'

        return StreamingResponse(
            _err(),
            media_type="text/event-stream",
            headers=NO_STORE_HEADERS,
        )

    async def event_generator():
        async for event in _poll_status_sse(
            session["fingerprint_id"],
            session["fingerprint_hash"],
            session["expires_at"],
            session["status_url"],
        ):
            yield event
        _sessions.pop(fingerprint_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=NO_STORE_HEADERS,
    )


@app.post("/api/verify")
async def verify_token_endpoint(request: Request):
    body = await request.json()
    token = body.get("token", "").strip()
    if not token:
        return {"ok": False, "info": "missing token"}

    async with _http_client() as client:
        try:
            resp = await client.get(
                VERIFY_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "*/*",
                    "User-Agent": "Bun/1.3.11",
                },
                timeout=15,
            )
            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "info": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "info": f"network error: {e}"}
