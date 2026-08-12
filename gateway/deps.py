from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from .core.config import Settings


def record_contribution(
    request: Request,
    kind: str,
    title: str,
    summary: str,
    metadata: dict[str, str],
) -> None:
    """Best-effort local draft creation; never let telemetry affect a request."""
    try:
        request.app.state.contributions.add(kind, title, summary, metadata)
    except Exception:
        # Contributions are optional and must never mask the original error.
        pass


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def detect_client(request: Request) -> str:
    """Best-effort detection of which client is calling the gateway.

    Reads User-Agent + common client headers (x-app / x-title / x-client)
    so logs and chat history can say whether the request came from the
    Claude Code extension/CLI, Cline, OpenAI Codex, Cursor, a raw API
    caller or an SDK. Unknown callers fall back to "api".
    """
    headers = request.headers
    ua = (headers.get("user-agent") or "").lower()
    x_app = (headers.get("x-app") or "").lower()
    x_title = (headers.get("x-title") or "").lower()
    x_client = (headers.get("x-client") or "").lower()
    combined = " ".join([ua, x_app, x_title, x_client])

    # Claude Code CLI/extension: claude-cli/x.y.z (...), x-app: cli
    if (
        "claude-cli" in ua
        or "claude-code" in ua
        or x_app == "cli"
        or "claude" in combined
    ):
        return "claude-code"
    # OpenAI Codex CLI: codex_cli_rs/...
    if "codex" in combined:
        return "codex"
    # Cline: often sends x-title: Cline (LiteLLM convention)
    if "cline" in combined:
        return "cline"
    # Cursor: User-Agent: Cursor/x.y.z
    if "cursor" in combined:
        return "cursor"
    # Generic SDK callers
    if "openai" in combined or "anthropic-sdk" in combined or "stainless" in combined:
        return "api-sdk"
    return "api"


def provider_label(request: Request, model: str) -> str:
    """Best-effort display name for the provider serving a model.

    Falls back to the model id when the provider cannot be resolved.
    """
    try:
        gateway = request.app.state.gateway
        provider_id, _ = gateway.resolve(model)
        cfg = request.app.state.provider_crud.get(provider_id)
        return (cfg.name if cfg and cfg.name else provider_id) if provider_id else model
    except Exception:
        return model


def check_local_auth(request: Request) -> None:
    settings = get_settings(request)
    api_key = settings.local_api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if request.headers.get("authorization") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def error_response(error: Exception, request: Request | None = None) -> Any:
    from providers.freebuff import CodebuffError
    from providers.openai_compatible import GatewayProviderError
    from fastapi.responses import JSONResponse

    if isinstance(error, CodebuffError):
        if request is not None:
            record_contribution(
                request,
                "provider",
                "Freebuff upstream request failed",
                "Freebuff could not complete a request after its normal retry and failover handling.",
                {"provider": "freebuff", "status_code": str(error.status_code)},
            )
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            },
        )
    if isinstance(error, GatewayProviderError):
        if request is not None:
            record_contribution(
                request,
                "provider",
                "Provider request failed",
                "An OpenAI-compatible provider could not complete a request.",
                {"status_code": str(error.status_code or 502)},
            )
        return JSONResponse(
            status_code=error.status_code or 502,
            content={
                "error": {
                    "message": str(error),
                    "type": "provider_error",
                    "code": "provider_unavailable",
                }
            },
        )
    raise error
