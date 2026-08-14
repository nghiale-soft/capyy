from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from providers.freebuff import CodebuffError

from ..core.logging import log_curl, redact_headers, render_debug
from ..deps import (
    check_local_auth,
    detect_client,
    error_response,
    get_settings,
    provider_label,
)
from ..services.chat_history import inject_context
from ..services.chat_service import (
    build_payload,
    build_session_recover_callback,
    collect_completion,
    prepare_freebuff_dispatch,
    run_tool_loop_pass,
    start_freebuff_run_chain,
    stream_openai_chunks,
    stream_tool_agent_loop,
)
from ..compat.openai import normalize_chat_messages


router = APIRouter()
logger = logging.getLogger("gateway.routes.chat")


def _contribution_reporter(request: Request) -> Any:
    def report(kind: str, title: str, summary: str, metadata: dict[str, str]) -> None:
        try:
            request.app.state.contributions.add(
                kind,
                title,
                summary,
                {**metadata, "client": detect_client(request), "route": "openai"},
            )
        except Exception:
            logger.exception("failed to persist contribution kind=%s", kind)
    return report


async def _safe_stream(agen: Any) -> Any:
    """Convert failover/upstream errors inside a stream into a valid SSE event."""
    from providers.openai_compatible import GatewayProviderError
    from ..core.sse import encode_sse

    try:
        async for chunk in agen:
            yield chunk
    except GatewayProviderError as error:
        yield encode_sse(
            {
                "error": {
                    "message": str(error),
                    "type": "provider_error",
                    "code": "provider_unavailable",
                }
            }
        )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    _: None = Depends(check_local_auth),
) -> Any:
    body = await request.json()
    log_curl(request, body, logger=logger)
    settings = get_settings(request)
    gateway = request.app.state.gateway

    # Route provider theo model prefix
    try:
        provider_id, real_model = gateway.resolve(body.get("model"))
    except ValueError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    logger.info(
        "chat completion request provider=%s model=%s stream=%s messages=%s",
        provider_id,
        body.get("model"),
        body.get("stream") is True,
        len(body.get("messages") or []),
    )
    if settings.debug:
        logger.debug(
            "incoming chat request headers=%s",
            redact_headers(dict(request.headers)),
        )
        logger.debug(
            "chat completion request body=%s",
            render_debug(body, settings.log_body_chars),
        )

    # Freebuff provider -> giữ nguyên logic cũ (session/run-chain/ads)
    if gateway.is_freebuff(provider_id):
        return await _freebuff_chat(request, body, settings)

    # Generic openai-compatible provider
    if body.get("stream") is True:
        return StreamingResponse(
            _safe_stream(gateway.stream_chat(provider_id, body, real_model=real_model)),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        response = await gateway.chat(provider_id, body, real_model=real_model)
        return JSONResponse(response)
    except Exception as error:
        return error_response(error, request)


async def _freebuff_chat(request: Request, body: dict[str, Any], settings: Any) -> Any:
    """Xử lý chat qua Freebuff provider (giữ logic gốc)."""
    gateway = request.app.state.gateway
    accounts = request.app.state.accounts
    try:
        model_config = request.app.state.model_resolver(body.get("model"))
    except ValueError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})
    model = model_config.id
    messages = normalize_chat_messages(body.get("messages"))
    lease = None
    chat_history = request.app.state.chat_history
    project_key = chat_history.resolve_project(request, body)
    client_source = detect_client(request)
    logger.info(
        "openai client=%s ua=%s x-title=%s",
        client_source,
        request.headers.get("user-agent") or "",
        request.headers.get("x-title") or "",
    )
    _meta = {
        "source": "api",
        "client": client_source,
        "provider": provider_label(request, model),
        "via": "gateway",
    }
    chat_history.record_messages(
        project_key, messages, model=model, meta=_meta
    )
    context = chat_history.build_context(project_key, body.get("messages"))
    injected = False
    if context:
        messages = inject_context(messages, context)
        injected = True
        logger.info("injected chat history context project=%s chars=%s", project_key, len(context))
    try:
        lease, client, run, payload = await prepare_freebuff_dispatch(
            accounts, model_config, body, messages, settings
        )
        trace_session_id = str(uuid.uuid4())
        if settings.debug:
            logger.debug(
                "prepared upstream freebuff chat trace=%s run=%s payload=%s",
                trace_session_id,
                run,
                render_debug(payload, settings.log_body_chars),
            )
    except Exception as error:
        if lease is not None:
            await lease.aclose()
        logger.warning(
            "failed to prepare freebuff chat: %s",
            error,
            exc_info=settings.debug,
        )
        return error_response(error, request)

    recover_payload = build_session_recover_callback(
        lease,
        model_config,
        body,
        messages,
        settings,
        run,
    )

    def _on_assistant(parts: dict) -> None:
        chat_history.record(
            project_key,
            role="assistant",
            content=parts.get("text") or "",
            model=model,
            thinking=parts.get("thinking") or "",
            tool_calls=parts.get("tool_calls"),
            meta=_meta,
        )

    has_tools = bool(body.get("tools"))
    contribution_reporter = _contribution_reporter(request)
    logger.info(
        "openai route decision stream=%s tools=%s -> %s",
        body.get("stream") is True,
        has_tools,
        "tool-loop" if has_tools else "plain-stream",
    )

    if has_tools:
        # Native tool passthrough: ONE upstream pass; if the model emits a tool
        # call, stream it as a native tool_calls/tool_use delta so the client
        # (Claude Code / Cline) shows its own approval UI and runs the tool on
        # the host, then sends the result back on the next request.
        if body.get("stream") is True:
            return StreamingResponse(
                stream_tool_agent_loop(
                    client,
                    payload,
                    settings=settings,
                    model=model,
                    recover=recover_payload,
                    debug=settings.debug,
                    log_body_chars=settings.log_body_chars,
                    client_tools=body.get("tools"),
                    on_contribution=contribution_reporter,
                    account_lease=lease,
                    on_assistant=_on_assistant,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-History-Project": project_key,
                    "X-History-Injected": "1" if injected else "0",
                },
            )
        try:
            response, _client_call = await run_tool_loop_pass(
                client,
                payload,
                settings=settings,
                model=model,
                recover=recover_payload,
                debug=settings.debug,
                log_body_chars=settings.log_body_chars,
                client_tools=body.get("tools"),
                on_contribution=contribution_reporter,
            )
        except Exception as error:
            if (
                lease is not None
                and isinstance(error, CodebuffError)
                and error.status_code in {401, 403, 429}
            ):
                lease.mark_rate_limited(settings.account_cooldown, error=error)
            return error_response(error, request)
        finally:
            if lease is not None:
                await lease.aclose()
        assert response is not None
        message = (response.get("choices") or [{}])[0].get("message") or {}
        _on_assistant(
            {
                "text": message.get("content") or "",
                "thinking": message.get("reasoning_content") or "",
                "tool_calls": message.get("tool_calls") or [],
            }
        )
        return JSONResponse(
            response,
            headers={
                "X-History-Project": project_key,
                "X-History-Injected": "1" if injected else "0",
            },
        )

    if body.get("stream") is True:
        return StreamingResponse(
            stream_openai_chunks(
                client,
                payload,
                run,
                debug=settings.debug,
                log_body_chars=settings.log_body_chars,
                account_lease=lease,
                on_rate_limited=lambda error: lease.mark_rate_limited(
                    settings.account_cooldown, error=error
                ),
                recover=recover_payload,
                on_assistant=_on_assistant,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-History-Project": project_key,
                "X-History-Injected": "1" if injected else "0",
            },
        )

    try:
        response = await collect_completion(
            client,
            payload,
            run,
            model,
            debug=settings.debug,
            log_body_chars=settings.log_body_chars,
            recover=recover_payload,
        )
        message = (response.get("choices") or [{}])[0].get("message") or {}
        _on_assistant(
            {
                "text": message.get("content") or "",
                "thinking": message.get("reasoning_content") or "",
                "tool_calls": message.get("tool_calls") or [],
            }
        )
        return JSONResponse(
            response,
            headers={
                "X-History-Project": project_key,
                "X-History-Injected": "1" if injected else "0",
            },
        )
    except Exception as error:
        if (
            lease is not None
            and isinstance(error, CodebuffError)
            and error.status_code in {401, 403, 429}
        ):
            lease.mark_rate_limited(settings.account_cooldown, error=error)
        return error_response(error, request)
    finally:
        if lease is not None:
            await lease.aclose()
