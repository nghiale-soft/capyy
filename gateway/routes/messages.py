from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..compat.anthropic import (
    AnthropicStreamState,
    anthropic_request_to_openai,
    encode_anthropic_sse,
    finish_anthropic_stream,
    openai_chunk_to_anthropic_events,
    openai_response_to_anthropic,
)
from ..compat.openai import normalize_chat_messages, sanitize_stream_chunk
from ..core.logging import log_curl, redact_headers, render_debug
from ..core.sse import decode_sse_data
from ..deps import check_local_auth, detect_client, get_settings, provider_label
from ..services.chat_history import inject_context
from ..failover import FreebuffDispatchFailover, has_next_provider, should_fallback_to_next_provider
from ..services.chat_service import (
    _accumulate_assistant_parts,
    _log_assistant_marker_leak,
    _maybe_record_assistant,
    build_payload,
    chat_events_with_recovery,
    collect_completion,
    new_assistant_state,
    prepare_freebuff_dispatch,
    run_tool_loop_pass,
    schedule_finalize_run,
    start_freebuff_run_chain,
)
from providers.openai_compatible import GatewayProviderError
from providers.freebuff import CodebuffError

# Anthropic streaming protocol allows periodic `ping` events so the client knows
# the connection is alive while the gateway waits on a long tool pass / tool
# approval (Claude Code / Cline show their own approval UI). 15s keeps well
# inside any proxy idle timeout while adding negligible overhead.
STREAM_PING_SECONDS = 15.0
router = APIRouter()
logger = logging.getLogger("gateway.routes.messages")


# --- Reusable helpers for emitting user-facing status/error notifications ---


def _make_status_events(
    state: Any,
    requested_model: str,
    kind: str,
    text: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Create Anthropic SSE content events showing a status message to the user.

    Used by both ``_stream_tool_loop_anthropic`` and ``_stream_anthropic_chunks``
    to surface human-readable error/status messages (e.g. session expired,
    rate limited) directly in the chat thread.
    """
    delta = (
        {"reasoning_content": text}
        if kind == "reasoning"
        else {"content": text}
    )
    progress_chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return openai_chunk_to_anthropic_events(progress_chunk, state)


def _sse_status(state: Any, requested_model: str, text: str) -> list[bytes]:
    """Shortcut: return already-encoded SSE bytes for a status message."""
    return [
        encode_anthropic_sse(ev, ev_data)
        for ev, ev_data in _make_status_events(state, requested_model, "status", text)
    ]


# Map HTTP status codes to human-readable error labels for the user.
_ERROR_LABELS: dict[int, str] = {
    428: "⚠️ Freebuff session expired — recovery failed, retrying may help",
    401: "⚠️ Authentication failed — provider credentials invalid",
    403: "⚠️ Access denied — provider rejected the request",
    429: "⚠️ Rate limited — all provider accounts exhausted",
    500: "⚠️ Provider internal error — please retry",
    502: "⚠️ Provider unavailable — upstream returned bad gateway",
    503: "⚠️ Provider overloaded — service temporarily unavailable",
}

_TOOL_RESULT_SECRET_RE = re.compile(
    r"(?i)\b(?:bearer\s+|sk-[a-z0-9_-]{8,}|api[_ -]?key\s*[:=]\s*)[^\s,;)}\]]+"
)


def _tool_result_text(content: Any) -> str:
    """Return text-only tool-result content without serializing images/blobs."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _log_client_tool_results(body: dict[str, Any]) -> None:
    """Log a compact, secret-redacted client tool result for support tracing.

    Claude runs tools locally.  The gateway previously logged the generated
    tool call but not its returned result, so client-side statuses such as
    ``Wasted call`` could not be diagnosed.  Never log image data or a full
    source-file result; the preview is redacted and capped.
    """
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return
    tool_names: dict[str, str] = {}
    results: list[tuple[str, bool, str]] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_id = block.get("id")
                tool_name = block.get("name")
                if isinstance(tool_id, str) and isinstance(tool_name, str):
                    tool_names[tool_id] = tool_name
            elif block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                if isinstance(tool_id, str):
                    results.append((tool_id, block.get("is_error") is True, _tool_result_text(block.get("content"))))
    for tool_id, is_error, text in results[-8:]:
        redacted = _TOOL_RESULT_SECRET_RE.sub("<redacted>", text)
        preview = re.sub(r"\s+", " ", redacted).strip()[:240]
        logger.info(
            "anthropic client_tool_result tool=%s id=%s error=%s chars=%s preview=%r",
            tool_names.get(tool_id, "unknown"),
            tool_id[:16],
            is_error,
            len(text),
            preview,
        )


def _contribution_reporter(request: Request) -> Any:
    def report(kind: str, title: str, summary: str, metadata: dict[str, str]) -> None:
        try:
            request.app.state.contributions.add(
                kind,
                title,
                summary,
                {**metadata, "client": detect_client(request), "route": "anthropic"},
            )
        except Exception:
            logger.exception("failed to persist contribution kind=%s", kind)
    return report


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    _: None = Depends(check_local_auth),
) -> Any:
    anthropic_body = await request.json()
    _log_client_tool_results(anthropic_body)
    body = anthropic_request_to_openai(anthropic_body)
    log_curl(request, anthropic_body, logger=logger)
    settings = get_settings(request)
    gateway = request.app.state.gateway
    accounts = request.app.state.accounts

    # Unlike the original FreeBuff-only route, Messages requests must use the
    # same registry/priority decision as OpenAI requests.  Native OpenAI
    # compatible providers receive the client's tool schema untouched; only
    # FreeBuff needs the text-protocol adapter below because its upstream
    # rejects a ``tools`` field.
    try:
        provider_id, real_model = gateway.resolve(body.get("model"))
    except ValueError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    if not gateway.is_freebuff(provider_id):
        return await _generic_anthropic_messages(
            request,
            anthropic_body=anthropic_body,
            body=body,
            provider_id=provider_id,
            real_model=real_model,
        )

    try:
        model_config = request.app.state.model_resolver(body.get("model"))
    except ValueError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})
    model = model_config.id
    logger.info(
        "anthropic request phase=received request_id=%s model=%s stream=%s messages=%s tools=%s",
        request.headers.get("x-request-id") or uuid.uuid4().hex[:12],
        model,
        body.get("stream") is True,
        len(body.get("messages") or []),
        len(body.get("tools") or []),
    )

    if settings.debug:
        logger.debug(
            "incoming anthropic request headers=%s",
            redact_headers(dict(request.headers)),
        )
        logger.debug(
            "anthropic request body=%s",
            render_debug(anthropic_body, settings.log_body_chars),
        )
        logger.debug(
            "converted OpenAI request body=%s",
            render_debug(body, settings.log_body_chars),
        )

    messages = normalize_chat_messages(body.get("messages"))
    lease = None
    chat_history = request.app.state.chat_history
    project_key = chat_history.resolve_project(request, anthropic_body)
    session_id = getattr(chat_history, "resolve_session", lambda *_: "gateway")(request, anthropic_body)
    client_source = detect_client(request)
    logger.info(
        "anthropic client=%s ua=%s x-app=%s",
        client_source,
        request.headers.get("user-agent") or "",
        request.headers.get("x-app") or "",
    )
    _meta = {
        "source": "claude",
        "client": client_source,
        "provider": provider_label(request, model),
        "via": "gateway",
        "session_id": session_id,
    }
    chat_history.record_messages(
        project_key, messages, model=model, meta=_meta
    )
    context = chat_history.build_context(project_key, anthropic_body.get("messages"), session_id=session_id)
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
                "prepared upstream anthropic trace=%s run=%s payload=%s",
                trace_session_id,
                run,
                render_debug(payload, settings.log_body_chars),
            )
    except CodebuffError as error:
        if lease is not None:
            await lease.aclose()
        # Session/auth/quota failures happen before an Anthropic SSE response
        # begins, so it is still safe to continue down the registry priority
        # chain. GatewayService skips the failed FreeBuff adapter and tries the
        # next OpenAI-compatible provider.
        if error.status_code in {401, 403, 429}:
            if not has_next_provider(gateway):
                _contribution_reporter(request)(
                    "provider",
                    "No fallback provider after Freebuff failure",
                    "Freebuff failed and no enabled lower-priority provider is configured.",
                    {"provider": "freebuff", "status_code": str(error.status_code), "fallback_available": "false"},
                )
                return JSONResponse(
                    status_code=error.status_code,
                    content={"type": "error", "error": {"type": "api_error", "message": str(error)}},
                )
            logger.warning(
                "freebuff preparation failed status=%s; trying next provider",
                error.status_code,
            )
            return await _generic_anthropic_messages(
                request,
                anthropic_body=anthropic_body,
                body=body,
                provider_id="freebuff",
                real_model=body.get("model"),
                history_context=(project_key, messages, _meta, injected),
            )
        logger.warning("failed to prepare anthropic request: %s", error, exc_info=settings.debug)
        _contribution_reporter(request)(
            "provider",
            "Freebuff request preparation failed",
            "Freebuff could not prepare a request after its normal retry and failover handling.",
            {"provider": "freebuff", "status_code": str(error.status_code)},
        )
        return JSONResponse(
            status_code=error.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    except Exception as error:
        if lease is not None:
            await lease.aclose()
        logger.exception("failed to prepare anthropic request")
        raise error

    dispatch_failover = FreebuffDispatchFailover(
        accounts=accounts,
        model_config=model_config,
        body=body,
        messages=messages,
        settings=settings,
        lease=lease,
        client=client,
        run=run,
        payload=payload,
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
        "anthropic route decision stream=%s tools=%s -> %s",
        body.get("stream") is True,
        has_tools,
        "tool-loop" if has_tools else "plain-stream",
    )

    if has_tools:
        # Native tool passthrough: run ONE upstream pass; if the model emits a
        # tool call, stream it as a native `tool_use` block so the client
        # (Claude Code / Cline) shows its own approval UI and runs the tool on
        # the host. Upstream free models reject `tools`, so the gateway keeps
        # the text-protocol prompt and strips markers before streaming.
        if body.get("stream") is True:
            return StreamingResponse(
                _stream_tool_loop_anthropic(
                    client,
                    payload,
                    body=body,
                    settings=settings,
                    model=model,
                    requested_model=anthropic_body.get("model") or model,
                    account_lease=lease,
                    on_rate_limited=lambda error: lease.mark_rate_limited(
                        settings.account_cooldown, error=error
                    ),
                    recover=dispatch_failover.recover_session,
                    on_assistant=_on_assistant,
                    on_contribution=contribution_reporter,
                    history_executor=lambda call: chat_history.execute_history_tool(call, project_key),
                    dispatch_failover=dispatch_failover,
                    fallback_stream=(
                        lambda: gateway.stream_chat(
                            "freebuff", {**body, "messages": messages}, real_model=body.get("model")
                        )
                        if has_next_provider(gateway)
                        else None
                    ),
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
            response, client_call = await run_tool_loop_pass(
                client,
                payload,
                settings=settings,
                model=model,
                recover=dispatch_failover.recover_session,
                debug=settings.debug,
                log_body_chars=settings.log_body_chars,
                client_tools=body.get("tools"),
                on_contribution=contribution_reporter,
                history_executor=lambda call: chat_history.execute_history_tool(call, project_key),
                dispatch_failover=dispatch_failover,
            )
        except CodebuffError as error:
            if lease is not None and error.status_code in {401, 403, 429}:
                lease.mark_rate_limited(settings.account_cooldown, error=error)
            if should_fallback_to_next_provider(gateway, error):
                logger.warning("freebuff token pool unavailable; trying next provider")
                return await _generic_anthropic_messages(
                    request,
                    anthropic_body=anthropic_body,
                    body=body,
                    provider_id="freebuff",
                    real_model=body.get("model"),
                    history_context=(project_key, messages, _meta, injected),
                )
            contribution_reporter(
                "provider", "Freebuff tool pass failed",
                "Freebuff could not complete a native client tool pass.",
                {"provider": "freebuff", "status_code": str(error.status_code)},
            )
            return JSONResponse(
                status_code=error.status_code,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(error)},
                },
            )
        finally:
            await dispatch_failover.aclose()
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
            openai_response_to_anthropic(
                response,
                anthropic_body.get("model") or model,
            ),
            headers={
                "X-History-Project": project_key,
                "X-History-Injected": "1" if injected else "0",
            },
        )

    if body.get("stream") is True:
        # FreeBuff models can emit Claude XML/DSML tool markers even after a
        # client omits ``tools`` on a follow-up turn. Route every Anthropic
        # stream through the native tool-pass adapter so those markers become
        # Anthropic ``tool_use`` events rather than corrupting the SSE stream.
        return StreamingResponse(
            _stream_tool_loop_anthropic(
                client,
                payload,
                body=body,
                settings=settings,
                model=model,
                requested_model=anthropic_body.get("model") or model,
                account_lease=lease,
                on_rate_limited=lambda error: lease.mark_rate_limited(
                    settings.account_cooldown, error=error
                ),
                recover=dispatch_failover.recover_session,
                on_assistant=_on_assistant,
                on_contribution=contribution_reporter,
                history_executor=lambda call: chat_history.execute_history_tool(call, project_key),
                dispatch_failover=dispatch_failover,
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
            recover=dispatch_failover.recover_session,
            dispatch_failover=dispatch_failover,
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
            openai_response_to_anthropic(
                response,
                anthropic_body.get("model") or model,
            ),
            headers={
                "X-History-Project": project_key,
                "X-History-Injected": "1" if injected else "0",
            },
        )
    except CodebuffError as error:
        if lease is not None and error.status_code in {401, 403, 429}:
            lease.mark_rate_limited(settings.account_cooldown, error=error)
        if should_fallback_to_next_provider(gateway, error):
            logger.warning("freebuff token pool unavailable; trying next provider")
            return await _generic_anthropic_messages(
                request,
                anthropic_body=anthropic_body,
                body=body,
                provider_id="freebuff",
                real_model=body.get("model"),
                history_context=(project_key, messages, _meta, injected),
            )
        contribution_reporter(
            "provider", "Freebuff completion failed",
            "Freebuff could not complete a non-streaming response.",
            {"provider": "freebuff", "status_code": str(error.status_code)},
        )
        return JSONResponse(
            status_code=error.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    finally:
        await dispatch_failover.aclose()


async def _generic_anthropic_messages(
    request: Request,
    *,
    anthropic_body: dict[str, Any],
    body: dict[str, Any],
    provider_id: str,
    real_model: str | None,
    history_context: tuple[str, list[dict[str, Any]], dict[str, Any], bool] | None = None,
) -> Any:
    """Serve Anthropic Messages through an OpenAI-compatible provider.

    The provider registry performs priority-order failover before the first
    response chunk.  Tool calls remain native: the client owns approval and
    execution, while this gateway only converts protocol shapes.
    """
    gateway = request.app.state.gateway
    chat_history = request.app.state.chat_history
    requested_model = anthropic_body.get("model") or body.get("model") or "unknown"
    model = real_model or requested_model
    if history_context is not None:
        project_key, messages, meta, injected = history_context
    else:
        messages = list(body.get("messages") or [])
        project_key = chat_history.resolve_project(request, anthropic_body)
        session_id = getattr(chat_history, "resolve_session", lambda *_: "gateway")(request, anthropic_body)
        client_source = detect_client(request)
        meta = {
            "source": "claude",
            "client": client_source,
            "provider": provider_id,
            "via": "gateway",
            "session_id": session_id,
        }
        chat_history.record_messages(project_key, messages, model=model, meta=meta)
        context = chat_history.build_context(project_key, anthropic_body.get("messages"), session_id=session_id)
        injected = False
        if context:
            messages = inject_context(messages, context)
            injected = True

    payload = {**body, "messages": messages}
    headers = {
        "X-History-Project": project_key,
        "X-History-Injected": "1" if injected else "0",
    }

    def _on_assistant(parts: dict[str, Any]) -> None:
        chat_history.record(
            project_key,
            role="assistant",
            content=parts.get("text") or "",
            model=model,
            thinking=parts.get("thinking") or "",
            tool_calls=parts.get("tool_calls"),
            meta=meta,
        )

    if body.get("stream") is True:
        return StreamingResponse(
            _stream_generic_anthropic(
                gateway,
                provider_id,
                payload,
                real_model=real_model,
                requested_model=requested_model,
                on_assistant=_on_assistant,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                **headers,
            },
        )

    try:
        response = await gateway.chat(provider_id, payload, real_model=real_model)
    except GatewayProviderError as error:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    message = (response.get("choices") or [{}])[0].get("message") or {}
    _on_assistant(
        {
            "text": message.get("content") or "",
            "thinking": message.get("reasoning_content") or "",
            "tool_calls": message.get("tool_calls") or [],
        }
    )
    return JSONResponse(openai_response_to_anthropic(response, requested_model), headers=headers)


async def _stream_generic_anthropic(
    gateway: Any,
    provider_id: str,
    payload: dict[str, Any],
    *,
    real_model: str | None,
    requested_model: str,
    on_assistant: Any | None = None,
):
    """Convert a generic provider's OpenAI SSE stream to Anthropic SSE."""
    state = AnthropicStreamState(model=requested_model)
    assistant_state = new_assistant_state()
    try:
        async for line in gateway.stream_chat(provider_id, payload, real_model=real_model):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                for event, event_data in finish_anthropic_stream(state):
                    yield encode_anthropic_sse(event, event_data)
                break
            chunk = sanitize_stream_chunk(data)
            if chunk is None:
                continue
            _accumulate_assistant_parts(chunk, assistant_state)
            for event, event_data in openai_chunk_to_anthropic_events(chunk, state):
                yield encode_anthropic_sse(event, event_data)
        _maybe_record_assistant(on_assistant, assistant_state)
    except GatewayProviderError as error:
        yield encode_anthropic_sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(error)}},
        )
    except Exception as error:
        logger.exception("generic anthropic stream failed provider=%s", provider_id)
        yield encode_anthropic_sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(error)}},
        )


async def _stream_anthropic_chunks(
    client,
    payload: dict[str, Any],
    run,
    *,
    requested_model: str,
    debug: bool = False,
    account_lease: Any | None = None,
    on_rate_limited: Any = None,
    recover: Any | None = None,
    dispatch_failover: FreebuffDispatchFailover | None = None,
    on_assistant: Any | None = None,
):
    message_id: str | None = None
    state = AnthropicStreamState(model=requested_model)
    assistant_state = new_assistant_state()

    try:
        async for line in chat_events_with_recovery(
            client,
            payload,
            recover=dispatch_failover.recover_session if dispatch_failover else recover,
            failover=dispatch_failover.failover if dispatch_failover else None,
            debug=debug,
        ):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                for event, event_data in finish_anthropic_stream(state):
                    yield encode_anthropic_sse(event, event_data)
                break

            message_id = data.get("id") or message_id
            chunk = sanitize_stream_chunk(data)
            if chunk is None:
                continue
            _accumulate_assistant_parts(chunk, assistant_state)

            for event, event_data in openai_chunk_to_anthropic_events(chunk, state):
                yield encode_anthropic_sse(event, event_data)
        _log_assistant_marker_leak(assistant_state, "anthropic-stream")
        _maybe_record_assistant(on_assistant, assistant_state)

        active_run = dispatch_failover.run if dispatch_failover else run
        logger.info("anthropic stream phase=completed run_id=%s message_id=%s", active_run.run_id, message_id)

    except asyncio.CancelledError:
        logger.warning("anthropic stream phase=client_disconnected run_id=%s", run.run_id)
        raise

    except CodebuffError as error:
        logger.warning("anthropic stream failed run_id=%s: %s", run.run_id, error, exc_info=debug)
        _label = _ERROR_LABELS.get(error.status_code, f"⚠️ Provider error (HTTP {error.status_code})")
        for chunk in _sse_status(state, requested_model, _label):
            yield chunk
        if error.status_code in {401, 403, 429} and on_rate_limited is not None:
            on_rate_limited(error)
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    except Exception as error:
        logger.exception("anthropic stream failed run_id=%s", run.run_id)
        for chunk in _sse_status(state, requested_model, f"⚠️ Unexpected gateway error — {type(error).__name__}: {error}"):
            yield chunk
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    finally:
        active_client = dispatch_failover.client if dispatch_failover else client
        active_run = dispatch_failover.run if dispatch_failover else run
        logger.info("anthropic stream phase=finalize run_id=%s message_id=%s", active_run.run_id, message_id)
        schedule_finalize_run(active_client, active_run, message_id)
        if dispatch_failover is not None:
            await dispatch_failover.aclose()
        elif account_lease is not None:
            await account_lease.aclose()


async def _stream_tool_loop_anthropic(
    client,
    payload: dict[str, Any],
    *,
    body: dict[str, Any],
    settings: Any,
    model: str,
    requested_model: str,
    account_lease: Any | None = None,
    on_rate_limited: Any | None = None,
    recover: Any | None = None,
    on_assistant: Any | None = None,
    on_contribution: Any | None = None,
    history_executor: Any | None = None,
    dispatch_failover: FreebuffDispatchFailover | None = None,
    fallback_stream: Any | None = None,
):
    """Run ONE native tool pass and stream Anthropic SSE to the client.

    - Emits ``message_start`` immediately, then forwards neutral gateway phases
      and actual upstream reasoning into a collapsible Thinking block while the
      upstream pass runs.  Those updates are never emitted as assistant text,
      so they cannot leak into later conversation history.
    - If the model emitted a tool call, streams a native ``tool_use`` block and
      ends with ``stop_reason: tool_use`` — the client shows its own approval
      UI, runs the tool on the host and returns the result on the next request.
    - Otherwise streams the final answer normally.
    """
    state = AnthropicStreamState(model=requested_model)

    # message_start immediately (even an empty chunk triggers it), so the client
    # knows the request is alive before the upstream pass finishes.
    for event, event_data in openai_chunk_to_anthropic_events(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        },
        state,
    ):
        logger.info("anthropic sse phase=emit event=%s block=%s", event, event_data.get("type"))
        yield encode_anthropic_sse(event, event_data)

    progress: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    def progress_events(kind: str, update: str) -> list[tuple[str, dict[str, Any]]]:
        delta = (
            {"reasoning_content": update}
            if kind == "reasoning"
            else {"content": update}
        )
        progress_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }],
        }
        return openai_chunk_to_anthropic_events(progress_chunk, state)

    task = asyncio.create_task(
        run_tool_loop_pass(
            client,
            payload,
            settings=settings,
            model=model,
            recover=recover,
            debug=settings.debug,
            log_body_chars=settings.log_body_chars,
            client_tools=body.get("tools"),
            on_contribution=on_contribution,
            history_executor=history_executor,
            on_progress=lambda kind, text: progress.put_nowait((kind, text)),
            dispatch_failover=dispatch_failover,
        )
    )
    logger.info("anthropic tool-pass phase=started model=%s", model)
    streamed_thinking = False
    last_ping_at = time.monotonic()
    progress_wait = asyncio.create_task(progress.get())
    try:
        while True:
            remaining_ping = max(0.0, STREAM_PING_SECONDS - (time.monotonic() - last_ping_at))
            done, _ = await asyncio.wait(
                {task, progress_wait},
                timeout=remaining_ping,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if progress_wait in done:
                kind, update = progress_wait.result()
                if kind == "status":
                    logger.info("anthropic sse phase=gateway_status text=%s", update.strip())
                for event, event_data in progress_events(kind, update):
                    yield encode_anthropic_sse(event, event_data)
                streamed_thinking = streamed_thinking or kind == "reasoning"
                progress_wait = asyncio.create_task(progress.get())
            if done:
                if task in done:
                    break
            # Heartbeat so Claude Code / proxies keep the connection open while
            # the upstream pass (model thinking) is still running.
            if time.monotonic() - last_ping_at >= STREAM_PING_SECONDS:
                yield encode_anthropic_sse("ping", {"type": "ping"})
                last_ping_at = time.monotonic()
        if not progress_wait.done():
            progress_wait.cancel()
        while not progress.empty():
            kind, update = progress.get_nowait()
            if kind == "status":
                logger.info("anthropic sse phase=gateway_status text=%s", update.strip())
            for event, event_data in progress_events(kind, update):
                yield encode_anthropic_sse(event, event_data)
            streamed_thinking = streamed_thinking or kind == "reasoning"
        response, client_call = task.result()
        logger.info(
            "anthropic tool-pass phase=upstream_completed tool=%s",
            client_call.get("name") if client_call else None,
        )
    except asyncio.CancelledError:
        logger.warning("anthropic tool-pass phase=client_disconnected task_done=%s", task.done())
        raise
    except CodebuffError as error:
        logger.warning("anthropic tool pass failed: %s", error, exc_info=settings.debug)
        _error_labels = {
            428: "⚠️ Freebuff session expired — recovery failed, retrying may help",
            401: "⚠️ Authentication failed — provider credentials invalid",
            403: "⚠️ Access denied — provider rejected the request",
            429: "⚠️ Rate limited — all provider accounts exhausted",
            500: "⚠️ Provider internal error — please retry",
            502: "⚠️ Provider unavailable — upstream returned bad gateway",
            503: "⚠️ Provider overloaded — service temporarily unavailable",
        }
        _label = _error_labels.get(error.status_code, f"⚠️ Provider error (HTTP {error.status_code})")
        for _ev, _ev_data in progress_events("status", _label):
            yield encode_anthropic_sse(_ev, _ev_data)
        if (
            error.status_code in {401, 403, 429}
            and fallback_stream is not None
        ):
            logger.warning("freebuff token pool unavailable; trying next provider")
            for _ev, _ev_data in progress_events("status", "🔄 Trying fallback provider…"):
                yield encode_anthropic_sse(_ev, _ev_data)
            async for line in fallback_stream():
                data = decode_sse_data(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    for event, event_data in finish_anthropic_stream(state):
                        yield encode_anthropic_sse(event, event_data)
                    return
                chunk = sanitize_stream_chunk(data)
                if chunk is None:
                    continue
                for event, event_data in openai_chunk_to_anthropic_events(chunk, state):
                    yield encode_anthropic_sse(event, event_data)
            for event, event_data in finish_anthropic_stream(state):
                yield encode_anthropic_sse(event, event_data)
            return
        if (
            error.status_code in {401, 403, 429}
            and dispatch_failover is None
            and on_rate_limited is not None
        ):
            on_rate_limited(error)
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
        return
    except Exception as error:
        logger.exception("anthropic tool pass failed")
        for _ev, _ev_data in progress_events("status", f"⚠️ Unexpected gateway error — {type(error).__name__}: {error}"):
            yield encode_anthropic_sse(_ev, _ev_data)
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
        return
    finally:
        if not progress_wait.done():
            progress_wait.cancel()
        # If the client disconnected mid-stream (generator cancelled), stop the
        # upstream pass so the account/lease is freed and the run is finalized.
        if not task.done():
            task.cancel()
        if dispatch_failover is not None:
            await dispatch_failover.aclose()
        elif account_lease is not None:
            await account_lease.aclose()
        logger.info("anthropic tool-pass phase=lease_released task_done=%s", task.done())

    assert response is not None
    message = (response.get("choices") or [{}])[0].get("message") or {}
    finish_reason = (response.get("choices") or [{}])[0].get("finish_reason") or "stop"
    logger.info(
        "anthropic tool-pass phase=upstream_response id=%s finish_reason=%s content_chars=%s thinking_chars=%s tool_calls=%s",
        response.get("id"), finish_reason, len(message.get("content") or ""),
        len(message.get("reasoning_content") or ""), len(message.get("tool_calls") or []),
    )
    if on_assistant is not None:
        try:
            on_assistant(
                {
                    "text": message.get("content") or "",
                    "thinking": message.get("reasoning_content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                }
            )
        except Exception:
            logger.exception("failed to record anthropic tool pass assistant")

    delta: dict[str, Any] = {}
    # Every reasoning chunk has already been sent in real time above.  Replaying
    # the accumulated field here would duplicate the complete Thought block.
    if message.get("reasoning_content") and not streamed_thinking:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("content"):
        delta["content"] = message["content"]
    if client_call is not None:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": client_call["name"],
                    "arguments": json.dumps(
                        client_call.get("arguments") or {},
                        ensure_ascii=False,
                    ),
                },
            }
        ]
    chunk = {
        "id": response.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": response.get("created") or int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": (
                    "tool_calls"
                    if client_call is not None
                    else (response.get("choices") or [{}])[0].get("finish_reason")
                    or "stop"
                ),
            }
        ],
    }
    for event, event_data in openai_chunk_to_anthropic_events(chunk, state):
        logger.info(
            "anthropic sse phase=emit event=%s stop_reason=%s block_type=%s",
            event,
            event_data.get("delta", {}).get("stop_reason"),
            event_data.get("content_block", {}).get("type"),
        )
        yield encode_anthropic_sse(event, event_data)
    for event, event_data in finish_anthropic_stream(state):
        logger.info("anthropic sse phase=emit event=%s", event)
        yield encode_anthropic_sse(event, event_data)
