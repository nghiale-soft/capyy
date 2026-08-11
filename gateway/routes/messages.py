from __future__ import annotations

import asyncio
import json
import logging
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
from ..services.chat_service import (
    _accumulate_assistant_parts,
    _log_assistant_marker_leak,
    _maybe_record_assistant,
    build_payload,
    build_session_recover_callback,
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


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    _: None = Depends(check_local_auth),
) -> Any:
    anthropic_body = await request.json()
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
    }
    chat_history.record_messages(
        project_key, messages, model=model, meta=_meta
    )
    context = chat_history.build_context(project_key, anthropic_body.get("messages"))
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
                    on_rate_limited=lambda: lease.mark_rate_limited(
                        settings.account_cooldown
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
            response, client_call = await run_tool_loop_pass(
                client,
                payload,
                settings=settings,
                model=model,
                recover=recover_payload,
                debug=settings.debug,
                log_body_chars=settings.log_body_chars,
                client_tools=body.get("tools"),
            )
        except CodebuffError as error:
            if lease is not None and error.status_code in {401, 403, 429}:
                lease.mark_rate_limited(settings.account_cooldown)
            return JSONResponse(
                status_code=error.status_code,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(error)},
                },
            )
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
                on_rate_limited=lambda: lease.mark_rate_limited(
                    settings.account_cooldown
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
            lease.mark_rate_limited(settings.account_cooldown)
        return JSONResponse(
            status_code=error.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    finally:
        if lease is not None:
            await lease.aclose()


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
        client_source = detect_client(request)
        meta = {
            "source": "claude",
            "client": client_source,
            "provider": provider_id,
            "via": "gateway",
        }
        chat_history.record_messages(project_key, messages, model=model, meta=meta)
        context = chat_history.build_context(project_key, anthropic_body.get("messages"))
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
    on_assistant: Any | None = None,
):
    message_id: str | None = None
    state = AnthropicStreamState(model=requested_model)
    assistant_state = new_assistant_state()

    try:
        async for line in chat_events_with_recovery(
            client,
            payload,
            recover=recover,
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

        logger.info("anthropic stream phase=completed run_id=%s message_id=%s", run.run_id, message_id)

    except asyncio.CancelledError:
        logger.warning("anthropic stream phase=client_disconnected run_id=%s", run.run_id)
        raise

    except CodebuffError as error:
        logger.warning("anthropic stream failed run_id=%s: %s", run.run_id, error, exc_info=debug)
        if error.status_code in {401, 403, 429} and on_rate_limited is not None:
            on_rate_limited()
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    except Exception as error:
        logger.exception("anthropic stream failed run_id=%s", run.run_id)
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
    finally:
        logger.info("anthropic stream phase=finalize run_id=%s message_id=%s", run.run_id, message_id)
        schedule_finalize_run(client, run, message_id)
        if account_lease is not None:
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
):
    """Run ONE native tool pass and stream Anthropic SSE to the client.

    - Emits ``message_start`` immediately and ``ping`` heartbeats while the
      upstream pass runs, so Claude Code never sees a silent connection.
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
        )
    )
    logger.info("anthropic tool-pass phase=started model=%s", model)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=STREAM_PING_SECONDS)
            if done:
                break
            # Heartbeat so Claude Code / proxies keep the connection open while
            # the upstream pass (model thinking) is still running.
            yield encode_anthropic_sse("ping", {"type": "ping"})
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
        if error.status_code in {401, 403, 429} and on_rate_limited is not None:
            on_rate_limited()
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
        yield encode_anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            },
        )
        return
    finally:
        # If the client disconnected mid-stream (generator cancelled), stop the
        # upstream pass so the account/lease is freed and the run is finalized.
        if not task.done():
            task.cancel()
        if account_lease is not None:
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
    if message.get("reasoning_content"):
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
