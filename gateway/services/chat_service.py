from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator

from providers.freebuff import (
    CodebuffClient,
    CodebuffError,
    FreebuffRun,
    is_waiting_room_required,
    utc_now_iso,
)
from gateway.compat.models import CONTEXT_PRUNER_AGENT_ID, FreebuffModel
from gateway.compat.openai import (
    CompletionAccumulator,
    build_upstream_payload,
    normalize_chat_messages,
    sanitize_stream_chunk,
)
from gateway.core.logging import render_debug
from gateway.core.sse import decode_sse_data, encode_sse
from gateway.services.toolkit import (
    adapt_client_tool_call,
    client_tool_call,
    coerce_client_tool_call_arguments,
    declared_client_tool_names,
    detect_tool_markers,
    execute_tool,
    parse_tool_call,
    parse_compiler_protocol,
    tool_system_prompt,
    validate_client_tool_call,
)


logger = logging.getLogger("gateway.services.chat")

_RAW_LOG_SECRET_RE = re.compile(
    r"(?:figd|sk|sk-proj|ghp|github_pat|xox[bpras])[-_][A-Za-z0-9._-]+",
    re.IGNORECASE,
)


def _redact_raw_model_output(value: str) -> str:
    """Keep diagnostic text complete while never persisting credential values."""
    return _RAW_LOG_SECRET_RE.sub("<redacted-secret>", value)


async def start_freebuff_run_chain(
    client: CodebuffClient,
    model: FreebuffModel | str,
) -> FreebuffRun:
    if isinstance(model, str):
        model = FreebuffModel(model, model)
    if model.parent_agent_id:
        return await _start_child_chat_run_chain(client, model)

    agent_id = model.agent_id
    started_at = utc_now_iso()
    run_id = await client.start_run(agent_id)
    child_started_at = utc_now_iso()
    child_run_id = await client.start_run(
        CONTEXT_PRUNER_AGENT_ID,
        ancestor_run_ids=[run_id],
    )
    await client.record_run_step(
        child_run_id,
        step_number=1,
        child_run_ids=[],
        message_id=None,
        start_time=child_started_at,
    )
    await client.finish_run(child_run_id, total_steps=2)
    await client.record_run_step(
        run_id,
        step_number=1,
        child_run_ids=[child_run_id],
        message_id=None,
        start_time=started_at,
    )
    return FreebuffRun(
        run_id=run_id,
        agent_id=agent_id,
        started_at=started_at,
        child_run_id=child_run_id,
    )


async def prepare_freebuff_dispatch(
    accounts: Any,
    model_config: FreebuffModel,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    settings: Any,
) -> tuple[Any, CodebuffClient, FreebuffRun, dict[str, Any]]:
    """Prepare the shared FreeBuff request lifecycle for either API route.

    OpenAI and Anthropic route handlers own only protocol conversion/history.
    Session acquisition, required FreeBuff run setup and upstream payload
    construction have one implementation so their auth/429 behavior cannot
    drift.
    """
    lease = await accounts.acquire_session(model_config.session_id, messages=messages)
    try:
        client = lease.client
        await client.request_ad_chain(messages=messages)
        await client.validate_agents()
        run = await start_freebuff_run_chain(client, model_config)
        payload = build_payload(
            {**body, "messages": messages},
            session=lease.session,
            run=run,
            client_id=settings.client_id,
            upstream_model_id=model_config.upstream_id,
            max_tokens_cap=settings.max_tokens,
        )
        return lease, client, run, payload
    except Exception:
        await lease.aclose()
        raise


async def _start_child_chat_run_chain(
    client: CodebuffClient,
    model: FreebuffModel,
) -> FreebuffRun:
    assert model.parent_agent_id is not None

    started_at = utc_now_iso()
    parent_run_id = await client.start_run(model.parent_agent_id)
    chat_started_at = utc_now_iso()
    chat_run_id = await client.start_run(
        model.agent_id,
        ancestor_run_ids=[parent_run_id],
    )
    return FreebuffRun(
        run_id=parent_run_id,
        agent_id=model.parent_agent_id,
        started_at=started_at,
        child_run_id=chat_run_id,
        chat_run_id=chat_run_id,
        chat_started_at=chat_started_at,
    )


async def finalize_run(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    try:
        logger.debug(
            "finalize run start run_id=%s message_id=%s started_at=%s",
            run.run_id,
            message_id,
            run.started_at,
        )
        if run.chat_run_id and run.chat_run_id != run.run_id:
            await client.record_run_step(
                run.chat_run_id,
                step_number=1,
                child_run_ids=[],
                message_id=message_id,
                start_time=run.chat_started_at or run.started_at,
            )
            await client.finish_run(run.chat_run_id, total_steps=2)
            await client.record_run_step(
                run.run_id,
                step_number=1,
                child_run_ids=[run.chat_run_id],
                message_id=None,
                start_time=run.started_at,
            )
            await client.finish_run(run.run_id, total_steps=2)
            logger.debug("finalize parent/child run done run_id=%s", run.run_id)
            return

        await client.record_run_step(
            run.run_id,
            step_number=2,
            child_run_ids=[],
            message_id=message_id,
            start_time=run.started_at,
        )
        await client.finish_run(run.run_id, total_steps=3)
        logger.debug("finalize run done run_id=%s", run.run_id)
    except CodebuffError as error:
        logger.warning(
            "finalize run failed run_id=%s: %s",
            run.run_id,
            error,
            exc_info=client.settings.debug,
        )
    except Exception:
        logger.exception("finalize run failed run_id=%s", run.run_id)


def schedule_finalize_run(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    task = asyncio.create_task(finalize_run(client, run, message_id))

    def _log_background_error(done: asyncio.Task[None]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            logger.debug("background finalize task cancelled run_id=%s", run.run_id)
        except Exception:
            logger.exception("background finalize task failed run_id=%s", run.run_id)

    task.add_done_callback(_log_background_error)


async def chat_events_with_recovery(
    client: CodebuffClient,
    payload: dict[str, Any],
    *,
    recover: Any | None = None,
    debug: bool = False,
) -> AsyncIterator[str]:
    """Yield upstream chat lines, recovering once from 428 waiting_room_required.

    When upstream reports that the freebuff session is no longer active, the
    `recover` callback (built by the route) refreshes the session and returns a
    rebuilt payload; the stream is then retried once with the fresh session.
    Any other error, or a second 428, is re-raised as-is.
    """
    attempts = 0
    current_payload = payload
    while True:
        try:
            async for line in client.chat_events(current_payload):
                yield line
            return
        except CodebuffError as error:
            if (
                recover is None
                or attempts >= 1
                or not is_waiting_room_required(error)
            ):
                raise
            attempts += 1
            logger.warning(
                "chat upstream waiting_room_required 428; refreshing freebuff "
                "session and retrying: %s",
                error,
                exc_info=debug,
            )
            current_payload = await recover()


async def stream_openai_chunks(
    client: CodebuffClient,
    payload: dict[str, Any],
    run: FreebuffRun,
    *,
    debug: bool = False,
    log_body_chars: int = 2000,
    account_lease: Any | None = None,
    on_rate_limited: Any = None,
    recover: Any | None = None,
    on_assistant: Any | None = None,
) -> AsyncIterator[bytes]:
    message_id: str | None = None
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
                if debug:
                    logger.debug(
                        "chat stream done run_id=%s message_id=%s",
                        run.run_id,
                        message_id,
                    )
                yield encode_sse("[DONE]")
                break

            message_id = data.get("id") or message_id
            chunk = sanitize_stream_chunk(data)
            if chunk is not None:
                _accumulate_assistant_parts(chunk, assistant_state)
                if debug:
                    logger.debug("chat stream chunk=%s", chunk)
                yield encode_sse(chunk)
            elif debug:
                logger.debug("chat stream ignored data=%s", data)
        _log_assistant_marker_leak(assistant_state, "openai-stream")
        _maybe_record_assistant(on_assistant, assistant_state)
    except CodebuffError as error:
        logger.warning(
            "chat stream failed run_id=%s: %s",
            run.run_id,
            error,
            exc_info=debug,
        )
        if error.status_code in {401, 403, 429} and on_rate_limited is not None:
            on_rate_limited()
        yield encode_sse(
            {
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            }
        )
        yield encode_sse("[DONE]")
    finally:
        schedule_finalize_run(client, run, message_id)
        if account_lease is not None:
            await account_lease.aclose()


async def collect_completion(
    client: CodebuffClient,
    payload: dict[str, Any],
    run: FreebuffRun,
    model: str,
    *,
    debug: bool = False,
    log_body_chars: int = 2000,
    recover: Any | None = None,
) -> dict[str, Any]:
    message_id: str | None = None
    accumulator = CompletionAccumulator(model)
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
                break
            message_id = data.get("id") or message_id
            accumulator.add(data)
        response = accumulator.final_response()
        logger.info(
            "chat completion response run_id=%s message_id=%s content_chars=%s finish_reason=%s",
            run.run_id,
            message_id,
            len(response["choices"][0]["message"].get("content") or ""),
            response["choices"][0].get("finish_reason"),
        )
        return response
    finally:
        await finalize_run(client, run, message_id)


def new_assistant_state() -> dict[str, Any]:
    """Trạng thái tích luỹ assistant: text, thinking, tool_calls."""
    return {"text": [], "thinking": [], "tool_calls": {}}


def _accumulate_assistant_parts(
    chunk: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Gom content + reasoning_content + tool_calls từ các chunk stream."""
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            state["text"].append(content)
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str):
            state["thinking"].append(reasoning)
        for tool_call in delta.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            index = int(tool_call.get("index") or 0)
            current = state["tool_calls"].setdefault(
                index,
                {
                    "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": tool_call.get("type") or "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tool_call.get("id"):
                current["id"] = tool_call["id"]
            if tool_call.get("type"):
                current["type"] = tool_call["type"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                current["function"]["name"] = function["name"]
            if function.get("arguments"):
                current["function"]["arguments"] += function["arguments"]


def _log_assistant_marker_leak(state: dict[str, Any], where: str) -> None:
    """Log when accumulated assistant text still contains tool-call markers.

    This is the leak path: a client (manicode/Claude fork) taught the model a
    tool protocol (DSML/XML) that the gateway did not parse, so the markers are
    streamed to the client as raw text and the client hangs waiting for a tool
    result. Logging the exact text lets us diagnose which protocol leaked.
    """
    text = "".join(state.get("text") or [])
    markers = detect_tool_markers(text)
    if markers:
        logger.warning(
            "ASSISTANT LEAK in %s: tool markers=%s chars=%s body=%s",
            where,
            markers,
            len(text),
            render_debug(text, 2000),
        )


def _maybe_record_assistant(
    on_assistant: Any,
    state: dict[str, Any],
) -> None:
    """Gọi callback với payload {"text", "thinking"?, "tool_calls"?}."""
    if on_assistant is None:
        return
    text = "".join(state["text"])
    if not text.strip() and not state["tool_calls"]:
        return
    payload: dict[str, Any] = {"text": text}
    thinking = "".join(state["thinking"])
    if thinking:
        payload["thinking"] = thinking
    if state["tool_calls"]:
        payload["tool_calls"] = [
            state["tool_calls"][index] for index in sorted(state["tool_calls"])
        ]
    try:
        on_assistant(payload)
    except Exception:
        logger.exception("failed to record assistant chat history")


def build_payload(
    body: dict[str, Any],
    *,
    session: Any,
    run: FreebuffRun,
    client_id: str,
    upstream_model_id: str | None,
    max_tokens_cap: int | None = None,
) -> dict[str, Any]:
    trace_session_id = str(uuid.uuid4())
    return build_upstream_payload(
        {**body, "messages": normalize_chat_messages(body.get("messages"))},
        session=session,
        run_id=run.payload_run_id,
        client_id=client_id,
        trace_session_id=trace_session_id,
        upstream_model_id=upstream_model_id,
        max_tokens_cap=max_tokens_cap,
    )


def build_session_recover_callback(
    lease: Any,
    model_config: Any,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    settings: Any,
    run: FreebuffRun,
) -> Any:
    """Build the 428 waiting_room_required recovery callback for a route.

    Returns an async callable that refreshes the freebuff session on the held
    lease and rebuilds the upstream payload with the fresh instance id, so the
    chat can be retried once. Shared by the OpenAI and Anthropic routes.
    """

    async def _recover_payload() -> dict[str, Any]:
        session = await lease.refresh_session(model_config.session_id)
        return build_payload(
            {**body, "messages": messages},
            session=session,
            run=run,
            client_id=settings.client_id,
            upstream_model_id=model_config.upstream_id,
            max_tokens_cap=settings.max_tokens,
        )

    return _recover_payload


def tool_history_to_text_protocol(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert client tool round-trips into the upstream text protocol.

    When the client executes a tool natively (Claude Code / Cline tool_use), it
    sends the result back as an OpenAI ``role: tool`` message paired with the
    assistant ``tool_calls``. The upstream free model does not understand native
    tool messages (tools are stripped), so this rewrites them to the text
    protocol it was taught: assistant text (no tool_calls) followed by a user
    message ``[tool result for <name> <arguments>]:\n<output>``.
    """
    result: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                pending_calls[call.get("id")] = {
                    "name": function.get("name") or "tool",
                    "arguments": function.get("arguments") or "",
                }
            msg = {
                key: value for key, value in message.items() if key != "tool_calls"
            }
            # If the assistant had tool_calls but no content, convert the
            # tool calls to text-protocol markers so the upstream model
            # understands the context (upstream rejects assistant messages
            # with neither ``content`` nor ``tool_calls``).
            content = msg.get("content")
            if not content and message.get("tool_calls"):
                parts: list[str] = []
                for call in message["tool_calls"]:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") or {}
                    name = function.get("name") or "tool"
                    raw_args = function.get("arguments") or ""
                    if isinstance(raw_args, str):
                        try:
                            parsed = json.loads(raw_args)
                        except (TypeError, ValueError):
                            parsed = {"_raw": raw_args}
                    else:
                        parsed = raw_args
                    parts.append(
                        f"<<<TOOL_CALL>>>{json.dumps({'name': name, 'arguments': parsed}, ensure_ascii=False)}<<<END_TOOL_CALL>>>"
                    )
                msg["content"] = "\n".join(parts)
            result.append(msg)
        elif role == "tool":
            info = pending_calls.get(message.get("tool_call_id")) or {
                "name": "tool",
                "arguments": "",
            }
            result.append(
                {
                    "role": "user",
                    "content": (
                        f"[tool result for {info['name']} {info['arguments']}]:\n"
                        f"{message.get('content') or ''}"
                    ),
                }
            )
        else:
            result.append(message)
    return result


def _compiler_execution_context(raw_messages: Any) -> str:
    """Keep recent client-executed tool state available to the private compiler.

    A tool schema tells the compiler that ``fileKey`` is required, but only the
    preceding native tool calls/results contain the actual Figma key and page
    IDs. This is request context only; it is never recorded as a new history
    event or exposed to the user.
    """
    if not isinstance(raw_messages, list):
        return ""
    entries: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    entries.append(
                        {
                            "kind": "tool_call",
                            "name": function["name"],
                            "arguments": function.get("arguments") or "{}",
                        }
                    )
        elif message.get("role") == "tool":
            entries.append(
                {
                    "kind": "tool_result",
                    "tool_call_id": message.get("tool_call_id"),
                    "content": message.get("content") or "",
                }
            )
    if not entries:
        return ""
    return json.dumps(entries[-16:], ensure_ascii=False)


def _compiler_task_context(raw_messages: Any) -> str:
    """Return the active client instructions needed to verify a final answer.

    The private compiler previously saw only the upstream draft. It could label
    "I will update the files" as final because it did not see the still-pending
    user/system task. Preserve recent non-tool instructions verbatim; this is
    request-scoped context, not a history write or a language heuristic.
    """
    if not isinstance(raw_messages, list):
        return ""
    entries: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, dict) or message.get("role") == "tool":
            continue
        role = message.get("role")
        if role not in {"user", "system", "developer"}:
            continue
        content = message.get("content")
        if content not in (None, ""):
            entries.append({"role": role, "content": content})
    return json.dumps(entries[-16:], ensure_ascii=False)


async def run_tool_loop_pass(
    client: CodebuffClient,
    payload: dict[str, Any],
    *,
    settings: Any,
    model: str,
    recover: Any | None = None,
    debug: bool = False,
    log_body_chars: int = 2000,
    client_tools: Any | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Run ONE upstream pass with the text-protocol tool prompt.

    Returns ``(response, client_call)``:
    - ``response``: OpenAI completion dict when the model answered without a
      tool call (``client_call`` is ``None``).
    - ``client_call``: ``{"name", "arguments"}`` expressed with CLIENT tool
      names when the model emitted a tool call — the caller then streams a
      native ``tool_use`` block so the client executes it itself.

    Unlike the old local agent loop, tools are NOT executed here; the client
    (Claude Code / Cline) shows its own approval UI, runs the tool on the host
    and sends the result back on the next request.
    """
    tool_instructions = tool_system_prompt(
        settings.tool_workdir,
        bash_enabled=settings.tool_bash_enabled,
    )
    client_tool_names = declared_client_tool_names(client_tools)
    if client_tool_names:
        tool_instructions += (
            "\n\nThe IDE client declared these executable tool names: "
            + ", ".join(client_tool_names)
            + ". Invoke ONLY one of those names. Do not use an alias that is not listed."
        )
    messages = tool_history_to_text_protocol(payload.get("messages") or [])
    execution_context = _compiler_execution_context(payload.get("messages"))
    task_context = _compiler_task_context(payload.get("messages"))
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        content = first.get("content") or ""
        if isinstance(content, str):
            first["content"] = content + "\n\n" + tool_instructions
        else:
            first["content"] = [
                *content,
                {"type": "text", "text": tool_instructions},
            ]
        messages[0] = first
    else:
        messages.insert(
            0,
            {
                "role": "system",
                "content": tool_instructions,
            },
        )

    async def _collect(
        active_messages: list[dict[str, Any]], *, compiler_mode: bool = False, phase: str = "main"
    ) -> tuple[
        CompletionAccumulator, str, str
    ]:
        active_payload = dict(payload)
        active_payload["messages"] = active_messages
        if compiler_mode:
            # This is an OpenAI structured-output field, not a native tool
            # field. FreeBuff still receives no `tools`, but compatible models
            # are constrained to the JSON compiler contract.
            active_payload["response_format"] = {"type": "json_object"}
            active_payload["temperature"] = 0

        async def _recover_with_context() -> dict[str, Any]:
            if recover is None:
                return active_payload
            fresh = await recover()
            if isinstance(fresh, dict):
                fresh = dict(fresh)
                fresh["messages"] = active_messages
                if compiler_mode:
                    fresh["response_format"] = {"type": "json_object"}
                    fresh["temperature"] = 0
                return fresh
            return active_payload

        accumulator = CompletionAccumulator(model)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        async for line in chat_events_with_recovery(
            client,
            active_payload,
            recover=_recover_with_context,
            debug=debug,
        ):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            accumulator.add(data)
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)

        collected_text = "".join(text_parts)
        collected_reasoning = "".join(reasoning_parts)
        # Deliberately raw and untruncated at the user's request. This is an
        # operational diagnostic switch for this local gateway: Docker logs
        # may contain model output, source excerpts and tool-related context.
        logger.warning(
            "tool pass phase=%s raw_response_text=<<<%s>>> raw_response_reasoning=<<<%s>>>",
            phase,
            _redact_raw_model_output(collected_text),
            _redact_raw_model_output(collected_reasoning),
        )
        return accumulator, collected_text, collected_reasoning

    accumulator, full_text, full_reasoning = await _collect(messages, phase="main")

    # FreeBuff has no native tool API. When a client supplied tools, never use
    # natural-language (or language-specific) heuristics to decide whether the
    # response was "planning". Instead, if the main pass did not already emit
    # a parseable native call, run exactly one private protocol compiler. It
    # must classify the draft as either TOOL_CALL or <<<FINAL>>>.
    direct_call, _ = client_tool_call(full_text)
    if direct_call is None:
        direct_call, _ = client_tool_call(full_reasoning)
    compiler_required = isinstance(client_tools, list) and bool(client_tools) and direct_call is None
    compiler_final = False
    compiler_emitted_call = False
    if compiler_required:
        tool_contract = ""
        if isinstance(client_tools, list) and client_tools:
            try:
                # This is the client-side tool contract, not user source text.
                # Do not truncate it: a later MCP entry (for example Figma)
                # must remain available to the private compiler pass.
                tool_contract = json.dumps(client_tools, ensure_ascii=False)
            except (TypeError, ValueError):
                tool_contract = ""
        compiler_instruction = (
            "\n\nYou are a tool-call compiler, not a chat assistant. Convert the "
            "draft below into exactly one JSON object, with no Markdown, thinking, "
            "or explanation. Output either {\\\"action\\\":\\\"tool_call\\\",\\\"name\\\":\\\"...\\\","
            "\\\"arguments\\\":{...}} or {\\\"action\\\":\\\"final\\\"}. Use final only "
            "when the draft is already a complete answer requiring no tool. Use only a "
            "tool and arguments justified by the draft. Do not invent paths, URLs, "
            "or values. If no safe concrete invocation can be made, output nothing."
        )
        compiler_system = dict(messages[0])
        base_content = compiler_system.get("content") or ""
        compiler_system["content"] = (
            base_content + compiler_instruction
            if isinstance(base_content, str)
            else [
                *base_content,
                {"type": "text", "text": compiler_instruction},
            ]
        )
        draft = (full_reasoning + "\n" + full_text).strip()[:16000]
        compiler_messages = [
            compiler_system,
            {
                "role": "user",
                "content": (
                    "Draft to compile (do not answer it):\n" + draft
                    + ("\n\nClient-declared tools/schema:\n" + tool_contract if tool_contract else "")
                    + ("\n\nRecent client tool state (reuse concrete non-empty arguments when applicable):\n" + execution_context if execution_context else "")
                    + ("\n\nActive client task instructions. Return action=final ONLY if the draft fully satisfies these; otherwise select the next declared tool:\n" + task_context if task_context else "")
                ),
            },
        ]
        logger.info(
            "tool pass phase=compiler_started draft_chars=%s client_tools=%s",
            len(draft),
            len(client_tools) if isinstance(client_tools, list) else 0,
        )
        compiler_accumulator, compiler_text, compiler_reasoning = await _collect(
            compiler_messages, compiler_mode=True, phase="compiler"
        )
        compiled_call, compiler_final = parse_compiler_protocol(compiler_text)
        if compiled_call is None and not compiler_final:
            compiled_call, compiler_final = parse_compiler_protocol(compiler_reasoning)
        if compiled_call is None and not compiler_final:
            # ``response_format=json_object`` is only best-effort on FreeBuff;
            # it is not native function calling.  Give the compiler exactly
            # one deterministic repair chance rather than exposing its first
            # formatting slip as an interruption in Claude/Codex.  The repair
            # is private, has the same schema/task context, and is never
            # written to chat history.
            malformed = (compiler_text or compiler_reasoning).strip()
            repair_instruction = (
                "Your previous compiler output was invalid. Return ONLY one JSON "
                "object now: either {\"action\":\"tool_call\",\"name\":\"DECLARED_TOOL\","
                "\"arguments\":{...}} or exactly {\"action\":\"final\"}. "
                "Do not include prose, Markdown, reasoning, a plan, or a flat "
                "object. Copy tool arguments only from the supplied task/context."
            )
            repair_messages = [
                compiler_system,
                {
                    "role": "user",
                    "content": (
                        repair_instruction
                        + "\n\nInvalid previous output:\n"
                        + (malformed or "<empty>")
                        + "\n\nOriginal compiler input:\n"
                        + compiler_messages[1]["content"]
                    ),
                },
            ]
            logger.warning(
                "tool pass phase=compiler_repair_started invalid_chars=%s",
                len(malformed),
            )
            repair_accumulator, repair_text, repair_reasoning = await _collect(
                repair_messages, compiler_mode=True, phase="compiler_repair"
            )
            repaired_call, repaired_final = parse_compiler_protocol(repair_text)
            if repaired_call is None and not repaired_final:
                repaired_call, repaired_final = parse_compiler_protocol(repair_reasoning)
            if repaired_call is not None or repaired_final:
                compiler_accumulator = repair_accumulator
                compiler_text = repair_text
                compiler_reasoning = repair_reasoning
                compiled_call = repaired_call
                compiler_final = repaired_final
                logger.info(
                    "tool pass phase=compiler_repair_completed classification=%s",
                    "tool" if repaired_call is not None else "final",
                )
            else:
                logger.warning(
                    "tool pass phase=compiler_repair_failed response_chars=%s",
                    len(repair_text) + len(repair_reasoning),
                )
        if compiled_call is not None:
            compiler_emitted_call = True
            logger.info(
                "tool pass phase=compiler_completed tool=%s args=%s",
                compiled_call["name"], compiled_call.get("arguments"),
            )
            accumulator, full_text, full_reasoning = (
                compiler_accumulator,
                compiler_text,
                compiler_reasoning,
            )
        elif compiler_final:
            logger.info("tool pass phase=compiler_completed classification=final")
        else:
            logger.warning(
                "tool pass phase=compiler_no_valid_call draft_chars=%s compiler_chars=%s",
                len(draft), len(compiler_text) + len(compiler_reasoning),
            )

    if (
        compiler_required
        and not compiler_final
        and not compiler_emitted_call
        and direct_call is None
    ):
        # Do not silently turn an unclassified tool-bearing turn into an
        # end_turn. That was the original source of apparent Claude interrupts.
        # Returning a protocol error makes the state observable and retryable.
        call_after_compiler, _ = client_tool_call(full_text)
        if call_after_compiler is None:
            call_after_compiler, _ = client_tool_call(full_reasoning)
        if call_after_compiler is None:
            response = accumulator.final_response()
            response["choices"][0]["message"]["content"] = (
                "Gateway tool-protocol error: the upstream response could not be "
                "classified as a declared tool call or final answer. Please retry."
            )
            response["choices"][0]["finish_reason"] = "stop"
            return response, None

    # DeepSeek sometimes puts the complete XML/DSML invocation in its
    # reasoning channel. It is still a real tool call, but Claude would render
    # that channel as visible thinking unless we normalize it here too.
    call, clean_text = client_tool_call(full_text)
    if call is None and compiler_emitted_call:
        call, _ = parse_compiler_protocol(full_text)
        clean_text = ""
    call_from_reasoning = False
    clean_reasoning = full_reasoning
    if call is None:
        call, clean_reasoning = client_tool_call(full_reasoning)
        if call is None and compiler_emitted_call:
            call, _ = parse_compiler_protocol(full_reasoning)
            clean_reasoning = ""
        call_from_reasoning = call is not None
    if call is not None:
        adapted_call = adapt_client_tool_call(call, client_tools)
        if adapted_call != call:
            logger.info(
                "tool pass phase=adapted_client_tool from=%s to=%s",
                call.get("name"), adapted_call.get("name"),
            )
            call = adapted_call
        call = coerce_client_tool_call_arguments(call, client_tools)
        valid, reason = validate_client_tool_call(call, client_tools)
        if not valid:
            logger.warning(
                "tool pass phase=rejected_client_tool tool=%s reason=%s",
                call.get("name"), reason,
            )
            response = accumulator.final_response()
            response["choices"][0]["message"]["content"] = (
                "Gateway rejected an invalid tool request: " + reason
            )
            response["choices"][0]["message"].pop("tool_calls", None)
            response["choices"][0]["finish_reason"] = "stop"
            return response, None
        logger.info(
            "tool pass tool=%s args=%s clean_chars=%s body=%s",
            call["name"],
            call.get("arguments"),
            len(clean_text),
            render_debug(full_text, log_body_chars),
        )
        response = accumulator.final_response()
        message = response["choices"][0]["message"]
        if call_from_reasoning:
            message["reasoning_content"] = clean_reasoning or None
        else:
            message["content"] = clean_text or None
        message["tool_calls"] = [
            {
                "index": 0,
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments") or {},
                        ensure_ascii=False,
                    ),
                },
            }
        ]
        response["choices"][0]["finish_reason"] = "tool_calls"
        return response, call

    if detect_tool_markers(full_text):
        logger.warning(
            "tool pass FINAL ANSWER still contains markers — leaking to client; chars=%s body=%s",
            len(full_text),
            render_debug(full_text, log_body_chars),
        )
    return accumulator.final_response(), None


async def run_tool_agent_loop(
    client: CodebuffClient,
    payload: dict[str, Any],
    *,
    body: dict[str, Any],
    settings: Any,
    model: str,
    recover: Any | None = None,
    debug: bool = False,
    log_body_chars: int = 2000,
    approval: Any | None = None,
    project_key: str = "",
) -> dict[str, Any]:
    """Run the local tool-execution agent loop against the freebuff upstream.

    The upstream free model rejects requests carrying ``tools``, so the gateway
    strips them (in ``build_upstream_payload``) and instead drives the model via
    the ``<<<TOOL_CALL>>>`` text protocol: the model emits a tool call, the
    gateway executes it locally (read_file / bash / ...) and appends the result
    as a new user message, then loops until the model answers without a tool
    call or the iteration budget is exhausted.

    Returns a complete OpenAI completion dict built from the final accumulated
    text (including the tool-call markers removed).
    """
    # The upstream requires the FIRST system message to open with the canonical
    # Buffy identity (403 free_mode_cli_required otherwise), so the tool
    # instructions must be appended to that message — never inserted before it.
    tool_instructions = tool_system_prompt(
        settings.tool_workdir,
        bash_enabled=settings.tool_bash_enabled,
    )
    messages = list(payload.get("messages") or [])
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        content = first.get("content") or ""
        if isinstance(content, str):
            first["content"] = content + "\n\n" + tool_instructions
        else:
            first["content"] = [
                *content,
                {"type": "text", "text": tool_instructions},
            ]
        messages[0] = first
    else:
        messages.insert(
            0,
            {
                "role": "system",
                "content": tool_instructions,
            },
        )

    max_iterations = max(1, settings.tool_max_iterations)
    final_accumulator: CompletionAccumulator | None = None
    for iteration in range(max_iterations):
        loop_payload = dict(payload)
        loop_payload["messages"] = messages
        accumulator = CompletionAccumulator(model)
        text_parts: list[str] = []

        async def _recover_with_context() -> dict[str, Any]:
            # The route's recover refreshes the freebuff session and rebuilds the
            # payload from the ORIGINAL body. The tool loop has since mutated the
            # conversation (tool prompt + tool-result messages), so overlay the
            # current messages onto the recovered payload to keep loop context.
            if recover is None:
                return loop_payload
            fresh = await recover()
            if isinstance(fresh, dict):
                fresh = dict(fresh)
                fresh["messages"] = messages
                return fresh
            return loop_payload

        async for line in chat_events_with_recovery(
            client,
            loop_payload,
            recover=_recover_with_context,
            debug=debug,
        ):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            accumulator.add(data)
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    text_parts.append(content)

        full_text = "".join(text_parts)
        markers = detect_tool_markers(full_text)
        if markers:
            logger.warning(
                "tool loop iteration=%s raw text contains tool markers=%s chars=%s body=%s",
                iteration + 1,
                markers,
                len(full_text),
                render_debug(full_text, log_body_chars),
            )
        call, clean_text = parse_tool_call(full_text)
        if call is None:
            # Final answer — no more tool calls. Return the accumulated response.
            if markers:
                logger.warning(
                    "tool loop iteration=%s FINAL ANSWER still contains markers=%s — leaking to client; chars=%s body=%s",
                    iteration + 1,
                    markers,
                    len(full_text),
                    render_debug(full_text, log_body_chars),
                )
            else:
                logger.info(
                    "tool loop iteration=%s final answer (no tool call) chars=%s",
                    iteration + 1,
                    len(full_text),
                )
            return accumulator.final_response()
        logger.info(
            "tool loop iteration=%s tool=%s args=%s clean_chars=%s",
            iteration + 1,
            call["name"],
            call.get("arguments"),
            len(clean_text),
        )
        # Hướng B: tool approval — pause for the dashboard user when the tool
        # mode is "ask". ``None`` means the tool may run.
        verdict: str | None = None
        if approval is not None:
            verdict = await approval.request(
                call["name"],
                call.get("arguments") or {},
                settings.tool_workdir,
            )
        if verdict:
            result = verdict
        else:
            result = await execute_tool(
                call,
                settings.tool_workdir,
                bash_enabled=settings.tool_bash_enabled,
                command_timeout=settings.tool_command_timeout,
                output_cap=settings.tool_output_cap,
                file_cap=settings.tool_file_cap,
                project_key=project_key,
            )
        if debug:
            logger.debug(
                "tool result tool=%s chars=%s body=%s",
                call["name"],
                len(result),
                render_debug(result, log_body_chars),
            )
        if clean_text:
            messages.append({"role": "assistant", "content": clean_text})
        messages.append(
            {
                "role": "user",
                "content": f"[tool result for {call['name']} {call.get('arguments')}]:\n{result}",
            }
        )
        final_accumulator = accumulator

    logger.warning(
        "tool agent loop exhausted after %s iterations; returning last text",
        max_iterations,
    )
    assert final_accumulator is not None
    return final_accumulator.final_response()


async def stream_tool_agent_loop(
    client: CodebuffClient,
    payload: dict[str, Any],
    *,
    settings: Any,
    model: str,
    recover: Any | None = None,
    debug: bool = False,
    log_body_chars: int = 2000,
    account_lease: Any | None = None,
    on_assistant: Any | None = None,
    client_tools: Any | None = None,
) -> AsyncIterator[bytes]:
    """Run ONE native tool pass and stream OpenAI SSE to the client.

    Emits ``: ping`` keep-alive comments while the upstream pass runs, then
    streams the final answer (or a native ``tool_calls`` delta) so the client
    (Cline / Codex / OpenAI-compatible) executes the tool itself.
    """
    from gateway.core.sse import encode_sse

    task = asyncio.create_task(
        run_tool_loop_pass(
            client,
            payload,
            settings=settings,
            model=model,
            recover=recover,
            debug=debug,
            log_body_chars=log_body_chars,
            client_tools=client_tools,
        )
    )
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=15.0)
            if done:
                break
            yield b": ping\n\n"
        response, client_call = task.result()
    except CodebuffError as error:
        if error.status_code in {401, 403, 429} and account_lease is not None:
            account_lease.mark_rate_limited(settings.account_cooldown)
        yield encode_sse(
            {
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            }
        )
        yield encode_sse("[DONE]")
        return
    finally:
        if not task.done():
            task.cancel()
        if account_lease is not None:
            await account_lease.aclose()

    assert response is not None
    message = (response.get("choices") or [{}])[0].get("message") or {}
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
            logger.exception("failed to record openai tool pass assistant")

    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls") or []
    created = response.get("created") or int(time.time())
    chunk_id = response.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
    finish_reason = (
        "tool_calls"
        if client_call is not None
        else (response.get("choices") or [{}])[0].get("finish_reason") or "stop"
    )
    if reasoning:
        yield encode_sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": reasoning},
                        "finish_reason": None,
                    }
                ],
            }
        )
    if content:
        yield encode_sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
        )
    if client_call is not None:
        yield encode_sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
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
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
    yield encode_sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": response.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )
    yield encode_sse("[DONE]")
