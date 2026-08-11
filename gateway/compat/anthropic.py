from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _tool_call_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _text_from_content(content: Any) -> str:
    """
    Convert Anthropic/OpenAI content into plain text where appropriate.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []

    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)

        elif item_type == "thinking":
            thinking = item.get("thinking")
            if isinstance(thinking, str):
                parts.append(thinking)

    return "\n".join(part for part in parts if part)


def _normalize_system(system: Any) -> str:
    """
    Anthropic accepts system as either a string or an array of content blocks.
    OpenAI expects a system message.
    """
    return _text_from_content(system)


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    """
    Anthropic:
    {
        "name": "read_file",
        "description": "...",
        "input_schema": {...}
    }

    OpenAI:
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "...",
            "parameters": {...}
        }
    }
    """
    if not isinstance(tools, list):
        return []

    result: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue

        function: dict[str, Any] = {
            "name": name,
            "parameters": tool.get("input_schema")
            if isinstance(tool.get("input_schema"), dict)
            else {
                "type": "object",
                "properties": {},
            },
        }

        description = tool.get("description")
        if isinstance(description, str) and description:
            function["description"] = description

        result.append(
            {
                "type": "function",
                "function": function,
            }
        )

    return result


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    """
    Anthropic tool_choice examples:

    {"type": "auto"}
    {"type": "any"}
    {"type": "tool", "name": "read_file"}
    {"type": "none"}
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return "auto"

    if not isinstance(tool_choice, dict):
        return None

    choice_type = tool_choice.get("type")

    if choice_type == "auto":
        return "auto"

    if choice_type == "none":
        return "none"

    if choice_type == "any":
        return "required"

    if choice_type == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {
                "type": "function",
                "function": {
                    "name": name,
                },
            }

    return "auto"


def _anthropic_user_content_to_openai(
    content: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convert one Anthropic user message.

    Returns:
    - ordinary OpenAI user content blocks
    - OpenAI tool result messages
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}], []

    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}], []

    user_parts: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, str):
            user_parts.append(
                {
                    "type": "text",
                    "text": block,
                }
            )
            continue

        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                user_parts.append(
                    {
                        "type": "text",
                        "text": text,
                    }
                )
            continue

        if block_type == "image":
            source = block.get("source") or {}
            source_type = source.get("type")

            if source_type == "base64":
                media_type = source.get("media_type") or "image/png"
                data = source.get("data")

                if isinstance(data, str):
                    user_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}",
                            },
                        }
                    )

            elif source_type == "url":
                url = source.get("url")
                if isinstance(url, str):
                    user_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                            },
                        }
                    )

            continue

        if block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")

            if not isinstance(tool_use_id, str) or not tool_use_id:
                continue

            result_content = block.get("content", "")

            # MCP tools (notably Chrome DevTools) may return screenshots inside
            # a tool_result. Preserve them as normal multimodal user parts so
            # the upstream vision model receives pixels rather than a JSON
            # string containing base64 data.
            if isinstance(result_content, list):
                for part in result_content:
                    if not isinstance(part, dict) or part.get("type") != "image":
                        continue
                    source = part.get("source") or {}
                    if source.get("type") == "base64":
                        media_type = source.get("media_type") or "image/png"
                        data = source.get("data")
                        if isinstance(data, str):
                            user_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{data}",
                                    },
                                }
                            )
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        user_parts.append(
                            {"type": "image_url", "image_url": {"url": source["url"]}}
                        )

            if isinstance(result_content, str):
                rendered_content = result_content
            else:
                rendered_content = _text_from_content(result_content)

                if not rendered_content:
                    try:
                        rendered_content = json.dumps(
                            result_content,
                            ensure_ascii=False,
                        )
                    except (TypeError, ValueError):
                        rendered_content = str(result_content)

            if isinstance(result_content, list) and any(
                isinstance(part, dict) and part.get("type") == "image"
                for part in result_content
            ):
                rendered_content = (rendered_content + "\n[Tool result includes image input above.]").strip()

            rendered_content = _explain_tool_timeout(rendered_content)

            if block.get("is_error") is True:
                rendered_content = f"Tool execution error:\n{rendered_content}"

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": rendered_content,
                }
            )

    return user_parts, tool_messages


_CLIENT_TOOL_TIMEOUT_RE = re.compile(
    r"(?:did not complete within (?:its )?|timed out after )(?P<seconds>\d+(?:\.\d+)?)\s*(?:s|sec(?:onds?)?)\b",
    re.IGNORECASE,
)


def _explain_tool_timeout(content: str) -> str:
    """Make a host-client tool timeout intelligible in the Claude transcript.

    Claude Code executes native tools itself, outside the gateway. It only sends
    the result once its own timeout expires, so we cannot observe a macOS
    permission dialog while it is pending. A clear result prevents the model
    and user from mistaking a timeout for either a completed command or a
    gateway/network interruption.
    """
    match = _CLIENT_TOOL_TIMEOUT_RE.search(content or "")
    if match is None:
        return content
    seconds = match.group("seconds")
    return (
        f"[Tool timeout — command stopped after {seconds}s]\n"
        "The command did not finish. This is not a gateway interruption and does "
        "not confirm that the operation succeeded. If this command opened a macOS "
        "permission dialog (for example Photos/Automation), allow or deny it, then "
        "press Continue to retry or inspect the result.\n\n"
        f"Original tool output:\n{content}"
    )


def _anthropic_assistant_to_openai(content: Any) -> dict[str, Any]:
    """
    Convert an Anthropic assistant message into an OpenAI assistant message.
    """
    if isinstance(content, str):
        return {
            "role": "assistant",
            "content": content,
        }

    if not isinstance(content, list):
        return {
            "role": "assistant",
            "content": str(content),
        }

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)

        elif block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str):
                reasoning_parts.append(thinking)

        elif block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            tool_input = block.get("input", {})

            if not isinstance(tool_id, str) or not tool_id:
                tool_id = _tool_call_id()

            if not isinstance(name, str) or not name:
                continue

            try:
                arguments = json.dumps(
                    tool_input,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                arguments = "{}"

            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }

    if reasoning_parts:
        message["reasoning_content"] = "\n".join(reasoning_parts)

    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


def anthropic_messages_to_openai(
    system: Any,
    messages: Any,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    system_text = _normalize_system(system)

    if system_text:
        result.append(
            {
                "role": "system",
                "content": system_text,
            }
        )

    if not isinstance(messages, list):
        return result

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")

        if role == "assistant":
            result.append(_anthropic_assistant_to_openai(content))
            continue

        if role != "user":
            continue

        user_parts, tool_messages = _anthropic_user_content_to_openai(content)

        if user_parts:
            if len(user_parts) == 1 and user_parts[0].get("type") == "text":
                user_content: Any = user_parts[0].get("text", "")
            else:
                user_content = user_parts

            result.append(
                {
                    "role": "user",
                    "content": user_content,
                }
            )

        result.extend(tool_messages)

    return result


def anthropic_request_to_openai(
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert Anthropic Messages API request to OpenAI Chat Completions request.
    """
    result: dict[str, Any] = {
        "model": body.get("model"),
        "messages": anthropic_messages_to_openai(
            body.get("system"),
            body.get("messages"),
        ),
        "stream": body.get("stream") is True,
    }

    max_tokens = body.get("max_tokens")

    if isinstance(max_tokens, int):
        result["max_tokens"] = max_tokens

    for key in (
        "temperature",
        "top_p",
        "metadata",
    ):
        if body.get(key) is not None:
            result[key] = body[key]

    stop_sequences = body.get("stop_sequences")

    if isinstance(stop_sequences, list) and stop_sequences:
        result["stop"] = stop_sequences

    tools = anthropic_tools_to_openai(body.get("tools"))

    if tools:
        result["tools"] = tools

    tool_choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))

    if tool_choice is not None:
        result["tool_choice"] = tool_choice

    thinking = body.get("thinking")

    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")

        if thinking_type == "enabled":
            result["reasoning_effort"] = "high"
        elif thinking_type == "disabled":
            result["reasoning_effort"] = "none"

    return result


def _parse_tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments

    if not isinstance(arguments, str):
        return {}

    if not arguments.strip():
        return {}

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {
            "_raw": arguments,
        }


def openai_finish_reason_to_anthropic(reason: Any) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }

    if isinstance(reason, str):
        return mapping.get(reason, "end_turn")

    return "end_turn"


def openai_response_to_anthropic(
    response: dict[str, Any],
    requested_model: str | None = None,
) -> dict[str, Any]:
    """
    Convert a non-stream OpenAI Chat Completions response to Anthropic format.
    """
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content_blocks: list[dict[str, Any]] = []

    reasoning_content = message.get("reasoning_content")

    if isinstance(reasoning_content, str) and reasoning_content:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning_content,
                "signature": "",
            }
        )

    content = message.get("content")

    if isinstance(content, str) and content:
        content_blocks.append(
            {
                "type": "text",
                "text": content,
            }
        )

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get("function") or {}
        name = function.get("name")

        if not isinstance(name, str) or not name:
            continue

        tool_call_id = tool_call.get("id")

        if not isinstance(tool_call_id, str) or not tool_call_id:
            tool_call_id = _tool_call_id()

        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call_id,
                "name": name,
                "input": _parse_tool_arguments(function.get("arguments")),
            }
        )

    if not content_blocks:
        content_blocks.append(
            {
                "type": "text",
                "text": "",
            }
        )

    usage = response.get("usage") or {}

    return {
        "id": response.get("id") or _message_id(),
        "type": "message",
        "role": "assistant",
        "model": requested_model or response.get("model") or "",
        "content": content_blocks,
        "stop_reason": openai_finish_reason_to_anthropic(
            choice.get("finish_reason")
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_creation_input_tokens": int(
                usage.get("cache_creation_input_tokens") or 0
            ),
            "cache_read_input_tokens": int(
                usage.get("cache_read_input_tokens") or 0
            ),
        },
    }


def anthropic_error_response(
    message: str,
    *,
    error_type: str = "api_error",
) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def encode_anthropic_sse(
    event: str,
    data: dict[str, Any],
) -> bytes:
    rendered = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"event: {event}\ndata: {rendered}\n\n".encode("utf-8")


@dataclass
class AnthropicStreamState:
    """
    State used while converting OpenAI streaming chunks into Anthropic SSE.
    """

    message_id: str = field(default_factory=_message_id)
    model: str = ""
    started: bool = False
    stopped: bool = False

    next_block_index: int = 0
    text_block_index: int | None = None
    thinking_block_index: int | None = None

    active_block_indexes: set[int] = field(default_factory=set)

    tool_blocks: dict[int, int] = field(default_factory=dict)
    tool_ids: dict[int, str] = field(default_factory=dict)
    tool_names: dict[int, str] = field(default_factory=dict)

    input_tokens: int = 0
    output_tokens: int = 0

    finish_reason: str | None = None

    def allocate_block(self) -> int:
        index = self.next_block_index
        self.next_block_index += 1
        self.active_block_indexes.add(index)
        return index


def _message_start_event(
    state: AnthropicStreamState,
) -> tuple[str, dict[str, Any]]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": state.message_id,
                "type": "message",
                "role": "assistant",
                "model": state.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": state.input_tokens,
                    "output_tokens": 0,
                },
            },
        },
    )


def _content_block_start_text(
    index: int,
) -> tuple[str, dict[str, Any]]:
    return (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "text",
                "text": "",
            },
        },
    )


def _content_block_start_thinking(
    index: int,
) -> tuple[str, dict[str, Any]]:
    return (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "",
            },
        },
    )


def _content_block_start_tool(
    index: int,
    tool_id: str,
    name: str,
) -> tuple[str, dict[str, Any]]:
    return (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        },
    )


def _content_block_stop(
    index: int,
) -> tuple[str, dict[str, Any]]:
    return (
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


def _close_active_blocks(
    state: AnthropicStreamState,
) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    for index in sorted(state.active_block_indexes):
        events.append(_content_block_stop(index))

    state.active_block_indexes.clear()
    return events


def openai_chunk_to_anthropic_events(
    chunk: dict[str, Any],
    state: AnthropicStreamState,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Convert one OpenAI streaming chunk into zero or more Anthropic SSE events.

    Return format:
        [
            ("message_start", {...}),
            ("content_block_delta", {...}),
            ...
        ]
    """
    events: list[tuple[str, dict[str, Any]]] = []

    chunk_id = chunk.get("id")

    if isinstance(chunk_id, str) and chunk_id:
        if chunk_id.startswith("msg_"):
            state.message_id = chunk_id

    chunk_model = chunk.get("model")

    if isinstance(chunk_model, str) and chunk_model:
        state.model = chunk_model

    usage = chunk.get("usage")

    if isinstance(usage, dict):
        state.input_tokens = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or state.input_tokens
            or 0
        )
        state.output_tokens = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or state.output_tokens
            or 0
        )

    if not state.started:
        state.started = True
        events.append(_message_start_event(state))

    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue

        delta = choice.get("delta") or {}

        reasoning_content = delta.get("reasoning_content")

        if isinstance(reasoning_content, str) and reasoning_content:
            if state.thinking_block_index is None:
                state.thinking_block_index = state.allocate_block()
                events.append(
                    _content_block_start_thinking(
                        state.thinking_block_index
                    )
                )

            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": state.thinking_block_index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": reasoning_content,
                        },
                    },
                )
            )

        content = delta.get("content")

        if isinstance(content, str) and content:
            if state.text_block_index is None:
                state.text_block_index = state.allocate_block()
                events.append(
                    _content_block_start_text(
                        state.text_block_index
                    )
                )

            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": state.text_block_index,
                        "delta": {
                            "type": "text_delta",
                            "text": content,
                        },
                    },
                )
            )

        for tool_call in delta.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue

            tool_position = int(tool_call.get("index") or 0)
            function = tool_call.get("function") or {}

            tool_id = tool_call.get("id")

            if isinstance(tool_id, str) and tool_id:
                state.tool_ids[tool_position] = tool_id

            name = function.get("name")

            if isinstance(name, str) and name:
                state.tool_names[tool_position] = name

            if tool_position not in state.tool_blocks:
                block_index = state.allocate_block()
                state.tool_blocks[tool_position] = block_index

                resolved_tool_id = state.tool_ids.get(
                    tool_position,
                    _tool_call_id(),
                )
                resolved_name = state.tool_names.get(
                    tool_position,
                    "",
                )

                state.tool_ids[tool_position] = resolved_tool_id

                events.append(
                    _content_block_start_tool(
                        block_index,
                        resolved_tool_id,
                        resolved_name,
                    )
                )

            arguments = function.get("arguments")

            if isinstance(arguments, str) and arguments:
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state.tool_blocks[tool_position],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        },
                    )
                )

        finish_reason = choice.get("finish_reason")

        if isinstance(finish_reason, str) and finish_reason:
            state.finish_reason = finish_reason

    return events


def finish_anthropic_stream(
    state: AnthropicStreamState,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Emit final Anthropic events after OpenAI sends [DONE].
    """
    if state.stopped:
        return []

    state.stopped = True

    events = _close_active_blocks(state)

    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": openai_finish_reason_to_anthropic(
                        state.finish_reason
                    ),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": state.output_tokens,
                },
            },
        )
    )

    events.append(
        (
            "message_stop",
            {
                "type": "message_stop",
            },
        )
    )

    return events
